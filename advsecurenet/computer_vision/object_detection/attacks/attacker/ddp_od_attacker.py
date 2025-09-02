import logging
import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from advsecurenet.distributed.ddp_base_task import DDPBaseTask

logger = logging.getLogger(__name__)

class DDPODAttacker(DDPBaseTask):
    """Lightweight DDP wrapper for object detection attackers.

    We only shard the dataset via DistributedSampler; the underlying model is
    left as-is (no gradient sync needed for evaluation-phase attacks)."""

    TEMP_DIR = ".adv_od_tmp"  # simple deterministic temp dir (can override via ADV_OD_TMP env)

    def __init__(self, attacker_class, config, **kwargs):
        # For parity with existing DDPBaseTask signature we pass model=None; we don't wrap model.
        DDPBaseTask.__init__(self, model=config.model, rank=dist.get_rank(), world_size=dist.get_world_size())
        self.attacker_class = attacker_class
        self.config = config
        self.kwargs = kwargs
        self.attacker = None
        self._sampler = None

    def setup(self):
        self.attacker = self.attacker_class(self.config, **self.kwargs)
        dl = getattr(self.attacker, "_dataloader", None)
        if isinstance(dl, torch.utils.data.DataLoader):
            # Respect original shuffle flag from config if available
            shuffle_flag = getattr(getattr(self.config, "dataloader", object()), "shuffle", False)
            sampler = DistributedSampler(
                dl.dataset,
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=bool(shuffle_flag),
            )
            self.attacker._dataloader = torch.utils.data.DataLoader(
                dl.dataset,
                batch_size=dl.batch_size,
                sampler=sampler,
                num_workers=dl.num_workers,
                pin_memory=getattr(dl, "pin_memory", False),
                collate_fn=getattr(dl, "collate_fn", None),
                drop_last=getattr(dl, "drop_last", False),
            )
            self._sampler = sampler

    def run_task(self):
        shard_indices = None
        if self._sampler is not None:
            # epoch fixed (0) for reproducibility; if shuffle True sampler uses this epoch seed
            self._sampler.set_epoch(0)
            try:
                # Capture the exact order this rank will iterate (works for shuffle True/False)
                shard_indices = list(iter(self._sampler))
            except Exception:
                shard_indices = None
        # Suppress verbose logging on non-rank0 ranks (keep warnings+errors)
        if dist.is_initialized() and dist.get_rank() != 0:
            logging.getLogger().setLevel(logging.WARNING)
        result = self.attacker.execute()
        # Persist adversarial images locally per rank if requested (torch.save for tensors)
        if self.config.return_adversarial_images and result:
            try:
                temp_dir = os.environ.get("ADV_OD_TMP", self.TEMP_DIR)
                if dist.get_rank() == 0 and not os.path.exists(temp_dir):
                    os.makedirs(temp_dir, exist_ok=True)
                dist.barrier()  # ensure directory exists for all
                out_path = os.path.join(temp_dir, f"adv_images_rank{dist.get_rank()}.pt")
                # Flatten result to count images
                flat_count = 0
                for batch in result:
                    if hasattr(batch, "shape") and len(batch.shape) == 4:
                        flat_count += batch.shape[0]
                    else:
                        flat_count += 1
                dataset_len = len(self.attacker._dataloader.dataset) if hasattr(self.attacker._dataloader, "dataset") else None
                torch.save({
                    "images": result,
                    "indices": shard_indices,
                    "local_image_count": flat_count,
                    "dataset_len": dataset_len
                }, out_path)
            except Exception as e:
                logging.error("Failed to store adversarial images on rank %d: %s", dist.get_rank(), e)
        dist.barrier()
        return result

    @staticmethod
    def gather_results(world_size: int) -> list:
        """Gather per-rank stored adversarial images into a single list (rank0 only).

        Looks in ADV_OD_TMP env dir or default TEMP_DIR.
        Cleans up files and removes directory if empty at the end.
        """
        temp_dir = os.environ.get("ADV_OD_TMP", DDPODAttacker.TEMP_DIR)
        gathered = []
        shards = []  # collect (indices, images)
        total_local_counts = 0
        dataset_len_reported = None
        for r in range(world_size):
            path = os.path.join(temp_dir, f"adv_images_rank{r}.pt")
            if not os.path.exists(path):
                continue
            try:
                payload = torch.load(path, map_location="cpu")
                if isinstance(payload, dict):
                    imgs = payload.get("images", [])
                    idxs = payload.get("indices", None)
                    local_count = payload.get("local_image_count", None)
                    if local_count is not None:
                        total_local_counts += int(local_count)
                    if dataset_len_reported is None:
                        dataset_len_reported = payload.get("dataset_len", None)
                    shards.append((idxs, imgs))
            finally:
                try:
                    os.remove(path)
                except OSError:
                    pass
        # If all shards have per-sample indices and counts match, restore ordering.
        if shards and all(s[0] is not None for s in shards):
            # Compute total number of images (flatten batches)
            total_images = 0
            flat_shards = []  # list of (indices, flat_images)
            for idxs, imgs in shards:
                # Flatten list of batch tensors into list of image tensors
                flat = []
                for batch in imgs:
                    if hasattr(batch, "shape") and len(batch.shape) == 4:  # batch tensor
                        for img in batch:
                            flat.append(img)
                    else:
                        flat.append(batch)
                flat_shards.append((idxs, flat))
                total_images += len(flat)
            # Validate index coverage
            total_indices = sum(len(idxs) for idxs, _ in flat_shards)
            if total_indices == total_images:
                ordered = [None] * total_indices
                for idxs, flat in flat_shards:
                    for index, img in zip(idxs, flat):
                        if 0 <= index < total_indices:
                            ordered[index] = img
                # Append in order, skipping any None holes
                for img in ordered:
                    if img is not None:
                        gathered.append(img)
            else:
                # Fallback concatenation if mismatch
                for _, flat in flat_shards:
                    gathered.extend(flat)
        else:
            # simple concatenation if indices unavailable
            for idxs, imgs in shards:
                for batch in imgs:
                    if hasattr(batch, "shape") and len(batch.shape) == 4:
                        for img in batch:
                            gathered.append(img)
                    else:
                        gathered.append(batch)
        # attempt to remove temp dir
        try:
            if os.path.isdir(temp_dir) and not os.listdir(temp_dir):
                os.rmdir(temp_dir)
        except OSError:
            pass
        # Consistency assertion for total image counts if dataset length known
        if dataset_len_reported is not None and total_local_counts > 0:
            if total_local_counts != dataset_len_reported:
                logger.error(
                    "[DDP OD][ASSERT] Total processed images (%d) != dataset length (%s)",
                    total_local_counts,
                    str(dataset_len_reported),
                )
            else:
                logger.info(
                    "[DDP OD] Image count consistency verified: %d images processed.",
                    total_local_counts,
                )
        return gathered