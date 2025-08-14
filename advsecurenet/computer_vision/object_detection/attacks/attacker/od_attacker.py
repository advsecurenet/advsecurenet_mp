import logging
import abc
import numpy as np
import torch
from tqdm.auto import tqdm

from advsecurenet.evaluation.adversarial_evaluator import AdversarialEvaluator
from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.shared.types.configs.attack_configs.od_attacker_config import (
    ODAttackerConfig,
)
from advsecurenet.utils.device_utils import move_batch_to_device, setup_device
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class ODAttacker(ABC):
    """
    Orchestrates an object-detection attack where:
      - `config.attack.object_detector` is used to *generate* the patch
      - `config.model` is used to *detect* on the patched images
    """

    def __init__(self, config: ODAttackerConfig):
        self._config = config
        self._device = setup_device(config.device.processor)
        self._eval_model = config.model.to(self._device).eval()
        self._dataloader = self._create_dataloader()
        try:
            dl_len = len(self._dataloader)  # type: ignore[arg-type]
        except Exception:
            dl_len = "unknown"
        logger.info(
            "Initialized ODAttacker: device=%s, dataloader=%s, batches=%s",
            str(self._device),
            type(self._dataloader).__name__,
            dl_len,
        )

    def _create_dataloader(self):
        """
        Creates a DataLoader instance based on the configuration.
        Returns:
            torch.utils.data.DataLoader: The DataLoader instance.
        """
        dl = self._config.dataloader
        if isinstance(dl, torch.utils.data.DataLoader):
            return dl
        return DataLoaderFactory.create_dataloader(dl)

    def preprocess_images(self, images):
        # Denormalize if a torchvision Normalize(mean,std) was applied
        images_for_patch = images
        ds = getattr(self._dataloader, "dataset", None)
        tfm = getattr(ds, "transform", None) if ds is not None else None
        for t in getattr(tfm, "transforms", []) if tfm is not None else []:
            if hasattr(t, "mean") and hasattr(t, "std"):
                mean = torch.as_tensor(
                    t.mean, dtype=images.dtype, device=images.device
                ).view(1, -1, 1, 1)
                std = torch.as_tensor(
                    t.std, dtype=images.dtype, device=images.device
                ).view(1, -1, 1, 1)
                images_for_patch = (images * std + mean).clamp(0.0, 1.0)
                break
        x_np = images_for_patch.detach().cpu().numpy()
        arr = x_np.astype(np.float32, copy=False)
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmin < -0.1 and vmax <= 1.5:
            arr = (arr + 1.0) / 2.0 * 255.0
        elif vmax > 1.1:
            pass
        else:
            arr = arr * 255.0
        vmin, vmax = float(arr.min()), float(arr.max())
        if vmin < 0 or vmax > 255.0:
            logger.warning(
                "Image values out of [0,255] before patch: min=%.3f max=%.3f; clipping.",
                vmin,
                vmax,
            )
        images_np_for_dpatch = np.clip(arr, 0.0, 255.0)
        return images_np_for_dpatch

    def preprocess_targets_dict(self, targets_dict):
        boxes = targets_dict["boxes"]
        labels = targets_dict["labels"]
        targets = []
        for b, l in zip(boxes, labels):
            raw = l.detach().cpu().numpy().astype(int)
            mapped = np.array(raw, dtype=int)
            targets.append(
                {
                    "boxes": b.detach().cpu().numpy(),
                    "labels": mapped,
                    "scores": np.ones(len(mapped), dtype=float),
                }
            )
        return targets

    def process_batch(self, data_batch):
        original_images, targets_dict = data_batch
        original_images, targets_dict = move_batch_to_device(
            original_images, targets_dict, self._device
        )
        images_preprocessed = self.preprocess_images(original_images)
        targets = self.preprocess_targets_dict(targets_dict)
        return images_preprocessed, targets, original_images

    @abstractmethod
    def execute(self):
        """
        Generate adversarial samples (or other attack outputs).
        Must be overridden in subclasses.
        """
        raise NotImplementedError(
            "Subclasses of ODAttackerBase must implement the execute function."
        )
