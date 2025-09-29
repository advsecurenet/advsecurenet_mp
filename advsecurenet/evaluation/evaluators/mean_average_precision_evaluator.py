from advsecurenet.evaluation.base_evaluator import BaseEvaluator
from advsecurenet.models.base_model import BaseModel
from advsecurenet.datasets.label_utils import get_dataset_classes_count
from mean_average_precision import MetricBuilder
import torch
import torch.distributed as dist
import warnings
import numpy as np
from typing import List, Dict, Any

_PANDAS_CONCAT_MSG = r"^The behavior of DataFrame concatenation with empty or all-NA entries is deprecated"
warnings.filterwarnings("ignore", category=FutureWarning, message=_PANDAS_CONCAT_MSG)


class MeanAveragePrecisionEvaluator(BaseEvaluator):
    """
    Evaluator for mean Average Precision (mAP) for object detection.
    It calculates the mAP for both original and adversarial images and returns both scores,
    as well as the gap between them.
    """

    def __init__(self, dataset_name: str = "coco"):
        self.dataset_name = dataset_name.lower()
        self.num_classes = get_dataset_classes_count(self.dataset_name)
        # One metric for clean images, one for adversarial
        self.clean_metric = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=True, num_classes=self.num_classes
        )
        self.adv_metric = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=True, num_classes=self.num_classes
        )
        self._clean_entries = []
        self._adv_entries = []

    def reset(self):
        """
        Resets both clean and adversarial metric calculators.
        """
        self.clean_metric.reset()
        self.adv_metric.reset()
        self._clean_entries.clear()
        self._adv_entries.clear()

    def detections_to_dicts(self, detections, expects_numpy=False):
        results = []
        if expects_numpy:
            for det_tensor in detections.pred:
                det_tensor_cpu = det_tensor.detach().cpu()
                boxes = det_tensor_cpu[:, :4].numpy()
                scores = det_tensor_cpu[:, 4].numpy()
                labels = det_tensor_cpu[:, 5].numpy().astype(int)
                results.append({"boxes": boxes, "labels": labels, "scores": scores})
        else:
            for d in detections:
                results.append(
                    {
                        "boxes": d["boxes"].detach().cpu().numpy(),
                        "scores": d["scores"].detach().cpu().numpy(),
                        "labels": d["labels"].detach().cpu().numpy().astype(int),
                    }
                )
        return results

    def tensor_to_numpy_images(self, images: torch.Tensor):
        # images: [B, C, H, W], values in [0, 1] or [0, 255]
        imgs = []
        for img in images:
            img_np = img.detach().cpu().numpy()
            img_np = np.transpose(img_np, (1, 2, 0))  # [H, W, C]
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)
            imgs.append(img_np)
        return imgs

    def to_tensor_list(self, x, device):
        imgs = []
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(x, torch.Tensor) and x.ndim == 4:
            it = [x[i] for i in range(x.shape[0])]
        elif isinstance(x, torch.Tensor) and x.ndim == 3:
            it = [x]
        elif isinstance(x, list):
            it = [torch.from_numpy(i) if isinstance(i, np.ndarray) else i for i in x]
        else:
            raise TypeError(type(x))
        for t in it:
            if t.ndim != 3:
                raise ValueError(f"Each image must be 3D; got {tuple(t.shape)}")
            if t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):  # HWC->CHW
                t = t.permute(2, 0, 1)
            t = t.float()
            if t.max().item() > 1.5:
                t = t / 255.0
            imgs.append(t.to(device, non_blocking=True))
        return imgs

    def _process_and_update(self, metric_builder, predictions, ground_truths):
        distributed = dist.is_available() and dist.is_initialized()
        for pred_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            try:
                # Process ground truth data
                gt_boxes = gt["boxes"]
                gt_labels = gt["labels"]
                pred_boxes = pred["boxes"]
                pred_labels = pred["labels"]
                pred_scores = pred["scores"]
                # Format for the library: [xmin, ymin, xmax, ymax, class_id, confidence]
                preds_formatted = [
                    list(b) + [int(l), float(s)]
                    for b, l, s in zip(pred_boxes, pred_labels, pred_scores)
                ]
                gts_formatted = [
                    list(b) + [int(l), 0, 0] for b, l in zip(gt_boxes, gt_labels)
                ]
                entry = (np.array(preds_formatted), np.array(gts_formatted))
                if metric_builder is self.clean_metric:
                    self._clean_entries.append(entry)
                else:
                    self._adv_entries.append(entry)
                if not distributed:
                    metric_builder.add(entry[0], entry[1])
            except Exception as e:
                warnings.warn(f"Error processing prediction {pred_idx}: {e}")

    def update(
        self,
        model: BaseModel,
        original_images: torch.Tensor,
        adversarial_images: torch.Tensor,
        targets: List[Dict[str, Any]],
    ):
        """
        Update the mAP metrics by running the model on original and adversarial images
        and comparing them to ground truth targets.
        Args:
            model (BaseModel): The object detection model to evaluate.
            original_images (torch.Tensor): A batch of original images.
            adversarial_images (torch.Tensor): A batch of adversarial images.
            targets (List[Dict[str, Any]]): A list of ground truth dictionaries, each with 'boxes' and 'labels'.
        """
        model.eval()
        device = next(model.parameters()).device
        expects_numpy_images = bool(getattr(model, "expects_numpy_images", False))
        if expects_numpy_images:
            with torch.no_grad():
                clean_predictions = self.detections_to_dicts(
                    model(self.tensor_to_numpy_images(original_images)),
                    expects_numpy=expects_numpy_images,
                )
                adv_predictions = self.detections_to_dicts(
                    model(self.tensor_to_numpy_images(adversarial_images)),
                    expects_numpy=expects_numpy_images,
                )
        else:
            with torch.no_grad():
                clean_predictions = model.translate_predictions_for_map_evaluator(
                    model(self.to_tensor_list(original_images, device)),
                    dataset_name=self.dataset_name,
                )
                adv_predictions = model.translate_predictions_for_map_evaluator(
                    model(self.to_tensor_list(adversarial_images, device)),
                    dataset_name=self.dataset_name,
                )
        # Update both clean and adversarial metrics
        self._process_and_update(self.clean_metric, clean_predictions, targets)
        self._process_and_update(self.adv_metric, adv_predictions, targets)

    def get_results(self):
        """
        Returns the mAP results for clean and adversarial data, and the gap.
        Using the COCO metric configuration.
        """
        map_eval_format = {
            "iou_thresholds": np.arange(0.5, 1.0, 0.05),
            "recall_thresholds": np.arange(0.0, 1.01, 0.01),
            "mpolicy": "soft",
        }
        distributed = dist.is_available() and dist.is_initialized()
        if distributed:
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            # Gather entries
            try:
                clean_lists = [None] * world_size
                adv_lists = [None] * world_size
                dist.barrier()
                dist.all_gather_object(clean_lists, self._clean_entries)
                dist.all_gather_object(adv_lists, self._adv_entries)
                if rank == 0:
                    # Rebuild metrics centrally
                    self.clean_metric.reset()
                    self.adv_metric.reset()
                    for entries in clean_lists:
                        if entries:
                            for preds_arr, gts_arr in entries:
                                self.clean_metric.add(preds_arr, gts_arr)
                    for entries in adv_lists:
                        if entries:
                            for preds_arr, gts_arr in entries:
                                self.adv_metric.add(preds_arr, gts_arr)
                    clean_map = self.clean_metric.value(**map_eval_format)["mAP"]
                    adv_map = self.adv_metric.value(**map_eval_format)["mAP"]
                    t = torch.tensor(
                        [clean_map, adv_map],
                        dtype=torch.float32,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                else:
                    t = torch.zeros(
                        2,
                        dtype=torch.float32,
                        device="cuda" if torch.cuda.is_available() else "cpu",
                    )
                # Broadcast final maps to all ranks
                dist.broadcast(t, src=0)
                clean_map, adv_map = float(t[0].item()), float(t[1].item())
            except Exception:
                # Fallback: compute local only
                clean_map = self.clean_metric.value(**map_eval_format)["mAP"]
                adv_map = self.adv_metric.value(**map_eval_format)["mAP"]
        else:
            clean_map = self.clean_metric.value(**map_eval_format)["mAP"]
            adv_map = self.adv_metric.value(**map_eval_format)["mAP"]
        return {
            "clean_mAP": clean_map,
            "adversarial_mAP": adv_map,
            "mAP_gap": clean_map - adv_map,
        }
