from advsecurenet.evaluation.base_evaluator import BaseEvaluator
from advsecurenet.models.base_model import BaseModel
from mean_average_precision import MetricBuilder
import torch
import warnings
import numpy as np
from typing import List, Dict, Any


class MeanAveragePrecisionEvaluator(BaseEvaluator):
    """
    Evaluator for mean Average Precision (mAP) for object detection.
    It calculates the mAP for both original and adversarial images and returns both scores,
    as well as the gap between them.
    """

    def __init__(self, num_classes: int = 80):
        self.num_classes = num_classes
        # One metric for clean images, one for adversarial
        self.clean_metric = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=True, num_classes=num_classes
        )
        self.adv_metric = MetricBuilder.build_evaluation_metric(
            "map_2d", async_mode=True, num_classes=num_classes
        )

    def reset(self):
        """
        Resets both clean and adversarial metric calculators.
        """
        self.clean_metric.reset()
        self.adv_metric.reset()

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
                results.append({
                    "boxes":  d["boxes"].detach().cpu().numpy(),
                    "scores": d["scores"].detach().cpu().numpy(),
                    "labels": d["labels"].detach().cpu().numpy().astype(int),
                })
        return results

    def _process_and_update(self, metric_builder, predictions, ground_truths):
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
                metric_builder.add(np.array(preds_formatted), np.array(gts_formatted))
            except Exception as e:
                warnings.warn(f"Error processing prediction {pred_idx}: {e}")

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
            if t.shape[0] not in (1,3) and t.shape[-1] in (1,3):  # HWC->CHW
                t = t.permute(2,0,1)
            t = t.float()
            if t.max().item() > 1.5:
                t = t / 255.0
            imgs.append(t.to(device, non_blocking=True))
        return imgs

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
        is_custom_yolov5 = bool(getattr(model.model, "IS_CUSTOM_YOLOV5", False)) # yolov5 expects numpy
        if is_custom_yolov5:
            with torch.no_grad():
                # 1. Run predictions on original and adversarial images
                clean_predictions = self.detections_to_dicts(
                    model(self.tensor_to_numpy_images(original_images)),
                    expects_numpy=is_custom_yolov5
                )
                adv_predictions = self.detections_to_dicts(
                    model(self.tensor_to_numpy_images(adversarial_images)),
                    expects_numpy=is_custom_yolov5
                )
        else:
            with torch.no_grad():
                clean_predictions = self.detections_to_dicts(
                    model(self.to_tensor_list(original_images, device)), expects_numpy=is_custom_yolov5
                )
                adv_predictions = self.detections_to_dicts(
                    model(self.to_tensor_list(adversarial_images, device)), expects_numpy=is_custom_yolov5
                )
        # Update both clean and adversarial metrics
        self._process_and_update(self.clean_metric, clean_predictions, targets)
        self._process_and_update(self.adv_metric, adv_predictions, targets)

    def get_results(self):
        """
        Returns the mAP results for clean and adversarial data, and the gap.
        Using the COCO metric configuration.
        """
        coco_format = {
            "iou_thresholds": np.arange(0.5, 1.0, 0.05),
            "recall_thresholds": np.arange(0.0, 1.01, 0.01),
            "mpolicy": "soft",
        }
        clean_map = self.clean_metric.value(**coco_format)["mAP"]
        adv_map = self.adv_metric.value(**coco_format)["mAP"]
        return {
            "clean_mAP": clean_map,
            "adversarial_mAP": adv_map,
            "mAP_gap": clean_map - adv_map,
        }
