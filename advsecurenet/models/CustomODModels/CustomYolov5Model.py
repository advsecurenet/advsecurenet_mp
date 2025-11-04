from contextlib import contextmanager
import os
from typing import Any, Dict, List
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import yolov5
from yolov5.utils.loss import ComputeLoss
from yolov5.models.common import AutoShape
from unittest.mock import patch

from advsecurenet.models.CustomODModels.CustomODBaseModel import CustomODBaseModel
from advsecurenet.datasets.label_utils import coco_label_ids_to_pascal


@contextmanager
def _suppress_yolov5_autocast_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            module=r"yolov5\.models\.common",
            message=r".*`torch\.cuda\.amp\.autocast\(.*\)` is deprecated.*",
        )
        yield


def translate_predictions_for_map_evaluator_yolo(
    predictions, dataset_name: str, expects_numpy: bool = True
) -> List[Dict[str, Any]]:
    results = []
    dataset_name = (dataset_name or "").lower()
    map_to_pascal = dataset_name == "pascal_voc"
    if expects_numpy:
        for det_tensor in predictions.pred:
            det_tensor_cpu = det_tensor.detach().cpu()
            boxes = det_tensor_cpu[:, :4].numpy()
            scores = det_tensor_cpu[:, 4].numpy()
            labels = det_tensor_cpu[:, 5].numpy().astype(int)
            # n = min(len(boxes), len(scores), len(labels))
            # boxes, scores, labels = boxes[:n], scores[:n], labels[:n]
            if map_to_pascal:
                mapped = np.array(
                    coco_label_ids_to_pascal(
                        labels.tolist(), assume_contiguous=True, unmapped_value=-1
                    ),
                    dtype=np.int64,
                )
                keep = mapped >= 0
                boxes, scores, labels = boxes[keep], scores[keep], mapped[keep]
            results.append({"boxes": boxes, "labels": labels, "scores": scores})
    else:
        for d in predictions:
            boxes = d["boxes"].detach().cpu().numpy()
            scores = d["scores"].detach().cpu().numpy()
            labels = d["labels"].detach().cpu().numpy().astype(int)
            if map_to_pascal:
                mapped = np.array(
                    coco_label_ids_to_pascal(
                        labels.tolist(), assume_contiguous=True, unmapped_value=-1
                    ),
                    dtype=np.int64,
                )
                keep = mapped >= 0
                boxes, scores, labels = boxes[keep], scores[keep], mapped[keep]
            results.append({"boxes": boxes, "scores": scores, "labels": labels})
    return results


class CustomYolov5Model(CustomODBaseModel):
    def __init__(
        self,
        num_classes: int = 80,
        model_weights_path="yolov5s.pt",
        device: str | int | torch.device | None = None,
    ):
        super().__init__()
        self.expects_numpy_images = True
        self._resolve_device(device)
        self.load_model_weights(model_weights_path)
        self.compute_loss = ComputeLoss(self._model)

    def _parameters_sha256(self):
        import hashlib

        with torch.no_grad():
            flat = torch.nn.utils.parameters_to_vector(
                [p.detach().cpu() for p in self._model.parameters()]
            )
        return hashlib.sha256(flat.numpy().tobytes()).hexdigest()

    def load_model_weights(self, model_weights_path):
        original_torch_load = torch.load

        def load_with_weights_only_false(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        # Decide loading strategy
        is_plain_state_dict = model_weights_path.endswith(".pth")
        base_arch_weights = "yolov5s.pt"
        arch_source = (
            model_weights_path if not is_plain_state_dict else base_arch_weights
        )
        with patch("torch.load", side_effect=load_with_weights_only_false):
            self._model = yolov5.load(
                arch_source, autoshape=False, device=self.device
            ).model
            self._autoshape = AutoShape(self._model)
            self._model.to(self.device)
        # Freeze BatchNorm running stats to avoid per-rank drift during adversarial gradients
        for m in self._model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
                m.track_running_stats = False
        if is_plain_state_dict and os.path.isfile(model_weights_path):
            try:
                before_hash = self._parameters_sha256()
                sd = torch.load(model_weights_path, map_location="cpu")
                if isinstance(sd, dict):
                    for k in ["state_dict", "model", "weights"]:
                        if k in sd and isinstance(sd[k], dict):
                            sd = sd[k]
                            break

                def _clean(d):
                    target_keys = set(self._model.state_dict().keys())
                    cleaned = {}
                    for k, v in d.items():
                        nk = k
                        for prefix in (
                            "model._model.model.",
                            "model.model.",
                            "model._model.",
                        ):
                            if nk.startswith(prefix):
                                nk = nk[len(prefix) :]
                                break
                        if nk not in target_keys and f"model.{nk}" in target_keys:
                            nk = f"model.{nk}"
                        cleaned[nk] = v
                    return cleaned

                sd_clean = _clean(sd)
                self._model.load_state_dict(sd_clean, strict=False)
                after_hash = self._parameters_sha256()
                if before_hash != after_hash:
                    print(
                        f"[CustomYolov5Model] Model parameters changed after loading '{model_weights_path}'."
                    )
                else:
                    print(
                        f"[CustomYolov5Model][WARN] Loading '{model_weights_path}' did not change model parameters."
                    )
            except Exception as e:
                print(f"[CustomYolov5Model][WARN] Failed to load .pth state_dict: {e}")
        self._model.hyp = {
            "box": 0.05,
            "obj": 1.0,
            "cls": 0.5,
            "anchor_t": 4.0,
            "cls_pw": 1.0,
            "obj_pw": 1.0,
            "fl_gamma": 0.0,
        }

    def forward(self, x, targets=None):
        try:
            x = x.float()
        except Exception as e:
            pass
        dev = self._module_device()
        if x.device != dev:
            x = x.to(dev, non_blocking=True)
        if self.training and targets is not None:
            outputs = self._model(x)  # raw logits, pre-nms
            if (
                isinstance(targets, torch.Tensor)
                and targets.device != outputs[0].device
            ):
                targets = targets.to(outputs[0].device)
            loss, loss_items = self.compute_loss(outputs, targets)
            loss_components_dict = {"loss_total": loss}
            loss_components_dict["loss_box"] = loss_items[0]
            loss_components_dict["loss_obj"] = loss_items[1]
            loss_components_dict["loss_cls"] = loss_items[2]
            return loss_components_dict
        else:
            return self._autoshape(x)  # after nms

    def predict(self, x_pre, training):
        preds = self.forward(x_pre)[0] if training else self.predict_raw(x_pre)[0]
        return preds

    def predict_raw(self, x):
        """
        Predicts raw logits without applying NMS.
        """
        dev = self._module_device()
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(dev)
        elif x.device != dev:
            x = x.to(dev)
        was_training = self._model.training
        self._model.train()
        preds = self._model(x)
        self._model.train(was_training)
        return preds

    def predict_per_batch(self, imgs, inference_model, clip_values):
        imgs = imgs.detach().cpu().numpy()
        imgs = [
            (img.transpose(1, 2, 0)).clip(0, clip_values[1]).astype(np.uint8)
            for img in imgs
        ]
        total_outs = []
        with torch.no_grad():
            with _suppress_yolov5_autocast_warning():
                outputs = inference_model(imgs, size=self.input_shape[1])
            for i, det in enumerate(outputs.xyxy):
                arr = det.cpu().numpy() if isinstance(det, torch.Tensor) else det
                if arr.size == 0:
                    out = {
                        "boxes": np.empty((0, 4)),
                        "scores": np.empty((0,)),
                        "labels": np.empty((0,), dtype=int),
                    }
                else:
                    # Get raw logits from outputs.pred
                    raw_pred = (
                        outputs.pred[i].cpu().numpy()
                    )  # shape: [num_detections, 5 + num_classes]
                    logits = raw_pred[:, 5:]  # shape: [num_detections, num_classes]
                    out = {
                        "boxes": arr[:, :4],  # x1, y1, x2, y2
                        "scores": arr[:, 4],
                        "labels": arr[:, 5].astype(int),
                        "logits": logits,  # Add logits here
                    }
                total_outs.append(out)
        return total_outs

    def initialize_inference_model(self, model, device=None, conf_thresh=0.7):
        try:
            if hasattr(model, "_autoshape"):
                inference_model = model._autoshape
            else:
                inference_model = AutoShape(model)
            if hasattr(inference_model, "conf"):
                inference_model.conf = conf_thresh
        except Exception as e:
            inference_model = model
        return inference_model

    def prepare_training_inputs(self, images: torch.Tensor, targets: list[dict]):
        images = images.to(self.device)
        images.requires_grad_(True)
        yolo_targets = self._convert_targets(targets, images.shape)
        return images, yolo_targets

    def calculate_loss(self, preds, target_val):
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for p in preds:
            obj_logit = p[..., 4]
            targets = torch.full_like(
                obj_logit, fill_value=target_val, device=self.device
            )
            loss += F.binary_cross_entropy_with_logits(
                obj_logit,
                targets,
                reduction="sum",
            )
        return loss

    def preprocess_x_for_loss_calculation(self, x, requires_grad=True):
        if isinstance(x, np.ndarray):
            # x shape is (B, C, H, W), values already in [0..255] or [0..1]
            x_preprocessed = torch.from_numpy(x).float().to(self.device)
        else:
            x_preprocessed = x.float().to(self.device)
        # Move inputs to device
        if requires_grad:
            x_preprocessed.requires_grad_(True)
        return x_preprocessed

    def translate_predictions_for_map_evaluator(
        self, predictions, dataset_name: str, expects_numpy: bool = True
    ) -> List[Dict[str, Any]]:
        return translate_predictions_for_map_evaluator_yolo(
            predictions, dataset_name, expects_numpy
        )

    def translate_labels(
        self,
        labels: list[dict[str, "torch.Tensor"]],
        batch_size: int,
    ) -> "torch.Tensor":
        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]
        labels_xcycwh_list = []
        for i, label_dict in enumerate(labels):
            # For each image in the batch, make a [N,6] tensor:
            # [ image_index, class_label, x_center, y_center, w, h ]
            # Number of objects in this image
            N = len(label_dict["boxes"])
            # create 2D tensor to encode labels and bounding boxes
            label_xcycwh = torch.zeros((N, 6), device=self.device)
            # xcycwh stands for:
            # x_center: the x-coordinate of the center of the bounding box.
            # y_center: the y-coordinate of the center of the bounding box.
            # w: the width of the bounding box.
            # h: the height of the bounding box.
            label_xcycwh[:, 0] = i  # image index
            # class labels
            raw_lbls = label_dict["labels"]
            if isinstance(raw_lbls, np.ndarray):
                lbls = torch.from_numpy(raw_lbls).to(self.device)
            else:
                lbls = raw_lbls.to(self.device)
            label_xcycwh[:, 1] = lbls
            # bounding boxes
            raw_boxes = label_dict["boxes"]
            if isinstance(raw_boxes, np.ndarray):
                boxes = torch.from_numpy(raw_boxes).float().to(self.device)
            else:
                boxes = raw_boxes.to(self.device)
            # boxes are [x1, y1, x2, y2]
            label_xcycwh[:, 2:6] = boxes
            # normalize bounding boxes to [0, 1]
            assert (
                width > 0 and height > 0
            ), f"Invalid input dimension: {self.input_shape}"
            label_xcycwh[:, 2:6:2] /= width
            label_xcycwh[:, 3:6:2] /= height
            # convert from x1y1x2y2 to xcycwh
            label_xcycwh[:, 4] -= label_xcycwh[:, 2]
            label_xcycwh[:, 5] -= label_xcycwh[:, 3]
            label_xcycwh[:, 2] += label_xcycwh[:, 4] / 2
            label_xcycwh[:, 3] += label_xcycwh[:, 5] / 2
            labels_xcycwh_list.append(label_xcycwh)
        labels_xcycwh = torch.vstack(labels_xcycwh_list)
        return labels_xcycwh

    # Model specific methods / helpers:

    def _convert_targets(self, targets, batch_shape) -> torch.Tensor:
        if isinstance(targets, dict):
            # dict-of-lists -> list[dict]
            targets = [
                {"boxes": targets["boxes"][i], "labels": targets["labels"][i]}
                for i in range(len(targets["boxes"]))
            ]
        assert isinstance(targets, list), "targets must be list[dict] or dict-of-lists"
        if len(targets) == 0:
            return torch.zeros((0, 6), device=self.device)
        _, _, H, W = batch_shape
        pieces = []
        for img_idx, td in enumerate(targets):
            boxes = td["boxes"].to(self.device).float()
            labels = td["labels"].to(self.device).long()
            if boxes.numel() == 0:
                pieces.append(torch.zeros((0, 6), device=self.device))
                continue
            x1, y1, x2, y2 = boxes.unbind(1)  # (N,)
            tgt = torch.zeros((boxes.size(0), 6), device=self.device)
            tgt[:, 0] = img_idx
            tgt[:, 1] = labels
            tgt[:, 2] = (x1 + x2) * 0.5 / W
            tgt[:, 3] = (y1 + y2) * 0.5 / H
            tgt[:, 4] = (x2 - x1) / W
            tgt[:, 5] = (y2 - y1) / H
            pieces.append(tgt)
        return (
            torch.cat(pieces, dim=0)
            if pieces
            else torch.zeros((0, 6), device=self.device)
        )

    def _resolve_device(self, device):
        if device is not None:
            if isinstance(device, (int,)):
                resolved_device = (
                    f"cuda:{device}" if torch.cuda.is_available() else "cpu"
                )
            else:
                resolved_device = str(device)
        else:
            if torch.cuda.is_available():
                try:
                    resolved_device = f"cuda:{torch.cuda.current_device()}"
                except Exception:
                    resolved_device = "cuda:0"
            elif (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                resolved_device = "mps"
            else:
                resolved_device = "cpu"
        self.device = resolved_device

    def _module_device(self) -> torch.device:
        try:
            return next(self._model.parameters()).device
        except Exception:
            try:
                return next(self.parameters()).device
            except Exception:
                return torch.device("cpu")

    def to(self, *args, **kwargs):
        # Let nn.Module move all registered submodules/buffers (incl. _model and _autoshape)
        super().to(*args, **kwargs)
        # Derive device from the inner model (single source of truth)
        try:
            dev = next(self._model.parameters()).device
            self.device = str(dev)
        except Exception:
            # Fallback keeps existing self.device if _model not ready yet
            dev = (
                torch.device(self.device)
                if hasattr(self, "device")
                else torch.device("cpu")
            )
        # Ensure AutoShape points at the moved model and is moved too
        if getattr(self, "_autoshape", None) is not None:
            try:
                self._autoshape.model = self._model
                self._autoshape.to(dev)
            except Exception:
                # If anything odd, just recreate it on the new device
                self._autoshape = AutoShape(self._model)
                self._autoshape.to(dev)
        # Rebuild ComputeLoss so its tensors live on the new device
        if getattr(self, "_model", None) is not None:
            self.compute_loss = ComputeLoss(self._model)
        return self
