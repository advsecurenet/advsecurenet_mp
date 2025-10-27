import torch
import os
import numpy as np
from unittest.mock import patch
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)

from advsecurenet.datasets.COCO.coco_utils import COCO_INSTANCE_CATEGORY_NAMES, ID_TO_CONTIGUOUS
from advsecurenet.models.CustomODModels.CustomODBaseModel import CustomODBaseModel
from advsecurenet.datasets.label_utils import coco_label_ids_to_pascal

FASTERRCNN_COCO_LABEL_OFFSET = -1  # background is 0, first class is 1


class CustomFasterRCNNModel(CustomODBaseModel):
    def __init__(
        self,
        num_classes: int = 91,
        pretrained: bool = True,
        pretrained_backbone: bool = True,
        model_weights_path: str | None = None,
        device: str | int | torch.device | None = None,
    ):
        super().__init__()
        self.expects_numpy_images = False
        self.num_classes = num_classes
        self.load_model_weights(
            model_weights_path=model_weights_path, 
            pretrained=pretrained, 
            pretrained_backbone=pretrained_backbone,
        )
        self._model_name = "CustomFasterRCNNModel"
        self.categories = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]
        # Resolve and move the model
        if device is None:
            device = (
                f"cuda:{torch.cuda.current_device()}"
                if torch.cuda.is_available()
                else "cpu"
            )
        self.device = torch.device(device)
        self._model.to(self.device)


    def _parameters_sha256(self):
        import hashlib
        with torch.no_grad():
            flat = torch.nn.utils.parameters_to_vector(
                [p.detach().cpu() for p in self._model.parameters()]
            )
        return hashlib.sha256(flat.numpy().tobytes()).hexdigest()

    def load_model_weights(self, model_weights_path, pretrained, pretrained_backbone):
        if pretrained:
            self._model = fasterrcnn_resnet50_fpn_v2(
                weights=(
                    FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
                    if pretrained_backbone
                    else None
                ),
                num_classes=None if pretrained_backbone else self.num_classes,
            )
        else:
            self._model = fasterrcnn_resnet50_fpn_v2(
                weights=None, num_classes=self.num_classes
            )
        for m in self._model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
                m.track_running_stats = False
        if model_weights_path and isinstance(model_weights_path, str) and model_weights_path.endswith(".pth") and os.path.isfile(model_weights_path):
            try:
                before_hash = self._parameters_sha256()
                original_torch_load = torch.load
                def load_with_weights_only_false(*args, **kwargs):
                    kwargs["weights_only"] = False
                    return original_torch_load(*args, **kwargs)
                with patch("torch.load", side_effect=load_with_weights_only_false):
                    sd = torch.load(model_weights_path, map_location="cpu")
                if isinstance(sd, dict):
                    for k in ["state_dict", "model", "weights"]:
                        if k in sd and isinstance(sd[k], dict):
                            sd = sd[k]
                            break
                target_keys = set(self._model.state_dict().keys())
                sd_clean = {}
                for k, v in sd.items():
                    nk = k
                    for prefix in (
                        "model._model.model.",
                        "model.model.",
                        "model._model.",
                        "model.",
                        "_model.model.",
                        "_model.",
                        "module.",
                    ):
                        if nk.startswith(prefix):
                            nk = nk[len(prefix):]
                    if nk not in target_keys and f"model.{nk}" in target_keys:
                        nk = f"model.{nk}"
                    sd_clean[nk] = v
                self._model.load_state_dict(sd_clean, strict=False)
                after_hash = self._parameters_sha256()
                if before_hash != after_hash:
                    print(f"[CustomFasterRCNNModel] Model parameters changed after loading '{model_weights_path}'.")
                else:
                    print(f"[CustomFasterRCNNModel][WARN] Loading '{model_weights_path}' did not change model parameters.")
            except Exception as e:
                print(f"[CustomFasterRCNNModel][WARN] Failed to load .pth state_dict: {e}")
        

    def forward(self, x, targets=None):
        """
        Forward pass for Faster R-CNN.

        Args:
            x: Input images tensor of shape (N, C, H, W)
            targets: List of dicts with 'boxes' and 'labels' keys (for training)

        Returns:
            During training: dict with loss components
            During inference: list of dicts with 'boxes', 'scores', 'labels'
        """
        if isinstance(x, torch.Tensor):
            x = [x[i] for i in range(x.shape[0])]
            # img_list = []
            # for i in range(x.shape[0]):
            #     im = x[i].to(self.device)
            #     # Ensure float32 and normalize if it looks like [0, 255]
            #     if im.dtype != torch.float32:
            #         im = im.float()
            #     if im.max() > 1.0:
            #         im = im / 255.0
            #     img_list.append(im)
            # x = img_list

        if self.training and targets is not None:
            loss_dict = self._model(x, targets)
            total_loss = sum(loss_dict.values())
            loss_dict["loss_total"] = total_loss
            return loss_dict
        else:
            return self._model(x)

    def predict(self, x, training):
        if not training:
            self.eval()
        return self.forward(x, None)

    def predict_raw(self, x):
        """
        Predicts raw outputs without post-processing.
        """
        return self._model(x)

    def predict_per_batch(self, imgs, inference_model, clip_values):
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs[i] for i in range(imgs.shape[0])]
        outs = inference_model(imgs)
        preds = self._translate_predictions(outs)
        return preds

    def initialize_inference_model(self, model, device, conf_thresh=0.7):
        inference_model = model.to(device)
        inference_model.eval()
        return inference_model

    def prepare_training_inputs(self, images: torch.Tensor, targets: list[dict]):
        images = images.to(self.device).float()
        img_list = [images[i] for i in range(images.size(0))]
        for t in img_list:
            t.requires_grad_(True)
        n = len(next(iter(targets.values()))) if targets else 0
        targets = [{k: v[i] for k, v in targets.items()} for i in range(n)]
        return img_list, targets

    def calculate_loss(self, predictions, target_val):
        # target_val = 0 - subtract, target_val = 1 - add
        target_val = 2 * target_val - 1  # map 0 -> -1, 1 -> 1
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for pred in predictions:
            if "scores" in pred:
                scores = pred["scores"]
                loss -= torch.sum(scores) * target_val
            else:
                if "logits" in pred:
                    logits = pred["logits"]
                    loss -= torch.sum(torch.max(logits, dim=1)[0]) * target_val
        return loss

    def preprocess_x_for_loss_calculation(self, x, requires_grad=True):
        # build list[Tensor(C,H,W)] float32 in [0,1]
        imgs = []
        if isinstance(x, np.ndarray):
            for i in range(x.shape[0]):
                t = torch.from_numpy(x[i]).to(self.device).float()
                if t.max() > 1:
                    t = t / 255.0
                imgs.append(t)
        else:
            if x.dim() == 4:
                for i in range(x.shape[0]):
                    t = x[i].to(self.device).float()
                    if t.max() > 1:
                        t = t / 255.0
                    imgs.append(t)
            else:
                t = x.to(self.device).float()
                if t.max() > 1:
                    t = t / 255.0
                imgs = [t]
        x_tensor = torch.stack(imgs, dim=0)
        if requires_grad:
            x_tensor.requires_grad_(True)
        # return [x_tensor[i] for i in range(x_tensor.shape[0])]
        return x_tensor

    def translate_labels(
        self, labels: list[dict[str, torch.Tensor | np.ndarray]], batch_size: int
    ):
        """From your labels [{'boxes': Nx4, 'labels': N}, …] to
        torchvision's targets: list of dicts with Tensors."""
        y = self._align_targets_to_batch(labels, batch_size=batch_size)
        targets = []
        for lab in y:
            boxes = lab["boxes"]
            classes = lab["labels"]
            if isinstance(boxes, np.ndarray):
                boxes = torch.from_numpy(boxes).float()
            if isinstance(classes, np.ndarray):
                classes = torch.from_numpy(classes).long()
            targets.append(
                {
                    "boxes": boxes.to(self.device),
                    "labels": classes.to(self.device),
                }
            )
        return targets

    # Model specific methods / helpers:

    def _empty_target_np(self):
        return {
            "boxes": np.empty((0, 4), dtype=np.float32),
            "labels": np.empty((0,), dtype=np.int64),
        }

    def _align_targets_to_batch(self, y, batch_size: int):
        """
        Ensure we have exactly one target dict per image.
        Pads with empty targets or truncates if needed.
        """
        if y is None:
            return [self._empty_target_np() for _ in range(batch_size)]
        y = list(y)
        if len(y) < batch_size:
            y = y + [self._empty_target_np() for _ in range(batch_size - len(y))]
        elif len(y) > batch_size:
            y = y[:batch_size]
        return y

    def _translate_predictions(self, outputs: list[dict[str, torch.Tensor]]):
        """From torchvision outputs (list of dicts) back to your np format."""
        preds = []
        for out in outputs:
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            pred = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
            pred["label_names"] = np.array([self.categories[int(l)] for l in labels])
            preds.append(pred)
        return preds

    def _translate_predictions_for_map_evaluator_coco(self, outputs: list[dict[str, torch.Tensor]]):
        preds = []
        for out in outputs:
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            mapped = np.array([ID_TO_CONTIGUOUS.get(int(l), -1) for l in labels],
                            dtype=np.int32)
            keep = mapped >= 0
            boxes, scores, mapped = boxes[keep], scores[keep], mapped[keep]
            keep = (mapped >= 0) & (mapped < self.num_classes)
            boxes, scores, mapped = boxes[keep], scores[keep], mapped[keep]
            pred = {
                "boxes": boxes,
                "scores": scores,
                "labels": mapped + FASTERRCNN_COCO_LABEL_OFFSET,  # map to COCO labels
                "label_names": np.array([COCO_INSTANCE_CATEGORY_NAMES[i] for i in mapped]),
            }
            preds.append(pred)
        return preds


    def translate_predictions_for_map_evaluator(self, outputs: list[dict[str, torch.Tensor]], dataset_name: str = "coco"):
        """From torchvision outputs (list of dicts) back to your np format."""
        if dataset_name.lower() == "coco":
            return self._translate_predictions_for_map_evaluator_coco(outputs)
        map_to_pascal = dataset_name.lower() == "pascal_voc"
        preds = []
        for out in outputs:
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            if map_to_pascal: # Assuming COCO label IDs as model's output
                labels_raw = out["labels"].detach().cpu().numpy().astype(int)
                mapped = np.array(
                    coco_label_ids_to_pascal(labels_raw.tolist(), assume_contiguous=False, unmapped_value=-1),
                    dtype=np.int64,
                )
                keep = mapped >= 0
                preds.append(
                    {
                        "boxes": boxes[keep],
                        "scores": scores[keep],
                        "labels": mapped[keep],
                    }
                )
            else:
                labels = out["labels"].detach().cpu().numpy()
                pred = {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                }
                preds.append(pred)
        return preds