import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import (
    RTDetrImageProcessor,
)
from transformers.utils import ModelOutput

from advsecurenet.models.CustomODModels.CustomODBaseModel import CustomODBaseModel
from advsecurenet.models.huggingface_model import HuggingFaceModel
from advsecurenet.shared.types.configs.model_config import HuggingFaceResolvedConfig


class RTDetrEvalAdapter(torch.nn.Module):
    def __init__(self, core_model, processor, device, conf_thresh: float):
        super().__init__()
        self.core_model = core_model
        self.processor = processor
        self.device = torch.device(device)
        self.conf_thresh = conf_thresh

    def forward(self, x=None, pixel_values=None, pixel_mask=None, labels=None, **kwargs):
        if pixel_values is not None and x is None:
            pv = pixel_values.to(self.device)
            pm = pixel_mask.to(self.device) if pixel_mask is not None else None
            with torch.no_grad():
                return self.core_model(pixel_values=pv, pixel_mask=pm, labels=labels, **kwargs)

        if x is None:
            raise ValueError("RTDetrEvalAdapter expects either x or pixel_values.")

        if isinstance(x, (list, tuple)):
            imgs = []
            for im in x:
                t = torch.from_numpy(im) if isinstance(im, np.ndarray) else im
                if t.dim() == 4 and t.size(0) == 1:
                    t = t[0]
                if t.dim() != 3:
                    raise ValueError(f"Expected CHW tensors; got {tuple(t.shape)}")
                imgs.append(t)
            batch = torch.stack([t.to(self.device).float() for t in imgs], dim=0)
        elif isinstance(x, torch.Tensor):
            if x.dim() == 3:
                batch = x.unsqueeze(0).to(self.device).float()
            elif x.dim() == 4:
                batch = x.to(self.device).float()
            else:
                raise ValueError(f"Expected CHW or BCHW tensor; got {tuple(x.shape)}")
        else:
            raise TypeError("Unsupported input type for RTDetrEvalAdapter.")
        imgs_list = [img.detach().cpu() for img in batch]
        enc = self.processor(images=imgs_list, return_tensors="pt", do_rescale=False)
        enc = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in enc.items()}
        with torch.no_grad():
            raw_out = self.core_model(**enc)
        if not isinstance(raw_out, ModelOutput):
            raw_out = ModelOutput(raw_out)
        model_out = ModelOutput({
            k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v)
            for k, v in raw_out.items()
        })
        H, W = batch.shape[-2], batch.shape[-1]
        target_sizes = torch.tensor([[H, W]] * batch.shape[0], dtype=torch.long)
        results = self.processor.post_process_object_detection(
            model_out, target_sizes=target_sizes, threshold=self.conf_thresh
        )
        detections = []
        for r in results:
            detections.append({
                "boxes": r["boxes"],
                "scores": r["scores"],
                "labels": r["labels"].to(torch.int64),
            })
        return detections


class CustomRTDetrModel(CustomODBaseModel):
    def __init__(
        self,
        model_name: str = "PekingU/rtdetr_r101vd_coco_o365",
        device: Optional[Union[str, int, torch.device]] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.expects_numpy_images = False
        if device is None:
            device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.processor: RTDetrImageProcessor = RTDetrImageProcessor.from_pretrained(
            model_name, cache_dir=cache_dir
        )
        cfg = HuggingFaceResolvedConfig(
            model_name=model_name,
            model_id=model_name,
            cache_dir=cache_dir,
        )
        self._model = HuggingFaceModel(cfg).to(self.device).model
        self.id2label = self._model.config.id2label
        self.label2id = self._model.config.label2id
        self.categories = [self.id2label[i] for i in range(len(self.id2label))]
        self.num_classes = len(self.categories)
        self._model_name = "CustomRTDetrModel"
        
    def forward(self, x, targets=None):
        """
        x: torch.Tensor [B,C,H,W] float32 (0..1 or 0..255)
        targets: list of dicts in HF RT-DETR format:
                {'class_labels': LongTensor[N], 'boxes': FloatTensor[N,4] (cxcywh in [0,1])}
        """
        images_list = self._to_image_list(x)
        enc = self.processor(images=images_list, return_tensors="pt", do_rescale=False)
        enc = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in enc.items()}
        if self.training and targets is not None:
            # HF computes loss internally when labels are provided
            outputs = self._model(**enc, labels=targets)
            loss_components = {"loss_total": outputs.loss}
            if hasattr(outputs, "loss_dict") and isinstance(outputs.loss_dict, dict):
                for k, v in outputs.loss_dict.items():
                    loss_components[f"loss_{k}"] = v
            else:
                for name in ("loss_ce", "loss_cls", "loss_bbox", "loss_giou",
                            "loss_cardinality", "loss_objectness"):
                    if hasattr(outputs, name) and getattr(outputs, name) is not None:
                        loss_components[name] = getattr(outputs, name)
            return loss_components
        return self._model(**enc)

    def predict(self, x, training):
        if not training:
            self.eval()
        return self.forward(x, None)

    def predict_raw(self, x):
        """Return raw HF outputs with `.logits` and `.pred_boxes`."""
        self.eval()
        return self.forward(x, None)

    def predict_per_batch(self, imgs, inference_model, clip_values):
        """
        Post-processed predictions in standardized numpy dicts:
        - boxes: xyxy pixels
        - scores: float32
        - labels: int64
        - label_names: np.ndarray[str]
        """
        self.eval()
        if isinstance(imgs, torch.Tensor):
            batch = imgs
        else:
            raise TypeError("imgs must be a torch.Tensor of shape [B,C,H,W]")
        _, _, H, W = batch.shape
        target_sizes = torch.tensor([[H, W]] * batch.shape[0], dtype=torch.long)
        images_list = self._to_image_list(batch)
        enc = self.processor(images=images_list, return_tensors="pt", do_rescale=False)
        pixel_values = enc["pixel_values"].to(self.device)
        pixel_mask = enc.get("pixel_mask", None)
        if pixel_mask is not None:
            pixel_mask = pixel_mask.to(self.device)
        with torch.no_grad():
            outputs = inference_model(pixel_values=pixel_values, pixel_mask=pixel_mask)
        if isinstance(outputs, list):
            results = outputs
        else:
            if not isinstance(outputs, ModelOutput):
                outputs = ModelOutput(outputs)
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor):
                    outputs[k] = v.detach().cpu()
            results = self.processor.post_process_object_detection(
                outputs,
                target_sizes=target_sizes,
                threshold=getattr(self, "conf_thresh", 0.7),
            )
        preds = []
        for res in results:
            boxes = res["boxes"].numpy().astype(np.float32) if len(res["boxes"]) else np.empty((0, 4), np.float32)
            scores = res["scores"].numpy().astype(np.float32) if len(res["scores"]) else np.empty((0,), np.float32)
            labels = res["labels"].numpy().astype(np.int64) if len(res["labels"]) else np.empty((0,), np.int64)
            label_names = np.array([self.id2label[int(i)] for i in labels], dtype=object) if labels.size else np.empty((0,), dtype=object)
            preds.append(
                {
                    "boxes": boxes,
                    "scores": scores,
                    "labels": labels,
                    "label_names": label_names,
                }
            )
        return preds

    def initialize_inference_model(self, model, device, conf_thresh=0.7):
        self.conf_thresh = conf_thresh
        hf_model = self._model.to(device).eval()
        proc = self.processor
        dev = torch.device(device)
        return RTDetrEvalAdapter(hf_model, proc, dev, conf_thresh)

    def prepare_training_inputs(self, images: torch.Tensor, targets: List[Dict]):
        """
        Returns a pair to feed into forward():
          - images_tensor [B,C,H,W] on device with requires_grad=True
          - targets in RT-DETR format (use translate_labels first)
        """
        x = images.to(self.device).float()
        x.requires_grad_(True)
        y = self.translate_labels(targets, batch_size=x.shape[0])
        return x, y

    def calculate_loss(self, predictions, target_val):
        """
        Generic score-based objective using class logits.

        If target_val = 0 -> encourage vanishing (reduce scores)
        If target_val = 1 -> encourage fabrication (increase scores)
        """
        direction = 2 * float(target_val) - 1.0
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        if hasattr(predictions, "logits") and isinstance(predictions.logits, torch.Tensor):
            logits = predictions.logits  # (B, Q, C)
            max_per_query, _ = logits.max(dim=-1)  # (B, Q)
            loss -= direction * max_per_query.sum()
            return loss
        if isinstance(predictions, list):
            for pred in predictions:
                if isinstance(pred, dict) and "scores" in pred:
                    s = pred["scores"]
                    if isinstance(s, np.ndarray):
                        s = torch.from_numpy(s).to(self.device)
                    loss -= direction * s.sum()
        return loss

    def preprocess_x_for_loss_calculation(self, x, requires_grad=True):
        """
        Ensure a float32 BCHW tensor on the right device, roughly in [0,1].
        """
        if isinstance(x, np.ndarray):
            t = torch.from_numpy(x).float()
        else:
            t = x.float()
        # Normalize if needed
        with torch.no_grad():
            if t.max() > 1.0:
                t = t / 255.0
        t = t.to(self.device)
        t.requires_grad_(bool(requires_grad))
        return t

    def translate_labels(
        self,
        labels: List[Dict[str, Union[torch.Tensor, np.ndarray]]],
        batch_size: int,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Convert your per-image labels:
          {'boxes': XYXY pixel coords [N,4], 'labels': [N]}
        into RT-DETR format expected by HF:
          {'boxes': cxcywh normalized [N,4], 'class_labels': [N]}
        """
        assert hasattr(self, "input_shape") and self.input_shape is not None, \
            "self.input_shape must be set (e.g., by ODWrapper) before translate_labels()"
        if self.channels_first:
            _, H, W = self.input_shape
        else:
            H, W, _ = self.input_shape

        def _to_tensor(a, dtype, device=self.device):
            if isinstance(a, np.ndarray):
                return torch.from_numpy(a).to(device=device, dtype=dtype)
            return a.to(device=device, dtype=dtype)
        y = list(labels or [])
        if len(y) < batch_size:
            y = y + [{"boxes": np.empty((0, 4), np.float32), "labels": np.empty((0,), np.int64)}
                     for _ in range(batch_size - len(y))]
        elif len(y) > batch_size:
            y = y[:batch_size]
        out: List[Dict[str, torch.Tensor]] = []
        for d in y:
            boxes_xyxy = d.get("boxes", np.empty((0, 4), np.float32))
            classes = d.get("labels", np.empty((0,), np.int64))
            boxes_xyxy = _to_tensor(boxes_xyxy, torch.float32)
            classes = _to_tensor(classes, torch.int64)
            if boxes_xyxy.numel() == 0:
                out.append({"boxes": boxes_xyxy, "class_labels": classes})
                continue
            # XYXY pixels -> CXCYWH normalized
            x1, y1, x2, y2 = boxes_xyxy.unbind(dim=1)
            w = (x2 - x1).clamp(min=0)
            h = (y2 - y1).clamp(min=0)
            cx = x1 + 0.5 * w
            cy = y1 + 0.5 * h
            # Normalize to [0,1] by image dims
            assert W > 0 and H > 0, f"Invalid input_shape: {self.input_shape}"
            boxes_cxcywh = torch.stack([cx / W, cy / H, w / W, h / H], dim=1)
            out.append({"boxes": boxes_cxcywh, "class_labels": classes})
        return out

    def _to_image_list(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Convert BCHW tensor to a list of CHW tensors for the processor.
        Keeps data in current numeric range; the processor will normalize.
        """
        if not isinstance(x, torch.Tensor) or x.dim() != 4:
            raise TypeError("Expected x as torch.Tensor [B,C,H,W]")
        return [x[i].detach().cpu() for i in range(x.shape[0])]
