import os
import hashlib
import torch
import yolov5
from yolov5.utils.loss import ComputeLoss
from yolov5.models.common import AutoShape
from unittest.mock import patch


class CustomYolov5Model(torch.nn.Module):
    def __init__(
        self,
        model_weights_path="yolov5s.pt",
        device: str | int | torch.device | None = None,
    ):
        super().__init__()
        original_torch_load = torch.load
        def load_with_weights_only_false(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)
        # Decide loading strategy
        is_plain_state_dict = model_weights_path.endswith(".pth")
        base_arch_weights = "yolov5s.pt"
        arch_source = model_weights_path if not is_plain_state_dict else base_arch_weights
        if device is not None:
            if isinstance(device, (int,)):
                resolved_device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"
            else:
                resolved_device = str(device)
        else:
            if torch.cuda.is_available():
                try:
                    resolved_device = f"cuda:{torch.cuda.current_device()}"
                except Exception:
                    resolved_device = "cuda:0"
            else:
                resolved_device = "cpu"
        with patch("torch.load", side_effect=load_with_weights_only_false):
            self._model = yolov5.load(arch_source, autoshape=False, device=resolved_device).model
            self._autoshape = AutoShape(self._model)
            self._model.to(resolved_device)
        # Freeze BatchNorm running stats to avoid per-rank drift during adversarial gradients
        for m in self._model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
                m.track_running_stats = False
        if is_plain_state_dict and os.path.isfile(model_weights_path):
            try:
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
                        for prefix in ("model._model.model.", "model.model.", "model._model.",):
                            if nk.startswith(prefix):
                                nk = nk[len(prefix):]
                                break
                        if nk not in target_keys and f"model.{nk}" in target_keys:
                            nk = f"model.{nk}"
                        cleaned[nk] = v
                    return cleaned
                sd_clean = _clean(sd)
                self._model.load_state_dict(sd_clean, strict=False)
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
        self.compute_loss = ComputeLoss(self._model)

    def forward(self, x, targets=None):
        try:
            x = x.float()
        except Exception as e:
            pass
        try:
            x_dev = x.device
            for p in self._model.parameters():
                param_dev = p.device
                break
            else:
                param_dev = x_dev
            if param_dev != x_dev:
                self._model.to(x_dev)
        except Exception:
            pass
        if self.training and targets is not None:
            outputs = self._model(x)  # raw logits, pre-nms
            loss, loss_items = self.compute_loss(outputs, targets)
            loss_components_dict = {"loss_total": loss}
            loss_components_dict["loss_box"] = loss_items[0]
            loss_components_dict["loss_obj"] = loss_items[1]
            loss_components_dict["loss_cls"] = loss_items[2]
            return loss_components_dict
        else:
            return self._autoshape(x)  # after nms

    def predict_raw(self, x):
        """
        Predicts raw logits without applying NMS.
        """
        return self._model(x)
