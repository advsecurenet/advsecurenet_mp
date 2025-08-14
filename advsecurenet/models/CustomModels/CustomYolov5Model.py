import torch
import yolov5
from yolov5.utils.loss import ComputeLoss
from yolov5.models.common import AutoShape
from unittest.mock import patch


class CustomYolov5Model(torch.nn.Module):
    def __init__(self, model_weights_path="yolov5s.pt"):
        super().__init__()
        original_torch_load = torch.load

        def load_with_weights_only_false(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_torch_load(*args, **kwargs)

        # Temporarily patch torch.load to fix weights_only=True default in PyTorch 2.6
        with patch("torch.load", side_effect=load_with_weights_only_false):
            self._model = yolov5.load(model_weights_path, autoshape=False).model
            self._autoshape = AutoShape(self._model)
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
