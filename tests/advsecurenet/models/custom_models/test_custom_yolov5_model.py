import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from advsecurenet.models.CustomModels import CustomYolov5Model as cym
import advsecurenet.models.CustomModels.CustomYolov5Model as cymod
# REMOVE: from advsecurenet.models.CustomModels.CustomYolov5Model import CustomYolov5Model

# Minimal YOLOv5-like model for ComputeLoss
class MinimalYoloModel:
    def __init__(self, device='cpu'):
        self._param = torch.nn.Parameter(torch.zeros(1, device=device))
        self.hyp = {
            'box': 0.05,
            'obj': 1.0,
            'cls': 0.5,
            'anchor_t': 4.0,
            'cls_pw': 1.0,
            'obj_pw': 1.0,
            'fl_gamma': 0.0
        }
        self.model = [MagicMock()]  # mimic a list of layers
    def parameters(self):
        return iter([self._param])
    def __call__(self, x, *args, **kwargs):
        return "raw_logits"
    def eval(self):
        return self
    def train(self, mode=True):
        return self

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_model_initialization():
    with patch("yolov5.load") as mock_load, \
         patch("yolov5.models.common.AutoShape") as mock_autoshape:
        from advsecurenet.models.CustomModels.CustomYolov5Model import CustomYolov5Model
        dummy_model = MinimalYoloModel()
        mock_load.return_value.model = dummy_model
        mock_autoshape.return_value = MagicMock()
        model = CustomYolov5Model()
        assert isinstance(model, nn.Module)
        assert hasattr(model, "_model")
        assert hasattr(model, "_autoshape")
        assert hasattr(model, "compute_loss")
        assert isinstance(model._model.hyp, dict)

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_forward_eval_mode():
    with patch("yolov5.models.common.AutoShape") as mock_autoshape:
        from advsecurenet.models.CustomModels.CustomYolov5Model import CustomYolov5Model
        dummy_model = MinimalYoloModel()
        with patch("yolov5.load") as mock_load:
            mock_load.return_value.model = dummy_model
            model = CustomYolov5Model()
            model.eval()
            x = torch.rand(2, 3, 640, 640)
            out = model(x)
            assert out == "raw_logits"

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_forward_train_mode():
    with patch("yolov5.load") as mock_load, \
         patch("yolov5.models.common.AutoShape") as mock_autoshape:
        from advsecurenet.models.CustomModels.CustomYolov5Model import CustomYolov5Model
        dummy_model = MinimalYoloModel()
        mock_load.return_value.model = dummy_model
        mock_autoshape.return_value = MagicMock()
        model = CustomYolov5Model()
        # Patch compute_loss to return fixed values for test
        model.compute_loss = MagicMock(return_value=(1.23, [1.0, 2.0, 3.0]))
        model.train()
        x = torch.rand(2, 3, 640, 640)
        targets = "dummy_targets"
        out = model(x, targets=targets)
        assert isinstance(out, dict)
        assert set(out.keys()) == {"loss_total", "loss_box", "loss_obj", "loss_cls"}
        assert out["loss_total"] == 1.23
        assert out["loss_box"] == 1.0
        assert out["loss_obj"] == 2.0
        assert out["loss_cls"] == 3.0

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_custom_weights_path():
    with patch("yolov5.load") as mock_load, \
         patch("yolov5.models.common.AutoShape") as mock_autoshape:
        from advsecurenet.models.CustomModels.CustomYolov5Model import CustomYolov5Model
        dummy_model = MinimalYoloModel()
        mock_load.return_value.model = dummy_model
        mock_autoshape.return_value = MagicMock()
        model = CustomYolov5Model(model_weights_path="some/other/path.pt")
        mock_load.assert_called_once_with("some/other/path.pt", autoshape=False)

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_predict_raw():
    with patch("yolov5.load") as mock_load, \
         patch("yolov5.models.common.AutoShape") as mock_autoshape:
        from advsecurenet.models.CustomModels.CustomYolov5Model import CustomYolov5Model
        dummy_model = MinimalYoloModel()
        mock_load.return_value.model = dummy_model
        mock_autoshape.return_value = MagicMock()
        model = CustomYolov5Model()
        x = torch.rand(2, 3, 640, 640)
        out = model.predict_raw(x)
        assert out == "raw_logits"
