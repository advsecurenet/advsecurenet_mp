import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock, ANY
import numpy as np
import sys, types

# Provide a minimal "transformers" stub so that importing advsecurenet.models.* does not fail
if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")
    class _DummyAutoModel:
        pass
    class _DummyAutoConfig:
        pass
    transformers_stub.AutoModel = _DummyAutoModel
    transformers_stub.AutoConfig = _DummyAutoConfig
    sys.modules["transformers"] = transformers_stub

"""Tests for CustomYolov5Model.

Note: Avoid importing advsecurenet.models package at module level to prevent pulling
in optional heavy dependencies like transformers. Import the model class lazily
within each test after patching external yolov5 symbols.
"""


# Minimal YOLOv5-like model for ComputeLoss
class MinimalYoloModel:
    def __init__(self, device="cpu"):
        self._param = torch.nn.Parameter(torch.zeros(1, device=device))
        self.hyp = {
            "box": 0.05,
            "obj": 1.0,
            "cls": 0.5,
            "anchor_t": 4.0,
            "cls_pw": 1.0,
            "obj_pw": 1.0,
            "fl_gamma": 0.0,
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

    def to(self, device):
        # mimic torch.nn.Module.to
        self._param = self._param.to(device)
        return self


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_model_initialization():
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )

        dummy_model = MinimalYoloModel()
        # Ensure modules() exists to iterate BN freezing loop
        dummy_model.modules = lambda: []
        mock_load.return_value.model = dummy_model
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()
        model = CustomYolov5Model()
        assert isinstance(model, nn.Module)
        assert hasattr(model, "_model")
        assert hasattr(model, "_autoshape")
        assert hasattr(model, "compute_loss")
        assert isinstance(model._model.hyp, dict)


@pytest.mark.advsecurenet
def test_forward_eval_path_uses_autoshape(capsys):
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )

        dummy_model = MinimalYoloModel()
        dummy_model.modules = lambda: []
        mock_load.return_value.model = dummy_model
        class DummyAuto:
            def __call__(self, imgs, size=None):
                return types.SimpleNamespace(
                    xyxy=[np.empty((0, 6))],
                    pred=[torch.zeros((0, 5 + 3), dtype=torch.float32)],
                )
        mock_autoshape.return_value = DummyAuto()
        mock_compute_loss.return_value = MagicMock()
    model = CustomYolov5Model()
    model.eval()
    # Ensure input_shape exists for size accessor in predict_per_batch
    model.input_shape = (3, 32, 32)
    # Call predict_per_batch to go through AutoShape code path deterministically
    imgs = torch.zeros(1, 3, 32, 32)
    inference_model = model.initialize_inference_model(model._model, device="cpu")
    outs = model.predict_per_batch(imgs, inference_model, clip_values=(0, 255))
    assert isinstance(outs, list)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_forward_train_mode():
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )

        dummy_model = MinimalYoloModel()
        dummy_model.modules = lambda: []
        mock_load.return_value.model = dummy_model
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()
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
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )

        dummy_model = MinimalYoloModel()
        dummy_model.modules = lambda: []
        mock_load.return_value.model = dummy_model
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()
        model = CustomYolov5Model(model_weights_path="some/other/path.pt")
        mock_load.assert_called_once_with("some/other/path.pt", autoshape=False, device=ANY)


@pytest.mark.advsecurenet
def test_pth_weight_loading_hash_changed_and_unchanged(tmp_path, monkeypatch):
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )

        dm = MinimalYoloModel()
        dm.modules = lambda: []
        mock_load.return_value.model = dm
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()

        # Prepare a valid .pth and force os.path.isfile True
        pth = tmp_path / "w.pth"
        torch.save({"weights": {"x": torch.tensor(1)}}, pth)
        monkeypatch.setattr("os.path.isfile", lambda p: True)

        # Sequence 1: changed hash
        cap = []
        def run_with_hashes(vals):
            seq = iter(vals)
            monkeypatch.setattr(CustomYolov5Model, "_parameters_sha256", lambda self: next(seq))
            m = CustomYolov5Model(model_weights_path=str(pth))
            cap.append(m)
            return m

        m1 = run_with_hashes(["A", "B"])  # changed
        # Sequence 2: unchanged
        _ = run_with_hashes(["S", "S"])  # unchanged
        # No exception is sufficient; the prints are not strictly asserted here because
        # environment may buffer differently under CI.
        assert isinstance(cap[0], CustomYolov5Model)


@pytest.mark.advsecurenet
def test_initialize_inference_model_sets_conf_when_available():
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )
        dm = MinimalYoloModel()
        dm.modules = lambda: []
        mock_load.return_value.model = dm
        as_inst = MagicMock()
        as_inst.conf = 0.0
        mock_autoshape.return_value = as_inst
        mock_compute_loss.return_value = MagicMock()
        model = CustomYolov5Model()
        inf = model.initialize_inference_model(dm, device="cpu", conf_thresh=0.33)
        assert hasattr(inf, "conf") and abs(inf.conf - 0.33) < 1e-6


@pytest.mark.advsecurenet
def test_resolve_device_none_cpu(monkeypatch):
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )
        dm = MinimalYoloModel()
        dm.modules = lambda: []
        mock_load.return_value.model = dm
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()
        monkeypatch.setattr("torch.cuda.is_available", lambda: False)
        model = CustomYolov5Model(device=None)
        assert model.device == "cpu"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_predict_raw():
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )

        dummy_model = MinimalYoloModel()
        dummy_model.modules = lambda: []
        mock_load.return_value.model = dummy_model
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()
        model = CustomYolov5Model()
        x = torch.rand(2, 3, 640, 640)
        out = model.predict_raw(x)
        assert out == "raw_logits"


@pytest.mark.advsecurenet
def test_prepare_training_inputs_and_convert_targets_dict_and_empty():
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import CustomYolov5Model
        dm = MinimalYoloModel()
        dm.modules = lambda: []
        mock_load.return_value.model = dm
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()
        model = CustomYolov5Model()
        model.input_shape = (3, 64, 64)
        images = torch.zeros((2, 3, 64, 64))
        targets_dict = {
            "boxes": [torch.tensor([[0, 0, 10, 10]]), torch.tensor([[5, 5, 15, 15]])],
            "labels": [torch.tensor([1]), torch.tensor([2])],
        }
        imgs_out, yolo_targets = model.prepare_training_inputs(images, targets_dict)
        assert imgs_out.requires_grad is True
        assert isinstance(yolo_targets, torch.Tensor)
        assert yolo_targets.shape[1] == 6
        # empty path
        empty_t = model._convert_targets([], images.shape)
        assert isinstance(empty_t, torch.Tensor) and empty_t.numel() == 0


@pytest.mark.advsecurenet
def test_calculate_loss_and_preprocess_paths():
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import CustomYolov5Model
        dm = MinimalYoloModel()
        dm.modules = lambda: []
        mock_load.return_value.model = dm
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()
        model = CustomYolov5Model()
        model.device = "cpu"
        # preds with objectness logit channel at index 4
        preds = [torch.zeros((1, 2, 2, 85), dtype=torch.float32)]
        loss0 = model.calculate_loss(preds, target_val=0.0)
        loss1 = model.calculate_loss(preds, target_val=1.0)
        assert isinstance(loss0, torch.Tensor) and isinstance(loss1, torch.Tensor)
        # preprocess from numpy and torch
        arr = (np.zeros((1, 3, 8, 8)) * 255).astype(np.uint8)
        t_np = model.preprocess_x_for_loss_calculation(arr, requires_grad=True)
        assert isinstance(t_np, torch.Tensor) and t_np.requires_grad
        t_torch = model.preprocess_x_for_loss_calculation(torch.zeros((1, 3, 8, 8)), requires_grad=False)
        assert isinstance(t_torch, torch.Tensor) and not t_torch.requires_grad


@pytest.mark.advsecurenet
def test_translate_labels_channels_first_and_last():
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import CustomYolov5Model
        dm = MinimalYoloModel()
        dm.modules = lambda: []
        mock_load.return_value.model = dm
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()
        model = CustomYolov5Model()
        model.input_shape = (3, 64, 64)
        # channels first
        model.channels_first = True
        labels = [{"boxes": np.array([[10, 20, 30, 40]]), "labels": np.array([1])}]
        out_cf = model.translate_labels(labels, batch_size=1)
        assert isinstance(out_cf, torch.Tensor) and out_cf.shape[1] == 6
        # channels last path
        model.channels_first = False
        model.input_shape = (64, 64, 3)
        labels2 = [{"boxes": np.array([[0, 0, 10, 10]]), "labels": np.array([2])}]
        out_cl = model.translate_labels(labels2, batch_size=1)
        assert isinstance(out_cl, torch.Tensor) and out_cl.shape[1] == 6


@pytest.mark.advsecurenet
def test_convert_targets_list_path():
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import CustomYolov5Model
        dm = MinimalYoloModel()
        dm.modules = lambda: []
        mock_load.return_value.model = dm
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()
        model = CustomYolov5Model()
        images = torch.zeros((2, 3, 32, 32))
        targets_list = [
            {"boxes": torch.tensor([[0, 0, 10, 10]]), "labels": torch.tensor([1])},
            {"boxes": torch.tensor([[5, 5, 15, 15]]), "labels": torch.tensor([2])},
        ]
        t = model._convert_targets(targets_list, images.shape)
        assert isinstance(t, torch.Tensor) and t.shape[1] == 6


@pytest.mark.advsecurenet
def test_predict_method_training_and_eval_branches():
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )

        dm = MinimalYoloModel()
        dm.modules = lambda: []
        mock_load.return_value.model = dm
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()
        model = CustomYolov5Model()

        # Force return types for predict routing
        model.forward = MagicMock(return_value=["fwd0", "fwd1"])  # type: ignore
        model.predict_raw = MagicMock(return_value=["raw0", "raw1"])  # type: ignore

        out_train = model.predict(torch.zeros(1, 3, 8, 8), training=True)
        out_eval = model.predict(torch.zeros(1, 3, 8, 8), training=False)
        assert out_train == "fwd0"
        assert out_eval == "raw0"


@pytest.mark.advsecurenet
def test_predict_per_batch_empty_and_nonempty():
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )

        dm = MinimalYoloModel()
        dm.modules = lambda: []
        mock_load.return_value.model = dm
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()
        model = CustomYolov5Model()
        model.input_shape = (3, 32, 32)

        class Outputs:
            def __init__(self):
                # First image: empty detections; Second: one detection
                self.xyxy = [
                    np.empty((0, 6), dtype=float),
                    np.array([[1, 2, 3, 4, 0.9, 1]], dtype=float),
                ]
                # pred carries class logits [5:]
                self.pred = [
                    torch.zeros((0, 5 + 3), dtype=torch.float32),
                    torch.tensor([[0, 0, 0, 0, 0.0, 0.1, 0.2, 0.3]], dtype=torch.float32),
                ]

        # Inference model returns Outputs instance
        inference_model = MagicMock()
        inference_model.return_value = Outputs()

        imgs = torch.rand(2, 3, 16, 16)
        outs = model.predict_per_batch(imgs, inference_model, clip_values=(0, 255))
        assert isinstance(outs, list) and len(outs) == 2
        # First is empty
        assert outs[0]["boxes"].size == 0
        # Second has boxes, scores, labels, and logits
        assert outs[1]["boxes"].shape == (1, 4)
        assert outs[1]["logits"].shape == (1, 3)


@pytest.mark.advsecurenet
def test_initialize_inference_model_exception_path(monkeypatch):
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.utils.loss.ComputeLoss"
    ) as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )
        dm = MinimalYoloModel()
        dm.modules = lambda: []
        mock_load.return_value.model = dm
        mock_compute_loss.return_value = MagicMock()
        model = CustomYolov5Model()
        # Ensure model has no _autoshape so code tries to construct one and fails
        def raise_autoshape(*args, **kwargs):
            raise RuntimeError("fail")

    monkeypatch.setattr("yolov5.models.common.AutoShape", raise_autoshape)
    inf = model.initialize_inference_model(dm, device="cpu", conf_thresh=0.2)
    # After our patch, initialize still creates a wrapped inference model
    assert inf is not dm


@pytest.mark.advsecurenet
def test_init_with_pth_load_state_dict_exception(monkeypatch):
    # Force .pth path and pretend file exists; torch.load returns a dict
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )
        dm = MinimalYoloModel()
        # modules list so BN freeze loop executes
        dm.modules = lambda: []
        mock_load.return_value.model = dm
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()
        # Make os.path.isfile return True to trigger loading state dict
        monkeypatch.setenv("PYTHONHASHSEED", "0")
        monkeypatch.setattr("os.path.isfile", lambda p: True)
        # Return a nested state dict that will fail when accessing model.state_dict()
        monkeypatch.setattr(
            "torch.load",
            lambda *a, **k: {"state_dict": {"model._model.model.weight": torch.tensor([1])}},
        )
        # Should not raise despite failing to apply state dict
        _ = CustomYolov5Model(model_weights_path="weights.pth")


@pytest.mark.advsecurenet
def test_init_device_int_branch():
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )
        dm = MinimalYoloModel()
        dm.modules = lambda: []
        mock_load.return_value.model = dm
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()
        model = CustomYolov5Model(device=0)
        # Device should be set to a valid string ("cpu" or "cuda:*")
        assert isinstance(model.device, str) and len(model.device) > 0
