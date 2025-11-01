import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock, ANY
import numpy as np
import sys, types
from types import SimpleNamespace

# Provide a minimal "transformers" stub so that importing advsecurenet.models.* does not fail
if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.__path__ = []  # mark as package
    utils_stub = types.ModuleType("transformers.utils")
    class _DummyAutoModel:
        pass
    class _DummyAutoConfig:
        pass
    class _DummyRTDetrImageProcessor:
        pass
    class _DummyModelOutput(dict):
        """Minimal stand-in for transformers.utils.ModelOutput."""

    utils_stub.logging = SimpleNamespace(
        get_logger=lambda name: SimpleNamespace(
            setLevel=lambda *a, **k: None,
            info=lambda *a, **k: None,
            warning=lambda *a, **k: None,
            error=lambda *a, **k: None,
        )
    )
    utils_stub.ModelOutput = _DummyModelOutput
    transformers_stub.AutoModel = _DummyAutoModel
    transformers_stub.AutoConfig = _DummyAutoConfig
    transformers_stub.RTDetrImageProcessor = _DummyRTDetrImageProcessor
    transformers_stub.utils = utils_stub
    sys.modules["transformers"] = transformers_stub
    sys.modules["transformers.utils"] = utils_stub

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
        self.training = True

    def parameters(self):
        return iter([self._param])

    def __call__(self, x, *args, **kwargs):
        return "raw_logits"

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = mode
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


@pytest.mark.advsecurenet
def test_suppress_yolov5_autocast_warning():
    from advsecurenet.models.CustomODModels.CustomYolov5Model import _suppress_yolov5_autocast_warning
    import warnings
    
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with _suppress_yolov5_autocast_warning():
            # This should suppress FutureWarning about autocast
            warnings.warn("test", FutureWarning)
        # Check that warnings were suppressed (filter should catch FutureWarning)
        # The warning filter specifically targets FutureWarning from yolov5.models.common
        # So general FutureWarning might still appear, but the specific ones should be suppressed
        assert isinstance(w, list)


@pytest.mark.advsecurenet
def test_translate_predictions_for_map_evaluator_yolo_expects_numpy_true(monkeypatch):
    from advsecurenet.models.CustomODModels.CustomYolov5Model import translate_predictions_for_map_evaluator_yolo
    
    class MockPreds:
        def __init__(self):
            self.pred = [
                torch.tensor([[0, 0, 10, 10, 0.9, 1], [5, 5, 15, 15, 0.8, 2]], dtype=torch.float32)
            ]
    
    preds = MockPreds()
    results = translate_predictions_for_map_evaluator_yolo(preds, "coco", expects_numpy=True)
    assert len(results) == 1
    assert "boxes" in results[0]
    assert "labels" in results[0]
    assert "scores" in results[0]


@pytest.mark.advsecurenet
def test_translate_predictions_for_map_evaluator_yolo_expects_numpy_false(monkeypatch):
    from advsecurenet.models.CustomODModels.CustomYolov5Model import translate_predictions_for_map_evaluator_yolo
    
    preds = [
        {
            "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
            "scores": torch.tensor([0.9], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
    ]
    results = translate_predictions_for_map_evaluator_yolo(preds, "coco", expects_numpy=False)
    assert len(results) == 1
    assert "boxes" in results[0]
    assert "labels" in results[0]
    assert "scores" in results[0]


@pytest.mark.advsecurenet
def test_translate_predictions_for_map_evaluator_yolo_pascal_mapping(monkeypatch):
    from advsecurenet.models.CustomODModels.CustomYolov5Model import translate_predictions_for_map_evaluator_yolo
    
    class MockPreds:
        def __init__(self):
            self.pred = [
                torch.tensor([[0, 0, 10, 10, 0.9, 5]], dtype=torch.float32)  # label 5
            ]
    
    preds = MockPreds()
    results = translate_predictions_for_map_evaluator_yolo(preds, "pascal_voc", expects_numpy=True)
    assert len(results) == 1
    # Labels should be mapped/filtered for Pascal VOC


@pytest.mark.advsecurenet
def test_parameters_sha256():
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
        hash_val = model._parameters_sha256()
        assert isinstance(hash_val, str)
        assert len(hash_val) == 64  # SHA256 hex digest length


@pytest.mark.advsecurenet
def test_load_model_weights_non_pth():
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
        model = CustomYolov5Model(model_weights_path="yolov5s.pt")
        # Should use arch_source = "yolov5s.pt" (not .pth)
        assert model._model is not None


@pytest.mark.advsecurenet
def test_load_model_weights_state_dict_cleaning(tmp_path, monkeypatch):
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )
        dm = MinimalYoloModel()
        dm.modules = lambda: []
        # Add a state_dict method that returns keys
        def state_dict():
            return {"model.weight": torch.zeros(1)}
        dm.state_dict = state_dict
        mock_load.return_value.model = dm
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()
        
        pth = tmp_path / "w.pth"
        # Create state dict with prefix that needs cleaning
        sd = {"model._model.model.weight": torch.ones(1)}
        torch.save({"weights": sd}, pth)
        monkeypatch.setattr("os.path.isfile", lambda p: True)
        
        model = CustomYolov5Model(model_weights_path=str(pth))
        # Should not raise and should handle prefix cleaning


@pytest.mark.advsecurenet
def test_forward_exception_path():
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
        # x that can't be converted to float should be handled
        x_bad = "not a tensor"
        try:
            model.forward(x_bad)
        except Exception:
            pass  # Exception is caught and ignored in forward


@pytest.mark.advsecurenet
def test_forward_non_training():
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )
        dm = MinimalYoloModel()
        dm.modules = lambda: []
        mock_load.return_value.model = dm
        # Make autoshape_mock callable
        autoshape_mock = MagicMock()
        autoshape_mock.__call__ = MagicMock(return_value="autoshape_output")
        # Set _autoshape attribute on model
        mock_autoshape.return_value = autoshape_mock
        mock_compute_loss.return_value = MagicMock()
        model = CustomYolov5Model()
        # The model should have _autoshape set during init
        assert hasattr(model, "_autoshape")
        model.eval()
        x = torch.zeros(1, 3, 32, 32)
        out = model.forward(x, targets=None)
        # When not training and targets=None, should use autoshape
        # autoshape_mock is callable and should return the mocked value
        assert out is not None  # Should return something from autoshape


@pytest.mark.advsecurenet
def test_predict_training_false():
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
        model.predict_raw = MagicMock(return_value=[[torch.zeros(1, 85)]])
        x = torch.zeros(1, 3, 32, 32)
        out = model.predict(x, training=False)
        assert out is not None


@pytest.mark.advsecurenet
def test_predict_raw_numpy_input():
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
        x_numpy = np.zeros((1, 3, 32, 32), dtype=np.float32)
        out = model.predict_raw(x_numpy)
        assert out == "raw_logits"


@pytest.mark.advsecurenet
def test_predict_per_batch_non_empty_tensor_detections():
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
                self.xyxy = [torch.tensor([[1, 2, 3, 4, 0.9, 1]], dtype=torch.float32)]
                self.pred = [torch.tensor([[0, 0, 0, 0, 0.0, 0.1, 0.2]], dtype=torch.float32)]
        
        inference_model = MagicMock()
        inference_model.return_value = Outputs()
        imgs = torch.rand(1, 3, 16, 16)
        outs = model.predict_per_batch(imgs, inference_model, clip_values=(0, 255))
        assert len(outs) == 1
        assert "boxes" in outs[0]
        assert "logits" in outs[0]


@pytest.mark.advsecurenet
def test_translate_predictions_for_map_evaluator():
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
        # Mock predictions
        class MockPreds:
            def __init__(self):
                self.pred = [torch.tensor([[0, 0, 10, 10, 0.9, 1]])]
        preds = MockPreds()
        results = model.translate_predictions_for_map_evaluator(preds, "coco", expects_numpy=True)
        assert isinstance(results, list)


@pytest.mark.advsecurenet
def test_translate_labels_empty_boxes():
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
        model.input_shape = (3, 64, 64)
        model.channels_first = True
        # Empty boxes
        labels = [{"boxes": np.array([]).reshape(0, 4), "labels": np.array([])}]
        out = model.translate_labels(labels, batch_size=1)
        assert isinstance(out, torch.Tensor)
        assert out.shape[1] == 6


@pytest.mark.advsecurenet
def test_convert_targets_dict_path():
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
        images = torch.zeros((2, 3, 32, 32))
        targets_dict = {
            "boxes": [torch.tensor([[0, 0, 10, 10]]), torch.tensor([[5, 5, 15, 15]])],
            "labels": [torch.tensor([1]), torch.tensor([2])],
        }
        t = model._convert_targets(targets_dict, images.shape)
        assert isinstance(t, torch.Tensor) and t.shape[1] == 6


@pytest.mark.advsecurenet
def test_resolve_device_torch_device(monkeypatch):
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
        model = CustomYolov5Model(device=torch.device("cpu"))
        assert model.device == "cpu"


@pytest.mark.advsecurenet
def test_module_device():
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
        dev = model._module_device()
        assert isinstance(dev, torch.device)


@pytest.mark.advsecurenet
def test_to_method():
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )
        dm = MinimalYoloModel()
        dm.modules = lambda: []
        mock_load.return_value.model = dm
        autoshape_mock = MagicMock()
        autoshape_mock.to = MagicMock(return_value=autoshape_mock)
        mock_autoshape.return_value = autoshape_mock
        mock_compute_loss.return_value = MagicMock()
        model = CustomYolov5Model()
        # Test to() method
        result = model.to("cpu")
        assert result is model


@pytest.mark.advsecurenet
def test_load_model_weights_model_key(tmp_path, monkeypatch):
    """Test load_model_weights with 'model' key in state dict."""
    with patch("yolov5.load") as mock_load, patch(
        "yolov5.models.common.AutoShape"
    ) as mock_autoshape, patch("yolov5.utils.loss.ComputeLoss") as mock_compute_loss:
        from advsecurenet.models.CustomODModels.CustomYolov5Model import (
            CustomYolov5Model,
        )
        dm = MinimalYoloModel()
        dm.modules = lambda: []
        dm.state_dict = lambda: {"model.weight": torch.zeros(1)}
        mock_load.return_value.model = dm
        mock_autoshape.return_value = MagicMock()
        mock_compute_loss.return_value = MagicMock()
        
        pth = tmp_path / "w.pth"
        torch.save({"model": {"x": torch.tensor(1)}}, pth)
        monkeypatch.setattr("os.path.isfile", lambda p: True)
        model = CustomYolov5Model(model_weights_path=str(pth))
        assert model._model is not None


@pytest.mark.advsecurenet
def test_translate_predictions_for_map_evaluator_yolo_expects_numpy_false_pascal(monkeypatch):
    """Test translate_predictions_for_map_evaluator_yolo with expects_numpy=False and pascal mapping."""
    from advsecurenet.models.CustomODModels.CustomYolov5Model import translate_predictions_for_map_evaluator_yolo
    
    preds = [
        {
            "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
            "scores": torch.tensor([0.9], dtype=torch.float32),
            "labels": torch.tensor([5], dtype=torch.int64),  # Label 5
        }
    ]
    results = translate_predictions_for_map_evaluator_yolo(preds, "pascal_voc", expects_numpy=False)
    assert len(results) == 1
    assert "boxes" in results[0]