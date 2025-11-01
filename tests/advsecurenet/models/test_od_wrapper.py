import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from advsecurenet.models.od_wrapper import ODWrapper


@pytest.fixture
def dummy_model():
    model = MagicMock()
    # Set explicit num_classes so production wrapper picks it up as int (avoids MagicMock value)
    model.num_classes = 91

    # inference model initializer
    def init_inference(m, device_type, conf_thresh):
        class Inference:
            def eval(self):  # mimic nn.Module interface used in wrapper.predict
                return self
        inf = Inference()
        return inf

    model.initialize_inference_model = MagicMock(side_effect=init_inference)

    # methods used by wrapper
    model.prepare_training_inputs = MagicMock(side_effect=lambda x, y: (x, y))
    model.preprocess_x_for_loss_calculation = MagicMock(
        side_effect=lambda x, requires_grad=False: torch.zeros(
            (x.shape[0], x.shape[1], x.shape[2], x.shape[3]) if isinstance(x, torch.Tensor) else (1, 3, 224, 224),
            requires_grad=requires_grad,
            dtype=torch.float32,
        )
    )
    model.translate_labels = MagicMock(side_effect=lambda y, batch_size=None: y)
    model.predict_per_batch = MagicMock(side_effect=lambda xb, inf, clip: [
        {"boxes": np.empty((0, 4), dtype=np.float32), "scores": np.empty((0,), dtype=np.float32), "labels": np.empty((0,), dtype=np.int64)}
        for _ in range(xb.shape[0])
    ])
    model.predict = MagicMock(side_effect=lambda x, training=False: {"dummy": True})
    model.calculate_loss = MagicMock(side_effect=lambda preds, target_val=0.0: torch.tensor(0.0, requires_grad=True))
    model.train = MagicMock()
    model.zero_grad = MagicMock()
    return model


@pytest.fixture
def wrapper(dummy_model):
    w = ODWrapper(
        model=dummy_model,
        input_shape=(3, 224, 224),
        device_type="cpu",
        clip_values=(0, 255),
        # Include only losses we will supply in dummy outputs to ensure tensor accumulation
        attack_losses=("loss_total", "loss_box", "loss_obj", "loss_cls"),
        conf_thresh=0.7,
        weight_dict=None,
    )
    return w


def test_constructor_sets_attributes(wrapper):
    assert wrapper.input_shape == (3, 224, 224)
    assert wrapper.clip_values == (0, 255)
    assert wrapper.conf_thresh == 0.7
    assert wrapper.device == "cpu"
    # Ordering matches fixture definition
    assert wrapper.attack_losses == ("loss_total", "loss_box", "loss_obj", "loss_cls")
    assert wrapper.channels_first is True
    # With no num_classes attribute on the underlying model, wrapper defaults to 91
    assert wrapper.num_classes == 91


def test_prepare_training_inputs_delegation(wrapper):
    x = torch.zeros((1, 3, 224, 224))
    y = [{"boxes": torch.zeros((1, 4)), "labels": torch.zeros((1,), dtype=torch.long)}]
    out = wrapper.prepare_training_inputs(x, y)
    assert out == (x, y)


def test_filter_boxes_basic_masking(wrapper):
    preds = {
        "boxes": np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32),
        "scores": np.array([0.9, 0.1], dtype=np.float32),
        "labels": np.array([1, 2], dtype=np.int64),
    }
    out = wrapper.filter_boxes(preds, conf_thresh=0.5)
    assert out["boxes"].shape[0] == 1
    assert out["scores"].shape[0] == 1
    assert out["labels"].shape[0] == 1


def test_filter_boxes_missing_keys_returns_empty_arrays(wrapper):
    preds = {"scores": np.array([], dtype=np.float32)}
    out = wrapper.filter_boxes(preds, conf_thresh=0.5)
    assert out["boxes"].shape == (0, 4)
    assert out["scores"].shape == (0,)
    assert out["labels"].shape == (0,)


def test_extract_total_loss_with_total_key(wrapper):
    loss = wrapper.extract_total_loss({"loss_total": torch.tensor(1.0)})
    assert isinstance(loss, torch.Tensor)


def test_extract_total_loss_sums_components(wrapper):
    out = wrapper.extract_total_loss({"loss_box": torch.tensor(0.5), "loss_obj": torch.tensor(0.3), "loss_cls": torch.tensor(0.2)})
    assert isinstance(out, torch.Tensor)
    assert torch.isclose(out, torch.tensor(1.0))


def test__should_freeze_bn(monkeypatch, wrapper):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    assert wrapper._should_freeze_bn() is True


def test__freeze_bn_sets_eval_and_no_grad(wrapper):
    # Build a small model with BN
    m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU())
    wrapper.model = m
    wrapper._freeze_bn()
    # BN should be eval and params not require grad
    for mod in wrapper.model.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            assert mod.training is False
            for p in mod.parameters():
                assert p.requires_grad is False


def test__get_losses_and_compute_loss_torch(wrapper, dummy_model):
    # prepare model call to return dict of losses
    dummy_model.return_value = {
        "loss_total": torch.tensor(1.0, requires_grad=True),
        "loss_box": torch.tensor(0.5),
        "loss_obj": torch.tensor(0.3),
        "loss_cls": torch.tensor(0.2),
    }
    x = torch.zeros((1, 3, 224, 224))
    y = [{"boxes": torch.zeros((1, 4)), "labels": torch.zeros((1,), dtype=torch.long)}]
    loss_components, x_pre = wrapper._get_losses(x, y)
    assert isinstance(loss_components, dict)
    assert isinstance(x_pre, torch.Tensor)
    loss = wrapper.compute_loss(x, y)
    assert isinstance(loss, torch.Tensor)


def test__get_losses_and_compute_loss_numpy(wrapper, dummy_model):
    dummy_model.return_value = {
        "loss_total": torch.tensor(1.0, requires_grad=True),
        "loss_box": torch.tensor(0.5),
        "loss_obj": torch.tensor(0.3),
        "loss_cls": torch.tensor(0.2),
    }
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    y = [{"boxes": np.zeros((1, 4)), "labels": np.zeros((1,), dtype=np.int64)}]
    loss_components, x_pre = wrapper._get_losses(x, y)
    assert isinstance(loss_components, dict)
    assert isinstance(x_pre, torch.Tensor)
    loss = wrapper.compute_loss(x, y)
    assert isinstance(loss, np.ndarray)


def test_compute_loss_with_weight_dict(wrapper, dummy_model):
    wrapper.weight_dict = {"loss_total": 1.0, "loss_box": 0.5, "loss_obj": 0.5, "loss_cls": 0.5}
    dummy_model.return_value = {
        "loss_total": torch.tensor(1.0),
        "loss_box": torch.tensor(0.5),
        "loss_obj": torch.tensor(0.3),
        "loss_cls": torch.tensor(0.2),
    }
    x = torch.zeros((1, 3, 224, 224))
    y = [{"boxes": torch.zeros((1, 4)), "labels": torch.zeros((1,), dtype=torch.long)}]
    loss = wrapper.compute_loss(x, y)
    assert isinstance(loss, torch.Tensor)


def test_loss_gradient_numpy_returns_numpy(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    y = [{"boxes": np.zeros((1, 4)), "labels": np.zeros((1,), dtype=np.int64)}]

    class DummyGradTensor(torch.Tensor):
        @property
        def grad(self):
            return torch.zeros_like(torch.from_numpy(x))

    with patch.object(wrapper, "_get_losses", return_value=({"loss_total": torch.tensor(1.0, requires_grad=True)}, DummyGradTensor())):
        with patch("builtins.sum", return_value=torch.tensor(1.0, requires_grad=True)):
            out = wrapper.loss_gradient(x, y)
            assert isinstance(out, np.ndarray)
            assert out.shape == x.shape


def test_loss_gradient_torch_returns_tensor(wrapper):
    x = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    y = [{"boxes": torch.zeros((1, 4)), "labels": torch.zeros((1,), dtype=torch.long)}]

    class DummyGradTensor(torch.Tensor):
        @property
        def grad(self):
            return torch.zeros_like(x)

    with patch.object(wrapper, "_get_losses", return_value=({"loss_total": torch.tensor(1.0, requires_grad=True)}, DummyGradTensor())):
        with patch("builtins.sum", return_value=torch.tensor(1.0, requires_grad=True)):
            out = wrapper.loss_gradient(x, y)
            assert isinstance(out, torch.Tensor)
            assert out.shape == x.shape


def test_loss_gradient_raises_when_grad_none(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    y = [{"boxes": np.zeros((1, 4)), "labels": np.zeros((1,), dtype=np.int64)}]

    class DummyGradTensor(torch.Tensor):
        @property
        def grad(self):
            return None

    with patch.object(wrapper, "_get_losses", return_value=({"loss_total": torch.tensor(1.0, requires_grad=True)}, DummyGradTensor())):
        with patch("builtins.sum", return_value=torch.tensor(1.0, requires_grad=True)):
            with pytest.raises(ValueError, match="Gradient term in PyTorch model is `None`."):
                wrapper.loss_gradient(x, y)


def test_predict_numpy_input(wrapper, dummy_model):
    x = np.zeros((2, 3, 224, 224), dtype=np.float32)
    # model.preprocess_x_for_loss_calculation already mocked to return zeros tensor
    preds = wrapper.predict(x, batch_size=1)
    assert isinstance(preds, list)
    assert all(isinstance(d, dict) for d in preds)


def test_predict_torch_input(wrapper):
    x = torch.zeros((2, 3, 224, 224), dtype=torch.float32)
    preds = wrapper.predict(x.numpy(), batch_size=1)
    assert isinstance(preds, list)
    assert all(isinstance(d, dict) for d in preds)


def test_compute_object_vanishing_gradient(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    with patch("torch.autograd.grad", return_value=[torch.zeros_like(torch.from_numpy(x))]):
        grad = wrapper.compute_object_vanishing_gradient(x, training=True)
        assert isinstance(grad, np.ndarray)
        assert grad.shape == x.shape


def test_compute_object_untargeted_gradient_no_detections(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    grad = wrapper.compute_object_untargeted_gradient(x, detections=None)
    assert np.all(grad == 0)


def test_compute_object_untargeted_gradient_with_detections(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    det = [{"boxes": np.zeros((1, 4)), "labels": np.zeros((1,), dtype=np.int64)}]
    with patch.object(wrapper, "compute_loss", return_value=torch.tensor(1.0, requires_grad=True)):
        with patch("torch.autograd.grad", return_value=[torch.zeros_like(torch.from_numpy(x))]):
            grad = wrapper.compute_object_untargeted_gradient(x, detections=det)
            assert isinstance(grad, np.ndarray)
            assert grad.shape == x.shape


def test_compute_object_fabrication_gradient(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    with patch("torch.autograd.grad", return_value=[torch.zeros_like(torch.from_numpy(x))]):
        grad = wrapper.compute_object_fabrication_gradient(x, training=False)
        assert isinstance(grad, np.ndarray)
        assert grad.shape == x.shape


def test_compute_object_mislabeling_gradient_paths(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    # no detections -> zeros
    g0 = wrapper.compute_object_mislabeling_gradient(detections=None, x=x)
    assert np.all(g0 == 0)
    # with targets list -> use _get_losses
    target_labels_list = [{"boxes": np.zeros((1, 4)), "labels": np.array([2], dtype=np.int64)}]

    class DummyGradTensor(torch.Tensor):
        @property
        def grad(self):
            return torch.zeros_like(torch.from_numpy(x))

    with patch.object(wrapper, "_get_losses", return_value=({"loss_total": torch.tensor(1.0, requires_grad=True)}, DummyGradTensor())):
        grad = wrapper.compute_object_mislabeling_gradient(
            detections=[{"boxes": np.zeros((1, 4)), "labels": np.array([1], dtype=np.int64)}],
            x=x,
            target_labels_list=target_labels_list,
            training=True,
        )
        assert isinstance(grad, np.ndarray)
        assert grad.shape == x.shape


@pytest.mark.advsecurenet
def test_wrapper_init_without_num_classes(dummy_model):
    del dummy_model.num_classes
    w = ODWrapper(
        model=dummy_model,
        input_shape=(3, 224, 224),
        device_type="cpu",
        clip_values=(0, 255),
        attack_losses=("loss_total",),
        conf_thresh=0.7,
    )
    assert w.num_classes == 91  # default


@pytest.mark.advsecurenet
def test_filter_boxes_with_label_names(wrapper):
    preds = {
        "boxes": np.array([[1, 2, 3, 4]], dtype=np.float32),
        "scores": np.array([0.9], dtype=np.float32),
        "labels": np.array([1], dtype=np.int64),
        "label_names": np.array(["person"], dtype=object),
    }
    out = wrapper.filter_boxes(preds, conf_thresh=0.5)
    assert "label_names" in out
    assert len(out["label_names"]) == 1


@pytest.mark.advsecurenet
def test_filter_boxes_label_names_empty(wrapper):
    preds = {
        "boxes": np.empty((0, 4), dtype=np.float32),
        "scores": np.empty((0,), dtype=np.float32),
        "labels": np.empty((0,), dtype=np.int64),
        "label_names": np.array([], dtype=object),
    }
    out = wrapper.filter_boxes(preds, conf_thresh=0.5)
    assert "label_names" in out
    assert len(out["label_names"]) == 0


@pytest.mark.advsecurenet
def test_compute_object_vanishing_gradient_training_false(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    with patch("torch.autograd.grad", return_value=[torch.zeros_like(torch.from_numpy(x))]):
        grad = wrapper.compute_object_vanishing_gradient(x, training=False)
        assert isinstance(grad, np.ndarray)
        assert grad.shape == x.shape


@pytest.mark.advsecurenet
def test_compute_object_vanishing_gradient_non_yolov5(wrapper, dummy_model):
    dummy_model.expects_numpy_images = False
    dummy_model.predict = MagicMock(return_value={"dummy": True})
    wrapper.model = dummy_model
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    with patch("torch.autograd.grad", return_value=[torch.zeros_like(torch.from_numpy(x))]):
        grad = wrapper.compute_object_vanishing_gradient(x, training=False)
        assert isinstance(grad, np.ndarray)


@pytest.mark.advsecurenet
def test_compute_object_vanishing_gradient_no_clip_values(wrapper):
    wrapper.clip_values = None
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    with patch("torch.autograd.grad", return_value=[torch.zeros_like(torch.from_numpy(x))]):
        grad = wrapper.compute_object_vanishing_gradient(x, training=False)
        assert isinstance(grad, np.ndarray)


@pytest.mark.advsecurenet
def test_compute_object_untargeted_gradient_training_false(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    det = [{"boxes": np.zeros((1, 4)), "labels": np.zeros((1,), dtype=np.int64)}]
    with patch.object(wrapper, "compute_loss", return_value=torch.tensor(1.0, requires_grad=True)):
        with patch("torch.autograd.grad", return_value=[torch.zeros_like(torch.from_numpy(x))]):
            grad = wrapper.compute_object_untargeted_gradient(x, detections=det, training=False)
            assert isinstance(grad, np.ndarray)


@pytest.mark.advsecurenet
def test_compute_object_untargeted_gradient_no_clip_values(wrapper):
    wrapper.clip_values = None
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    det = [{"boxes": np.zeros((1, 4)), "labels": np.zeros((1,), dtype=np.int64)}]
    with patch.object(wrapper, "compute_loss", return_value=torch.tensor(1.0, requires_grad=True)):
        with patch("torch.autograd.grad", return_value=[torch.zeros_like(torch.from_numpy(x))]):
            grad = wrapper.compute_object_untargeted_gradient(x, detections=det, training=True)
            assert isinstance(grad, np.ndarray)


@pytest.mark.advsecurenet
def test_compute_object_fabrication_gradient_training_true(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    with patch("torch.autograd.grad", return_value=[torch.zeros_like(torch.from_numpy(x))]):
        grad = wrapper.compute_object_fabrication_gradient(x, training=True)
        assert isinstance(grad, np.ndarray)


@pytest.mark.advsecurenet
def test_compute_object_fabrication_gradient_non_yolov5(wrapper, dummy_model):
    dummy_model.expects_numpy_images = False
    dummy_model.predict = MagicMock(return_value={"dummy": True})
    wrapper.model = dummy_model
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    with patch("torch.autograd.grad", return_value=[torch.zeros_like(torch.from_numpy(x))]):
        grad = wrapper.compute_object_fabrication_gradient(x, training=False)
        assert isinstance(grad, np.ndarray)


@pytest.mark.advsecurenet
def test_compute_object_fabrication_gradient_no_clip_values(wrapper):
    wrapper.clip_values = None
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    with patch("torch.autograd.grad", return_value=[torch.zeros_like(torch.from_numpy(x))]):
        grad = wrapper.compute_object_fabrication_gradient(x, training=False)
        assert isinstance(grad, np.ndarray)


@pytest.mark.advsecurenet
def test_compute_object_mislabeling_gradient_no_targets(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    det = [{"boxes": np.zeros((1, 4)), "labels": np.zeros((1,), dtype=np.int64)}]
    grad = wrapper.compute_object_mislabeling_gradient(detections=det, x=x, target_labels_list=None)
    assert np.all(grad == 0)


@pytest.mark.advsecurenet
def test_compute_object_mislabeling_gradient_empty_labels(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    det = [{"boxes": np.zeros((1, 4)), "labels": np.array([], dtype=np.int64)}]
    grad = wrapper.compute_object_mislabeling_gradient(detections=det, x=x, target_labels_list=None)
    assert np.all(grad == 0)


@pytest.mark.advsecurenet
def test_compute_object_mislabeling_gradient_training_false(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    target_labels_list = [{"boxes": np.zeros((1, 4)), "labels": np.array([2], dtype=np.int64)}]
    
    class DummyGradTensor(torch.Tensor):
        @property
        def grad(self):
            return torch.zeros_like(torch.from_numpy(x))

    with patch.object(wrapper, "_get_losses", return_value=({"loss_total": torch.tensor(1.0, requires_grad=True)}, DummyGradTensor())):
        grad = wrapper.compute_object_mislabeling_gradient(
            detections=[{"boxes": np.zeros((1, 4)), "labels": np.array([1], dtype=np.int64)}],
            x=x,
            target_labels_list=target_labels_list,
            training=False,
        )
        assert isinstance(grad, np.ndarray)


@pytest.mark.advsecurenet
def test_compute_object_mislabeling_gradient_no_clip_values(wrapper):
    wrapper.clip_values = None
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    target_labels_list = [{"boxes": np.zeros((1, 4)), "labels": np.array([2], dtype=np.int64)}]
    
    class DummyGradTensor(torch.Tensor):
        @property
        def grad(self):
            return torch.zeros_like(torch.from_numpy(x))

    with patch.object(wrapper, "_get_losses", return_value=({"loss_total": torch.tensor(1.0, requires_grad=True)}, DummyGradTensor())):
        grad = wrapper.compute_object_mislabeling_gradient(
            detections=[{"boxes": np.zeros((1, 4)), "labels": np.array([1], dtype=np.int64)}],
            x=x,
            target_labels_list=target_labels_list,
            training=True,
        )
        assert isinstance(grad, np.ndarray)


@pytest.mark.advsecurenet
def test_compute_loss_weight_dict_path(wrapper, dummy_model):
    wrapper.weight_dict = {"loss_total": 2.0, "loss_box": 0.5}
    dummy_model.return_value = {
        "loss_total": torch.tensor(1.0, requires_grad=True),
        "loss_box": torch.tensor(0.5),
    }
    x = torch.zeros((1, 3, 224, 224))
    y = [{"boxes": torch.zeros((1, 4)), "labels": torch.zeros((1,), dtype=torch.long)}]
    loss = wrapper.compute_loss(x, y)
    assert isinstance(loss, torch.Tensor)


@pytest.mark.advsecurenet
def test_predict_no_clip_values(wrapper, dummy_model):
    wrapper.clip_values = None
    x = np.zeros((2, 3, 224, 224), dtype=np.float32)
    preds = wrapper.predict(x, batch_size=1)
    assert isinstance(preds, list)