import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from advsecurenet.models.CustomODWrappers.CustomYolov5ODWrapper import CustomYolov5ODWrapper
import types
from advsecurenet.models.CustomODWrappers import ODWrapper

@pytest.fixture
def dummy_model():
    model = MagicMock()
    model.train = MagicMock()
    model.zero_grad = MagicMock()
    def model_call(x, y=None):
        return {
            'loss_total': torch.tensor(1.0),
            'loss_box': torch.tensor(0.5),
            'loss_obj': torch.tensor(0.3),
            'loss_cls': torch.tensor(0.2),
        }
    model.__call__ = MagicMock(side_effect=model_call)
    model.predict_raw = MagicMock(return_value=[[torch.zeros((1, 3, 3, 9))]])
    return model

@pytest.fixture
def wrapper(dummy_model):
    with patch('yolov5.load') as mock_load:
        mock_inference = MagicMock()
        mock_inference.conf = 0.7
        mock_inference.model = MagicMock()
        mock_inference.model.names = [str(i) for i in range(80)]
        mock_inference.__call__ = MagicMock()
        mock_inference.xyxy = [torch.zeros((0, 6))]
        mock_inference.pred = [torch.zeros((0, 85))]
        mock_load.return_value = mock_inference
        return CustomYolov5ODWrapper(
            model=dummy_model,
            input_shape=(3, 224, 224),
            clip_values=(0, 255),
            attack_losses=('loss_total', 'loss_cls', 'loss_box', 'loss_obj'),
            device_type='cpu',
            conf_thresh=0.7,
        )

def test_constructor_sets_attributes(wrapper):
    assert wrapper.input_shape == (3, 224, 224)
    assert wrapper.clip_values == (0, 255)
    assert wrapper.conf_thresh == 0.7
    assert wrapper.device == 'cpu'
    assert wrapper.attack_losses == ('loss_total', 'loss_cls', 'loss_box', 'loss_obj')
    assert wrapper.channels_first is True

def test_translate_labels_channels_first(wrapper):
    labels = [{
        'boxes': np.array([[10, 20, 30, 40]]),
        'labels': np.array([1])
    }]
    result = wrapper._translate_labels(labels)
    assert isinstance(result, torch.Tensor)
    assert result.shape[1] == 6

def test_translate_labels_channels_last(wrapper):
    wrapper.channels_first = False
    labels = [{
        'boxes': torch.tensor([[10, 20, 30, 40]]),
        'labels': torch.tensor([1])
    }]
    result = wrapper._translate_labels(labels)
    assert isinstance(result, torch.Tensor)
    assert result.shape[1] == 6

def test_translate_labels_empty(wrapper):
    # torch.vstack([]) raises RuntimeError, so expect it
    with pytest.raises(RuntimeError):
        wrapper._translate_labels([])

def test_translate_predictions_numpy(wrapper):
    preds = [np.zeros((1, 85))]
    result = wrapper._translate_predictions(preds)
    assert isinstance(result, list)
    assert all(isinstance(d, dict) for d in result)

def test_translate_predictions_list(wrapper):
    preds = [[0]*85]
    result = wrapper._translate_predictions([preds])
    assert isinstance(result, list)
    assert all(isinstance(d, dict) for d in result)

def test_translate_predictions_empty(wrapper):
    result = wrapper._translate_predictions([])
    assert result == []

def test_translate_predictions_list_input(wrapper):
    preds = [[[0.0]*85]]
    result = wrapper._translate_predictions(preds)
    assert isinstance(result, list)
    assert all(isinstance(d, dict) for d in result)

def test_get_losses_numpy():
    from advsecurenet.models.CustomODWrappers.CustomYolov5ODWrapper import CustomYolov5ODWrapper
    import torch
    import numpy as np
    from unittest.mock import MagicMock, patch
    dummy_model = MagicMock()
    dummy_model.train = lambda: None
    dummy_model.zero_grad = MagicMock()
    dummy_model.predict_raw = MagicMock(return_value=[[torch.zeros((1, 3, 3, 9))]])
    with patch('yolov5.load') as mock_load:
        mock_inference = MagicMock()
        mock_inference.conf = 0.7
        mock_inference.model = MagicMock()
        mock_inference.model.names = [str(i) for i in range(80)]
        mock_inference.__call__ = MagicMock()
        mock_inference.xyxy = [torch.zeros((0, 6))]
        mock_inference.pred = [torch.zeros((0, 85))]
        mock_load.return_value = mock_inference
        wrapper = CustomYolov5ODWrapper(
            model=dummy_model,
            input_shape=(3, 224, 224),
            clip_values=(0, 255),
            attack_losses=('loss_total', 'loss_cls', 'loss_box', 'loss_obj'),
            device_type='cpu',
            conf_thresh=0.7,
        )
        # Set the side_effect so that calling wrapper.model(x, y) returns a dict
        wrapper.model.side_effect = lambda x, y=None: {
            'loss_total': torch.tensor(1.0),
            'loss_box': torch.tensor(0.5),
            'loss_obj': torch.tensor(0.3),
            'loss_cls': torch.tensor(0.2),
        }
        x = np.zeros((1, 3, 224, 224), dtype=np.float32)
        y = [{'boxes': np.zeros((1, 4)), 'labels': np.zeros((1,))}]
        loss_components, x_pre = wrapper._get_losses(x, y)
        assert isinstance(loss_components, dict)
        assert isinstance(x_pre, torch.Tensor)

def test_get_losses_torch():
    from advsecurenet.models.CustomODWrappers.CustomYolov5ODWrapper import CustomYolov5ODWrapper
    import torch
    from unittest.mock import MagicMock, patch
    dummy_model = MagicMock()
    dummy_model.train = lambda: None
    dummy_model.zero_grad = MagicMock()
    dummy_model.predict_raw = MagicMock(return_value=[[torch.zeros((1, 3, 3, 9))]])
    with patch('yolov5.load') as mock_load:
        mock_inference = MagicMock()
        mock_inference.conf = 0.7
        mock_inference.model = MagicMock()
        mock_inference.model.names = [str(i) for i in range(80)]
        mock_inference.__call__ = MagicMock()
        mock_inference.xyxy = [torch.zeros((0, 6))]
        mock_inference.pred = [torch.zeros((0, 85))]
        mock_load.return_value = mock_inference
        wrapper = CustomYolov5ODWrapper(
            model=dummy_model,
            input_shape=(3, 224, 224),
            clip_values=(0, 255),
            attack_losses=('loss_total', 'loss_cls', 'loss_box', 'loss_obj'),
            device_type='cpu',
            conf_thresh=0.7,
        )
        # Set the side_effect so that calling wrapper.model(x, y) returns a dict
        wrapper.model.side_effect = lambda x, y=None: {
            'loss_total': torch.tensor(1.0),
            'loss_box': torch.tensor(0.5),
            'loss_obj': torch.tensor(0.3),
            'loss_cls': torch.tensor(0.2),
        }
        x = torch.zeros((1, 3, 224, 224))
        y = [{'boxes': torch.zeros((1, 4)), 'labels': torch.zeros((1,))}]
        loss_components, x_pre = wrapper._get_losses(x, y)
        assert isinstance(loss_components, dict)
        assert isinstance(x_pre, torch.Tensor)

def test_loss_gradient_numpy(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    y = [{'boxes': np.zeros((1, 4)), 'labels': np.zeros((1,))}]
    with patch.object(wrapper.model, '__call__', return_value={
        'loss_total': torch.tensor(1.0),
        'loss_box': torch.tensor(0.5),
        'loss_obj': torch.tensor(0.3),
        'loss_cls': torch.tensor(0.2),
    }):
        with patch('torch.autograd.grad', return_value=[torch.zeros_like(torch.from_numpy(x))]):
            with patch('builtins.sum', return_value=torch.tensor(1.0)):
                with patch.object(torch.Tensor, 'backward', return_value=None):
                    with pytest.raises(ValueError, match='Gradient term in PyTorch model is `None`.'):
                        wrapper.loss_gradient(x, y)

def test_loss_gradient_torch(wrapper):
    x = torch.zeros((1, 3, 224, 224))
    y = [{'boxes': torch.zeros((1, 4)), 'labels': torch.zeros((1,))}]
    with patch.object(wrapper.model, '__call__', return_value={
        'loss_total': torch.tensor(1.0),
        'loss_box': torch.tensor(0.5),
        'loss_obj': torch.tensor(0.3),
        'loss_cls': torch.tensor(0.2),
    }):
        with patch('torch.autograd.grad', return_value=[torch.zeros_like(x)]):
            with patch('builtins.sum', return_value=torch.tensor(1.0)):
                with patch.object(torch.Tensor, 'backward', return_value=None):
                    with pytest.raises(ValueError, match='Gradient term in PyTorch model is `None`.'):
                        wrapper.loss_gradient(x, y)

def test_loss_gradient_numpy_grad(wrapper):
    x = np.ones((1, 3, 224, 224), dtype=np.float32)
    y = [{'boxes': np.zeros((1, 4)), 'labels': np.zeros((1,))}]
    class DummyTensor(torch.Tensor):
        @property
        def grad(self):
            return torch.ones_like(torch.from_numpy(x))
    # Use a tensor with requires_grad=True for loss
    loss = torch.tensor(1.0, requires_grad=True)
    with patch.object(wrapper, '_get_losses', return_value=({'loss_total': loss}, DummyTensor())):
        with patch('builtins.sum', return_value=loss):
            out = wrapper.loss_gradient(x, y)
            assert isinstance(out, np.ndarray)
            assert out.shape == x.shape

def test_loss_gradient_torch_grad(wrapper):
    x = torch.ones((1, 3, 224, 224))
    y = [{'boxes': torch.zeros((1, 4)), 'labels': torch.zeros((1,))}]
    class DummyTensor(torch.Tensor):
        @property
        def grad(self):
            return torch.ones_like(x)
    loss = torch.tensor(1.0, requires_grad=True)
    with patch.object(wrapper, '_get_losses', return_value=({'loss_total': loss}, DummyTensor())):
        with patch('builtins.sum', return_value=loss):
            out = wrapper.loss_gradient(x, y)
            assert isinstance(out, torch.Tensor)
            assert out.shape == x.shape

def test_predict_numpy(wrapper):
    x = np.zeros((2, 3, 224, 224), dtype=np.float32)
    with patch.object(wrapper.inference_model, '__call__', return_value=MagicMock(xyxy=[torch.zeros((0, 6)), torch.zeros((1, 6))], pred=[torch.zeros((0, 85)), torch.zeros((1, 85))])):
        preds = wrapper.predict(x, batch_size=1)
        assert isinstance(preds, list)
        assert all(isinstance(d, dict) for d in preds)

def test_predict_empty(wrapper):
    x = np.zeros((0, 3, 224, 224), dtype=np.float32)
    preds = wrapper.predict(x)
    assert isinstance(preds, list)
    assert len(preds) == 0

def test_predict_mixed_empty_and_nonempty(wrapper):
    x = np.zeros((2, 3, 224, 224), dtype=np.float32)
    class DummyOutputs:
        xyxy = [torch.zeros((0, 6)), torch.ones((1, 6))]
        pred = [torch.zeros((0, 85)), torch.ones((1, 85))]
    with patch.object(wrapper.inference_model, '__call__', return_value=DummyOutputs()):
        preds = wrapper.predict(x, batch_size=1)
        assert isinstance(preds, list)
        assert all(isinstance(d, dict) for d in preds)

def test_compute_loss_no_weight_dict(wrapper):
    x = torch.zeros((1, 3, 224, 224))
    y = [{'boxes': torch.zeros((1, 4)), 'labels': torch.zeros((1,))}]
    with patch.object(wrapper.model, '__call__', return_value={
        'loss_total': torch.tensor(1.0),
        'loss_box': torch.tensor(0.5),
        'loss_obj': torch.tensor(0.3),
        'loss_cls': torch.tensor(0.2),
    }):
        with patch('builtins.sum', return_value=torch.tensor(1.0)):
            loss = wrapper.compute_loss(x, y)
            assert isinstance(loss, torch.Tensor)

def test_compute_loss_with_weight_dict(wrapper):
    wrapper.weight_dict = {'loss_total': 1.0, 'loss_box': 0.5, 'loss_obj': 0.5, 'loss_cls': 0.5}
    x = torch.zeros((1, 3, 224, 224))
    y = [{'boxes': torch.zeros((1, 4)), 'labels': torch.zeros((1,))}]
    with patch.object(wrapper.model, '__call__', return_value={
        'loss_total': torch.tensor(1.0),
        'loss_box': torch.tensor(0.5),
        'loss_obj': torch.tensor(0.3),
        'loss_cls': torch.tensor(0.2),
    }):
        with patch('builtins.sum', return_value=torch.tensor(1.0)):
            loss = wrapper.compute_loss(x, y)
            assert isinstance(loss, torch.Tensor)

def test_compute_loss_with_weight_dict_numpy(wrapper):
    wrapper.weight_dict = {'loss_total': 1.0, 'loss_box': 0.5, 'loss_obj': 0.5, 'loss_cls': 0.5}
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    y = [{'boxes': np.zeros((1, 4)), 'labels': np.zeros((1,))}]
    with patch.object(wrapper.model, '__call__', return_value={
        'loss_total': torch.tensor(1.0),
        'loss_box': torch.tensor(0.5),
        'loss_obj': torch.tensor(0.3),
        'loss_cls': torch.tensor(0.2),
    }):
        with patch('builtins.sum', return_value=torch.tensor(1.0)):
            loss = wrapper.compute_loss(x, y)
            assert isinstance(loss, np.ndarray)

def test_compute_object_vanishing_gradient(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    with patch('torch.autograd.grad', return_value=[torch.zeros_like(torch.from_numpy(x))]):
        grad = wrapper.compute_object_vanishing_gradient(x)
        assert grad.shape == x.shape

def test_compute_object_untargeted_gradient_no_detections(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    grad = wrapper.compute_object_untargeted_gradient(x, detections=None)
    assert np.all(grad == 0)

def test_compute_object_untargeted_gradient_with_detections(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    det = [{'boxes': np.zeros((1, 4)), 'labels': np.zeros((1,))}]
    with patch.object(wrapper.model, '__call__', return_value={
        'loss_total': torch.tensor(1.0),
        'loss_box': torch.tensor(0.5),
        'loss_obj': torch.tensor(0.3),
        'loss_cls': torch.tensor(0.2),
    }):
        with patch('builtins.sum', return_value=torch.tensor(1.0)):
            with patch('torch.autograd.grad', return_value=[torch.zeros_like(torch.from_numpy(x))]):
                grad = wrapper.compute_object_untargeted_gradient(x, detections=det)
                assert grad.shape == x.shape

def test_compute_object_fabrication_gradient(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    with patch('torch.autograd.grad', return_value=[torch.zeros_like(torch.from_numpy(x))]):
        grad = wrapper.compute_object_fabrication_gradient(x)
        assert grad.shape == x.shape

def test_compute_object_mislabeling_gradient_no_detections(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    grad = wrapper.compute_object_mislabeling_gradient(x, detections=None)
    assert np.all(grad == 0)

def test_compute_object_mislabeling_gradient_with_detections(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    det = [{'boxes': np.zeros((1, 4)), 'labels': np.array([1]), 'logits': np.zeros((1, 80))}]
    with patch('torch.autograd.grad', return_value=[torch.zeros_like(torch.from_numpy(x))]):
        grad = wrapper.compute_object_mislabeling_gradient(x, detections=det, mode='ml')
        assert grad.shape == x.shape
    with patch('torch.autograd.grad', return_value=[torch.zeros_like(torch.from_numpy(x))]):
        grad = wrapper.compute_object_mislabeling_gradient(x, detections=det, mode='ll')
        assert grad.shape == x.shape

def test_compute_object_mislabeling_gradient_invalid_mode(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    det = [{'boxes': np.zeros((1, 4)), 'labels': np.array([1]), 'logits': np.zeros((1, 80))}]
    with patch('torch.autograd.grad', return_value=[torch.zeros_like(torch.from_numpy(x))]):
        grad = wrapper.compute_object_mislabeling_gradient(x, detections=det, mode='invalid')
        assert grad.shape == x.shape

def test_compute_object_mislabeling_gradient_unknown_mode(wrapper, capsys):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    det = [{'boxes': np.zeros((1, 4)), 'labels': np.array([1]), 'logits': np.zeros((1, 80))}]
    with patch('torch.autograd.grad', return_value=[torch.zeros_like(torch.from_numpy(x))]):
        grad = wrapper.compute_object_mislabeling_gradient(x, detections=det, mode='unknown')
        assert grad.shape == x.shape
        out = capsys.readouterr().out
        assert "Warning: Unknown mode" in out

def test_compute_object_mislabeling_gradient_out_of_range_label(wrapper, capsys):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    det = [{'boxes': np.zeros((1, 4)), 'labels': np.array([999]), 'logits': np.zeros((1, 80))}]
    with patch('torch.autograd.grad', return_value=[torch.zeros_like(torch.from_numpy(x))]):
        grad = wrapper.compute_object_mislabeling_gradient(x, detections=det, mode='ml')
        assert grad.shape == x.shape
        out = capsys.readouterr().out
        assert "Warning: Label 999 is out of range" in out

def test_compute_object_mislabeling_gradient_grad_tensor_none(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    det = [{'boxes': np.zeros((1, 4)), 'labels': np.array([1]), 'logits': np.zeros((1, 80))}]
    with patch('torch.autograd.grad', return_value=[None]):
        grad = wrapper.compute_object_mislabeling_gradient(x, detections=det, mode='ml')
        assert np.all(grad == 0)

def test_compute_object_untargeted_gradient_clip_values(wrapper):
    wrapper.clip_values = (0, 255)
    x = np.ones((1, 3, 224, 224), dtype=np.float32)
    det = [{'boxes': np.zeros((1, 4)), 'labels': np.zeros((1,))}]
    with patch.object(wrapper.model, '__call__', return_value={
        'loss_total': torch.tensor(1.0),
        'loss_box': torch.tensor(0.5),
        'loss_obj': torch.tensor(0.3),
        'loss_cls': torch.tensor(0.2),
    }):
        with patch('builtins.sum', return_value=torch.tensor(1.0)):
            with patch('torch.autograd.grad', return_value=[torch.ones_like(torch.from_numpy(x))]):
                grad = wrapper.compute_object_untargeted_gradient(x, detections=det)
                assert grad.shape == x.shape

def test_translate_labels_numpy_input(wrapper):
    labels = [{
        'boxes': np.array([[10, 20, 30, 40]]),
        'labels': np.array([1])
    }]
    result = wrapper._translate_labels(labels)
    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 6)

def test_translate_predictions_tensor_input(wrapper):
    pred_tensor = torch.zeros((1, 85))
    result = wrapper._translate_predictions([pred_tensor])
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert 'boxes' in result[0]

def test_translate_predictions_invalid_type(wrapper):
    with pytest.raises(TypeError):
        wrapper._translate_predictions([123])  # unsupported type

def test_translate_predictions_invalid_logits(wrapper):
    preds = [np.zeros((1, 6))]  # missing class logits
    result = wrapper._translate_predictions(preds)
    assert isinstance(result, list)
    assert isinstance(result[0], dict)
    assert 'logits' in result[0]

def test_compute_object_fabrication_gradient_inference_mode(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    wrapper.model.predict_raw = MagicMock(return_value=[[torch.zeros((1, 3, 3, 9))]])
    with patch('torch.autograd.grad', return_value=[torch.ones_like(torch.from_numpy(x))]):
        grads = wrapper.compute_object_fabrication_gradient(x, training=False)
        assert grads.shape == x.shape

def test_compute_object_mislabeling_gradient_no_logits_entry(wrapper):
    x = np.zeros((1, 3, 224, 224), dtype=np.float32)
    det = [{'boxes': np.zeros((1, 4)), 'labels': np.array([1])}]  # no logits key
    with patch.object(wrapper.model, '__call__', return_value={
        'loss_total': torch.tensor(1.0),
        'loss_box': torch.tensor(0.5),
        'loss_obj': torch.tensor(0.3),
        'loss_cls': torch.tensor(0.2),
    }):
        with patch('torch.autograd.grad', return_value=[torch.ones_like(torch.from_numpy(x))]):
            grad = wrapper.compute_object_mislabeling_gradient(x, detections=det, mode='ml')
            assert grad.shape == x.shape

def make_minimal_odwrapper():
    class MinimalODWrapper(ODWrapper.ODWrapper):
        def compute_object_vanishing_gradient(self, x, training=False):
            return x
        def compute_object_untargeted_gradient(self, x, detections=None):
            return x
        def compute_object_fabrication_gradient(self, x, detections=None):
            return x
        def compute_object_mislabeling_gradient(self, x, detections=None):
            return x
    return MinimalODWrapper(model=None, conf_thresh=0.5, device_type='cpu', clip_values=(0,255), input_shape=(3,224,224))

def test_default_model_weights_path():
    from advsecurenet.models.CustomODWrappers.CustomYolov5ODWrapper import CustomYolov5ODWrapper
    from unittest.mock import patch, MagicMock
    with patch('yolov5.load') as mock_load:
        mock_inference = MagicMock()
        mock_inference.conf = 0.7
        mock_inference.model = MagicMock()
        mock_inference.model.names = [str(i) for i in range(80)]
        mock_inference.__call__ = MagicMock()
        mock_inference.xyxy = [torch.zeros((0, 6))]
        mock_inference.pred = [torch.zeros((0, 85))]
        mock_load.return_value = mock_inference
        CustomYolov5ODWrapper(
            model=MagicMock(),
            input_shape=(3, 224, 224),
            clip_values=(0, 255),
            attack_losses=('loss_total', 'loss_cls', 'loss_box', 'loss_obj'),
            device_type='cpu',
            conf_thresh=0.7,
        )
        # Should call yolov5.load with the correct path
        from pathlib import Path
        expected_path = str(Path("model_weights") / "yolov5s.pt")
        mock_load.assert_called_with(expected_path, device='cpu', autoshape=True)

def test_filter_boxes_all_above_conf():
    wrapper = make_minimal_odwrapper()
    preds = {
        "boxes": [np.array([1,2,3,4]), np.array([5,6,7,8])],
        "scores": np.array([0.9, 0.8]),
        "labels": np.array([1, 2]),
    }
    out = wrapper.filter_boxes(preds, conf_thresh=0.5)
    assert "boxes" in out and "scores" in out and "labels" in out
    assert out["boxes"].shape[0] == 2
    assert out["scores"].shape[0] == 2
    assert out["labels"].shape[0] == 2

def test_filter_boxes_some_below_conf():
    wrapper = make_minimal_odwrapper()
    preds = {
        "boxes": [np.array([1,2,3,4]), np.array([5,6,7,8])],
        "scores": np.array([0.9, 0.1]),
        "labels": np.array([1, 2]),
    }
    out = wrapper.filter_boxes(preds, conf_thresh=0.5)
    assert out["boxes"].shape[0] == 1
    assert out["scores"].shape[0] == 1
    assert out["labels"].shape[0] == 1

def test_filter_boxes_all_below_conf():
    wrapper = make_minimal_odwrapper()
    preds = {
        "boxes": [np.array([1,2,3,4]), np.array([5,6,7,8])],
        "scores": np.array([0.1, 0.2]),
        "labels": np.array([1, 2]),
    }
    out = wrapper.filter_boxes(preds, conf_thresh=0.5)
    assert out == {}

def test_filter_boxes_empty_lists():
    wrapper = make_minimal_odwrapper()
    preds = {"boxes": [], "scores": np.array([]), "labels": np.array([])}
    out = wrapper.filter_boxes(preds, conf_thresh=0.5)
    assert out == {}

def test_filter_boxes_missing_keys():
    wrapper = make_minimal_odwrapper()
    preds = {"boxes": [], "scores": np.array([])}  # missing labels
    out = wrapper.filter_boxes(preds, conf_thresh=0.5)
    assert out == {}
