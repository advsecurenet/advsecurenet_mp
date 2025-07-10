import pytest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from advsecurenet.models.CustomODWrappers.CustomYolov5ODWrapper import CustomYolov5ODWrapper

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
