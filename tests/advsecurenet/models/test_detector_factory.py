import pytest
from unittest.mock import patch, MagicMock
from advsecurenet.models.detector_factory import get_object_detector


@patch("advsecurenet.models.detector_factory.CustomYolov5Model")
@patch("advsecurenet.models.detector_factory.CustomYolov5ODWrapper")
def test_get_object_detector_yolov5_default(mock_wrapper, mock_model):
    mock_model_instance = MagicMock()
    mock_wrapper_instance = MagicMock()
    mock_model.return_value = mock_model_instance
    mock_wrapper.return_value = mock_wrapper_instance
    detector = get_object_detector("yolov5")
    mock_model.assert_called_once()
    mock_wrapper.assert_called_once()
    assert detector == mock_wrapper_instance


@patch("advsecurenet.models.detector_factory.CustomYolov5Model")
@patch("advsecurenet.models.detector_factory.CustomYolov5ODWrapper")
def test_get_object_detector_yolov5_custom_config(mock_wrapper, mock_model):
    mock_model_instance = MagicMock()
    mock_wrapper_instance = MagicMock()
    mock_model.return_value = mock_model_instance
    mock_wrapper.return_value = mock_wrapper_instance
    config = {
        "model_weights_path": "custom/path.pt",
        "conf_thresh": 0.5,
        "input_shape": (3, 320, 320),
        "clip_values": (0, 1),
        "attack_losses": ("loss_total",),
        "device_type": "cpu",
    }
    detector = get_object_detector("yolov5", config)
    mock_model.assert_called_once_with(model_weights_path="custom/path.pt")
    mock_wrapper.assert_called_once_with(
        model=mock_model_instance,
        conf_thresh=0.5,
        input_shape=(3, 320, 320),
        clip_values=(0, 1),
        attack_losses=("loss_total",),
        device_type="cpu",
    )
    assert detector == mock_wrapper_instance


def test_get_object_detector_fasterrcnn():
    detector = get_object_detector("fasterrcnn_resnet50_fpn")
    assert detector is None


def test_get_object_detector_unknown():
    with pytest.raises(ValueError, match="Unknown object detector: unknown_detector"):
        get_object_detector("unknown_detector")
