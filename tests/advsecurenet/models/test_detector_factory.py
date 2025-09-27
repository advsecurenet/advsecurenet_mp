import pytest
from advsecurenet.models import detector_factory as df


class _FakeWrapper:
    def __init__(self, **kwargs):
        # capture all args for assertions
        self.kwargs = kwargs


def test_get_object_detector_defaults(monkeypatch):
    captured = {}

    def fake_odwrapper(**kwargs):
        captured.update(kwargs)
        return _FakeWrapper(**kwargs)

    monkeypatch.setattr(df, "ODWrapper", lambda **kw: fake_odwrapper(**kw))

    detector = df.get_object_detector()
    assert isinstance(detector, _FakeWrapper)
    # defaults
    assert captured["model"] is None
    assert captured["input_shape"] == (3, 640, 640)
    assert captured["device_type"] == "cuda:0"
    assert captured["clip_values"] == (0, 255)
    assert captured["conf_thresh"] == 0.25
    assert isinstance(captured["attack_losses"], tuple)
    # ensure default losses include common keys
    for k in ("loss_total", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"):
        assert k in captured["attack_losses"]


def test_get_object_detector_with_config_and_existing_model(monkeypatch):
    captured = {}

    def fake_odwrapper(**kwargs):
        captured.update(kwargs)
        return _FakeWrapper(**kwargs)

    monkeypatch.setattr(df, "ODWrapper", lambda **kw: fake_odwrapper(**kw))

    cfg = {
        "input_shape": (3, 320, 320),
        "device_type": "cpu",
        "clip_values": (0, 1),
        "attack_losses": ("loss_total",),
        "conf_thresh": 0.5,
    }
    existing = object()
    detector = df.get_object_detector(cfg, existing_model=existing)
    assert isinstance(detector, _FakeWrapper)
    assert captured["model"] is existing
    assert captured["input_shape"] == (3, 320, 320)
    assert captured["device_type"] == "cpu"
    assert captured["clip_values"] == (0, 1)
    assert captured["attack_losses"] == ("loss_total",)
    assert captured["conf_thresh"] == 0.5


def test_infer_wrapper_name_attribute_override():
    class Model:
        _detector_wrapper = "custom_wrapper"

    assert df.infer_wrapper_name(Model()) == "custom_wrapper"


def test_infer_wrapper_name_from_classname_mapping_yolo():
    # Dynamically create a class with the expected name
    CustomYolov5Model = type("CustomYolov5Model", (), {})
    assert df.infer_wrapper_name(CustomYolov5Model()) == "yolov5"


def test_infer_wrapper_name_from_classname_mapping_fasterrcnn():
    CustomFasterRCNNModel = type("CustomFasterRCNNModel", (), {})
    assert df.infer_wrapper_name(CustomFasterRCNNModel()) == "fasterrcnn_resnet50_fpn"


def test_infer_wrapper_name_unknown_raises():
    class Unknown:
        pass

    with pytest.raises(ValueError, match=r"Cannot infer detector wrapper"):
        df.infer_wrapper_name(Unknown())
