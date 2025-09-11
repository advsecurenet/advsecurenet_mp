from advsecurenet.models.CustomModels.CustomYolov5Model import CustomYolov5Model
from advsecurenet.models.CustomModels.CustomFasterRCNNModel import CustomFasterRCNNModel
from advsecurenet.models.CustomODWrappers.CustomYolov5ODWrapper import (
    CustomYolov5ODWrapper,
)

from advsecurenet.models.CustomODWrappers.CustomFasterRCNNODWrapper import CustomFasterRCNNODWrapper

_YOLO_SINGLETON_CACHE = {}


def get_object_detector(name: str, config: dict = None, existing_model=None):
    name = name.lower()
    config = config or {}
    if name == "yolov5":
        model_weights_path = config.get("model_weights_path", "model_weights/yolov5s.pt")
        device = config.get("device_type", "cuda:0")
        # Reuse if provided explicitly
        if existing_model is not None:
            model = existing_model
            try:
                model.to(device)
            except Exception:
                pass
        else:
            cache_key = (model_weights_path, device)
            if config.get("reuse", True) and cache_key in _YOLO_SINGLETON_CACHE:
                model = _YOLO_SINGLETON_CACHE[cache_key]
            else:
                model = CustomYolov5Model(model_weights_path=model_weights_path, device=device)
                if config.get("reuse", True):
                    _YOLO_SINGLETON_CACHE[cache_key] = model
        detector = CustomYolov5ODWrapper(
            model=model,
            conf_thresh=config.get("conf_thresh", 0.25),
            input_shape=tuple(config.get("input_shape", (3, 640, 640))),
            clip_values=tuple(config.get("clip_values", (0, 255))),
            attack_losses=tuple(
                config.get(
                    "attack_losses", ("loss_total", "loss_cls", "loss_box", "loss_obj")
                )
            ),
            device_type=device,
        )
        return detector
    elif name == "fasterrcnn_resnet50_fpn":
        num_classes = config.get("num_classes", 91)  # pretrained fasterRCNN COCO classes (incl. background etc.)
        pretrained = config.get("pretrained", True)
        model = CustomFasterRCNNModel(num_classes=num_classes, pretrained=pretrained)
        detector = CustomFasterRCNNODWrapper(
            model=model,
            conf_thresh=config.get("conf_thresh", 0.7),
            input_shape=tuple(config.get("input_shape", (3, 800, 800))),
            clip_values=tuple(config.get("clip_values", (0, 255))),
            attack_losses=tuple(config.get("attack_losses", ("loss_total", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"))),
            device_type=config.get("device_type", "cuda:0"),
        )
        return detector
    # Add more mappings as needed
    else:
        raise ValueError(f"Unknown object detector: {name}")
    
DETECTOR_CLASS_TO_WRAPPER = {
    "customyolov5model": "yolov5",
    "customfasterrcnnmodel": "fasterrcnn_resnet50_fpn",
}

def infer_wrapper_name(model) -> str:
    if hasattr(model, "_detector_wrapper"):
        return getattr(model, "_detector_wrapper")
    cls_name = model.__class__.__name__.lower()
    for key, wrapper in DETECTOR_CLASS_TO_WRAPPER.items():
        if key in cls_name:
            return wrapper
    raise ValueError(
        f"Cannot infer detector wrapper for model class '{model.__class__.__name__}'. "
        "Add 'detector_wrapper' to adversarial_training config."
    )
