from advsecurenet.models.od_wrapper import ODWrapper


def get_object_detector(config: dict = None, existing_model=None):
    config = config or {}
    detector = ODWrapper(
        model=existing_model,
        input_shape=tuple(config.get("input_shape", (3, 640, 640))),
        device_type=config.get("device_type", "cuda:0"),
        clip_values=tuple(config.get("clip_values", (0, 255))),
        attack_losses=tuple(config.get("attack_losses", ("loss_total", "loss_classifier", "loss_box_reg", "loss_objectness", "loss_rpn_box_reg"))),
        conf_thresh=config.get("conf_thresh", 0.25),       
    )
    return detector

    
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
