from advsecurenet.models.CustomModels.CustomYolov5Model import CustomYolov5Model
from advsecurenet.models.CustomModels.CustomFasterRCNNModel import CustomFasterRCNNModel
from advsecurenet.models.CustomODWrappers.CustomYolov5ODWrapper import (
    CustomYolov5ODWrapper,
)

from advsecurenet.models.CustomODWrappers.CustomFasterRCNNODWrapper import CustomFasterRCNNODWrapper

def get_object_detector(name: str, config: dict = None):
    name = name.lower()
    config = config or {}
    if name == "yolov5":
        model_weights_path = config.get(
            "model_weights_path", "model_weights/yolov5s.pt"
        )
        model = CustomYolov5Model(model_weights_path=model_weights_path)
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
            device_type=config.get("device_type", "cuda:0"),
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
