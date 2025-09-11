import numpy as np
import torch
from abc import ABC, abstractmethod


class ODWrapper(ABC):
    def __init__(
        self,
        model,
        conf_thresh,
        device_type,
        clip_values,
        input_shape,
    ):
        self.model = model
        self.conf_thresh = conf_thresh
        self.device = device_type
        self.clip_values = clip_values
        self.input_shape = input_shape

    def filter_boxes(self, predictions, conf_thresh):
        mask = predictions["scores"] >= conf_thresh
        indices = np.where(mask)[0]
        boxes_list = [predictions["boxes"][i] for i in indices]
        scores_list = [predictions["scores"][i] for i in indices]
        labels_list = [predictions["labels"][i] for i in indices]
        if predictions["label_names"] is not None:
            label_names_list = [predictions["label_names"][i] for i in indices]
        dictionary = {}
        if boxes_list:
            dictionary["boxes"] = np.vstack(boxes_list)
            dictionary["scores"] = np.hstack(scores_list)
            dictionary["labels"] = np.hstack(labels_list)
            if label_names_list is not None:
                dictionary["label_names"] = np.hstack(label_names_list)
        return dictionary
    
    @abstractmethod
    def prepare_training_inputs(self, images: torch.Tensor, targets: list[dict]):
        pass

    @abstractmethod
    def extract_total_loss(self, model_output) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_object_vanishing_gradient(
        self, x: np.ndarray, training: bool = False
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_object_untargeted_gradient(
        self, x: np.ndarray, detections: dict = None
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_object_fabrication_gradient(
        self, x: np.ndarray, detections: dict = None
    ) -> np.ndarray:
        pass

    @abstractmethod
    def compute_object_mislabeling_gradient(
        self, x: np.ndarray, detections: dict = None
    ) -> np.ndarray:
        pass
