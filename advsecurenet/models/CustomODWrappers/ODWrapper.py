import numpy as np
import torch
from torch.nn import functional as F
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
        dictionary = {}
        if boxes_list:
            dictionary["boxes"] = np.vstack(boxes_list)
            dictionary["scores"] = np.hstack(scores_list)
            dictionary["labels"] = np.hstack(labels_list)
        return dictionary

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
