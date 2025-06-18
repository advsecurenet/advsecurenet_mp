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
        self.model=model
        self.conf_thresh=conf_thresh
        self.device=device_type
        self.clip_values=clip_values
        self.input_shape = input_shape


    def filter_boxes(self, predictions, conf_thresh):
        dictionary = {}
        boxes_list = []
        scores_list = []
        labels_list = []
        for i in range(len(predictions["boxes"])):
            score = predictions["scores"][i]
            if score >= conf_thresh:
                boxes_list.append(predictions["boxes"][i])
                scores_list.append(predictions["scores"][[i]])
                labels_list.append(predictions["labels"][[i]])
        if len(boxes_list)>0 and len(scores_list)>0 and len(labels_list)>0:
            dictionary["boxes"] = np.vstack(boxes_list)
            dictionary["scores"] = np.hstack(scores_list)
            dictionary["labels"] = np.hstack(labels_list)
        y = dictionary
        return y
    

    @abstractmethod
    def compute_object_vanishing_gradient(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        pass


    @abstractmethod
    def compute_object_untargeted_gradient(self, x: np.ndarray, detections: dict = None) -> np.ndarray:
        pass


    @abstractmethod
    def compute_object_fabrication_gradient(self, x: np.ndarray, detections: dict = None) -> np.ndarray:
        pass


    @abstractmethod
    def compute_object_mislabeling_gradient(self, x: np.ndarray, detections: dict = None) -> np.ndarray:
        pass