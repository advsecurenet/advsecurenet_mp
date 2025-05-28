import datasets
from datasets import load_dataset
import numpy as np
from PIL import Image

import torch
from torch import nn
from tqdm.auto import trange

from advsecurenet.shared.types.configs.attack_configs.tog_attack_config import TOGAttackConfig
from advsecurenet.computer_vision.object_detection.attacks.base.object_detection_attack import ObjectDetectionAttack
from advsecurenet.models.base_model import BaseModel


class TOG(ObjectDetectionAttack):
    """
    TOG attack
    """

    def __init__(self, config: TOGAttackConfig) -> None:
        super().__init__(config)
        self.object_detector = config.object_detector
        self.max_iter = config.max_iter
        self.eps = config.eps
        self.eps_iter = config.eps_iter

    def attack(
            self,
            x: np.array,  # (batch_size, channels, height, width)
            target_label: torch.tensor,
            mask: torch.tensor,
            *args,
            **kwargs
    ) -> torch.tensor:
        """
        Generates adversarial examples using the TOG attack.
        Args:
            model (BaseModel): The model to attack.
            x (torch.tensor): The original input tensor. Expected shape is (batch_size, channels, height, width).
            y (torch.tensor): The true labels for the input tensor. Expected shape is (batch_size, num_boxes, 4) (x1, y1, x2, y2).

        Returns:
            torch.tensor: The adversarial example tensor.
        """
        if np.max(x) > 1.0:
            x = x / 255.0
        return self.tog_vanishing(x_query=x, n_iter=self.max_iter, eps=self.eps, eps_iter=self.eps_iter)
    
    def tog_vanishing(self, x_query, n_iter=10, eps=8/255., eps_iter=2/255.):
        eta = np.random.uniform(-eps, eps, size=x_query.shape)
        x_adv = np.clip(x_query + eta, 0.0, 1.0)
        for _ in range(n_iter):
            grad = self.object_detector.compute_object_vanishing_gradient(x_adv, training=False)
            signed_grad = np.sign(grad)
            x_adv -= eps_iter * signed_grad
            eta = np.clip(x_adv - x_query, -eps, eps)
            x_adv = np.clip(x_query + eta, 0.0, 1.0)
        return x_adv
    
    def tog_fabrication(self, x_query, n_iter=10, eps=8/255., eps_iter=2/255.):
        eta = np.random.uniform(-eps, eps, size=x_query.shape)
        x_adv = np.clip(x_query + eta, 0.0, 1.0)
        for _ in range(n_iter):
            grad = self.object_detector.compute_object_fabrication_gradient(x_adv)
            signed_grad = np.sign(grad)
            x_adv -= eps_iter * signed_grad
            eta = np.clip(x_adv - x_query, -eps, eps)
            x_adv = np.clip(x_query + eta, 0.0, 1.0)
        return x_adv