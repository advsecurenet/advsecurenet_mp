"""
This code was adapted from the original implementation of the TOG attack:
https://github.com/git-disl/TOG
The TOG-universal and TOG-patch attacks have been skipped in this implementation.
"""

import numpy as np

import torch
from scipy.special import softmax
from enum import Enum

from advsecurenet.shared.types.configs.attack_configs.tog_attack_config import TOGAttackConfig
from advsecurenet.computer_vision.object_detection.attacks.base.object_detection_attack import ObjectDetectionAttack


class TOGAttackType(Enum):
    VANISHING = "vanishing"
    FABRICATION = "fabrication"
    MISLABELING = "mislabeling"
    UNTARGETED = "untargeted"


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


    @staticmethod
    def generate_attack_targets(y: list[dict[str, np.ndarray]], mode: str = "ll", confidence_threshold: float = 0.5, class_id: int | None = None, **kwargs) -> np.ndarray:
        """
        For each detection, keep the original box, but set the class to the adversarial class.
        Output: [batch_idx, adv_class, 1.0, x1, y1, x2, y2] for each detection.
        """
        assert mode.lower() in ['ml', 'll'], '`mode` should be one of `ML` or `LL`.'
        all_rows = []
        for img_idx, detection in enumerate(y):
            num_boxes = detection["boxes"].shape[0]
            if num_boxes == 0:
                continue
            logits = detection["logits"]  # [num_detections, num_classes]
            # Handle background class (if applicable)
            if logits.shape[1] % 10 == 1:
                logits[:, 0] = np.finfo(np.float32).max if mode.lower() == 'll' else np.finfo(np.float32).min
            # Find adversarial class
            if mode.lower() == 'll':
                adv_class = np.argmin(logits, axis=1)
            else:
                logits[softmax(logits, axis=1) > confidence_threshold] = np.finfo(np.float32).min
                adv_class = np.argmax(logits, axis=1)
            # Optionally only target a specific class
            if class_id is not None:
                if logits.shape[1] % 10 == 1:
                    class_id += 1
                orig_class = detection["labels"]
                mask = orig_class == class_id
                if not mask.any():
                    continue
                adv_class = np.where(mask, adv_class, orig_class)
            confs = np.ones_like(adv_class, dtype=np.float32)
            batch_idx_col = np.full(len(adv_class), img_idx, dtype=np.int32)
            # Use original boxes!
            rows = np.column_stack([
                batch_idx_col, adv_class, confs, detection["boxes"]
            ])
            all_rows.append(rows)
        if not all_rows:
            return np.zeros((0, 7), dtype=np.float32)
        return np.vstack(all_rows).astype(np.float32)


    def attack(
            self,
            x: np.ndarray,  # (batch_size, channels, height, width)
            target_label: torch.Tensor,
            mask: torch.Tensor,
            tog_variant: TOGAttackType,
            tog_mislabeling_mode: str = "ml",
            *args,
            **kwargs
    ) -> torch.Tensor:
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
        match tog_variant:
            case TOGAttackType.VANISHING:
                return self.tog_vanishing(x_query=x, n_iter=self.max_iter, eps=self.eps, eps_iter=self.eps_iter)
            case TOGAttackType.FABRICATION:
                return self.tog_fabrication(x_query=x, n_iter=self.max_iter, eps=self.eps, eps_iter=self.eps_iter)
            case TOGAttackType.MISLABELING:
                return self.tog_mislabeling(x_query=x, mode=tog_mislabeling_mode, n_iter=self.max_iter, eps=self.eps, eps_iter=self.eps_iter)
            case TOGAttackType.UNTARGETED:
                return self.tog_untargeted(x_query=x, n_iter=self.max_iter, eps=self.eps, eps_iter=self.eps_iter)


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
    

    def tog_mislabeling(self, x_query, mode, n_iter=10, eps=8/255., eps_iter=2/255.):
        # Handle batch of images by iterating
        if x_query.ndim == 4 and x_query.shape[0] > 1:
            adv_images_batch = []
            for i in range(x_query.shape[0]):
                # Process each image individually by calling the same function
                single_x_query_batch = np.expand_dims(x_query[i], axis=0)
                adv_img = self.tog_mislabeling(single_x_query_batch, mode, n_iter, eps, eps_iter)
                adv_images_batch.append(adv_img.squeeze(0))
            return np.stack(adv_images_batch, axis=0)
        # Original logic for a single image
        x_uint8 = (x_query * 255.0).clip(0,255).astype(np.uint8)
        detections_query = self.object_detector.predict(x_uint8)
        detections_target = TOG.generate_attack_targets(detections_query, mode=mode, confidence_threshold=self.object_detector.conf_thresh, class_id=None)
        if detections_target.shape[0] == 0:
            return x_query
        eta = np.random.uniform(-eps, eps, size=x_query.shape)
        x_adv = np.clip(x_query + eta, 0.0, 1.0)
        for _ in range(n_iter):
            grad = self.object_detector.compute_object_mislabeling_gradient(x_adv, detections=detections_target)
            signed_grad = np.sign(grad)
            x_adv -= eps_iter * signed_grad
            eta = np.clip(x_adv - x_query, -eps, eps)
            x_adv = np.clip(x_query + eta, 0.0, 1.0)
        return x_adv


    def tog_untargeted(self, x_query, n_iter=10, eps=8/255., eps_iter=2/255.):
        x_uint8 = (x_query * 255.0).clip(0,255).astype(np.uint8)
        detections_list = self.object_detector.predict(x_uint8)
        eta = np.random.uniform(-eps, eps, size=x_query.shape)
        x_adv = np.clip(x_query + eta, 0.0, 1.0)
        for _ in range(n_iter):
            grad = self.object_detector.compute_object_untargeted_gradient(x_adv, detections=detections_list)
            signed_grad = np.sign(grad)
            x_adv -= eps_iter * signed_grad
            eta = np.clip(x_adv - x_query, -eps, eps)
            x_adv = np.clip(x_query + eta, 0.0, 1.0)
        return x_adv
    