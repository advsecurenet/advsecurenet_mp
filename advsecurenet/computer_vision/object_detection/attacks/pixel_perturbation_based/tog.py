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
                adv_class = []
                orig_class = detection["labels"]
                for i in range(num_boxes):
                    label = orig_class[i]
                    print(f"Detection {i}: label={label}, logits.shape={logits[i].shape}, logits.shape[1]={logits.shape[1]}")
                    if label >= logits.shape[1]:
                        print(f"WARNING: label {label} is out of bounds for logits with shape {logits.shape}")
                        continue  # Skip this detection
                    logit_row = logits[i].copy()
                    logit_row[label] = -np.inf  # Exclude original class
                    adv_class.append(np.argmax(logit_row))
                adv_class = np.array(adv_class)
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
        if mode.lower() not in ["ml", "ll"]:
            print(f"Warning: Unknown mode '{mode}'. Using 'ml' instead.")
            mode = "ml"
        x_uint8 = (x_query * 255.0).clip(0, 255).astype(np.uint8)
        x_tensor = torch.from_numpy(x_uint8).float().to(next(self.object_detector.model.parameters()).device)
        initial_detections = self.object_detector.predict(x_tensor)
        eta = np.random.uniform(-eps, eps, size=x_query.shape)
        x_adv = np.clip(x_query + eta, 0.0, 1.0)
        for i in range(n_iter):
            grad = self.object_detector.compute_object_mislabeling_gradient(x_adv, detections=initial_detections, mode=mode)
            grad_norm = np.linalg.norm(grad)
            if i % 50 == 0:  # Log every 50 iterations
                print(f"Iteration {i}: Gradient norm = {grad_norm:.6f}")
            if grad_norm < 1e-8:
                print("Warning: Very small gradients detected, stopping early")
                break
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
    