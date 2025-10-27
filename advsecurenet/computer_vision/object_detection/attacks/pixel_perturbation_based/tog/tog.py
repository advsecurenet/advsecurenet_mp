"""
This code was adapted from the original implementation of the TOG attack:
https://github.com/git-disl/TOG
The TOG-universal and TOG-patch attacks have been skipped in this implementation.
"""

import numpy as np

import torch
import logging
import warnings
from typing import Literal, List, Dict

from advsecurenet.shared.types.configs.attack_configs.tog_attack_config import (
    TOGAttackConfig,
)
from advsecurenet.computer_vision.base.adversarial_attack import AdversarialAttack
from advsecurenet.computer_vision.object_detection.attacks.pixel_perturbation_based.tog.tog_attack_type import (
    TOGAttackType,
)

logger = logging.getLogger(__name__)


class TOG(AdversarialAttack):
    """
    TOG attack. Pixel-perturbation-based, white-box and iterative attack for object detection.
    It optimizes image pixels directly using objectness-related gradients to achieve one of four goals:
      - Vanishing: remove existing objects.
      - Fabrication: create false positives.
      - Mislabeling: change predicted classes (ML/LL modes).
      - Untargeted: reduce detector performance without a specific target.

    Args:
        _object_detector: The detector wrapper providing predict() and gradient APIs. Should be inheriting from the ODWrapper class, e.g. CustomYolov5ODWrapper.
        _max_iter (int): Number of attack iterations.
        _eps (float): L-infinity perturbation bound (max deviation from original input).
        _eps_iter (float): Per-iteration step size for the PGD-style update.

    References:
        [1] Chow, Ka-Ho and Liu, Ling and Loper, Margaret and Bae, Juhyun and Gursoy, Mehmet Emre and Truex, Stacey and Wei, Wenqi and Wu, Yanzhao (2020).
        Adversarial Objectness Gradient Attacks in Real-time Object Detection Systems.
        2020 Second IEEE International Conference on Trust, Privacy and Security in Intelligent Systems and Applications (TPS-ISA), pages 263-272.

    """

    def __init__(self, config: TOGAttackConfig) -> None:
        super().__init__(config)
        self._object_detector = config.object_detector
        self._max_iter = config.max_iter
        self._eps = config.eps
        self._eps_iter = config.eps_iter

    def _new_label_from_logits(
        self,
        orig_label: int,
        obj_logits: torch.Tensor,
        num_classes: int,
        mode: Literal["ml", "ll"],
    ) -> int:
        """Select a new class from logits, excluding the original class.

        Pads logits if needed so orig_label is in range, then:
        - ll: set original class to +inf and take argmin,
        - ml: set original class to -inf and take argmax.

        Args:
            orig_label (int): Original class id.
            obj_logits (torch.Tensor): 1D logits tensor for the detection (length C).
            num_classes (int): Total number of classes C.
            mode (Literal["ml", "ll"]): "ml" (most-likely non-original) or "ll" (least-likely).

        Returns:
            int: Selected target class id.
        """
        if len(obj_logits) <= orig_label:
            pseudo_logits = torch.randn(
                max(num_classes, orig_label + 1), device=obj_logits.device
            )
            pseudo_logits[: len(obj_logits)] = obj_logits
            obj_logits = pseudo_logits
        if mode == "ll":
            # Least likely: find class with lowest logit value
            obj_logits_mod = obj_logits.clone()
            obj_logits_mod[orig_label] = float("inf")  # Exclude original class
            return torch.argmin(obj_logits_mod).item()
        else:  # mode == 'ml'
            # Most likely: find class with highest logit value (excluding original)
            obj_logits_mod = obj_logits.clone()
            obj_logits_mod[orig_label] = float("-inf")  # Exclude original class
            return torch.argmax(obj_logits_mod).item()

    def generate_mislabeling_targets(
        self,
        detections: List[Dict[str, np.ndarray]],
        num_classes: int,
        mode: Literal["ml", "ll"],
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Build per-image mislabeling targets for TOG.

        For each detection, keep the original bounding box and assign a new target
        class according to the selection mode:
        - "ll": choose the least-likely class (argmin over logits, excluding the original class).
        - "ml": choose the most-likely non-original class (argmax with the original suppressed).

        Args:
            detections: List of detection dicts for each image. Each dict should contain:
                - "boxes" (np.ndarray): Shape (Ki, 4) in [x1, y1, x2, y2].
                - "labels" (np.ndarray): Shape (Ki,) integer labels.
                - "logits" (np.ndarray | None): Shape (Ki, C) class logits per detection (optional).
            num_classes: Total number of classes C used by the detector.
            mode: Mislabeling selection strategy: "ml" (most-likely non-original)
                or "ll" (least-likely). Any other value falls back to "ml" with a warning.

        Returns:
            List[dict[str, torch.Tensor]]: One dict per image with:
                - "boxes": torch.FloatTensor of shape (Ki, 4) on the detector device.
                - "labels": torch.LongTensor of shape (Ki,) target labels on the detector device.
        """
        assert mode.lower() in ["ml", "ll"], f"Unknown mode '{mode}'."
        device = next(self._object_detector.model.parameters()).device
        target_labels_list = []
        for det in detections:
            if len(det.get("labels", [])) == 0:
                continue
            boxes = torch.tensor(det["boxes"], dtype=torch.float32, device=device)
            original_labels = torch.from_numpy(det["labels"]).long().to(device)
            # Choose target labels based on the specified mode
            new_labels = original_labels.clone()
            for i in range(len(original_labels)):
                orig_label = original_labels[i].item()
                # Safe random selection if orig_label is out of range
                if orig_label >= num_classes:
                    logger.warning(
                        "Label %d is out of range (num_classes=%d); picking a random alternative.",
                        int(orig_label),
                        int(num_classes),
                    )
                    new_labels[i] = np.random.randint(0, num_classes)
                    continue
                # Use prediction confidence to create pseudo-logits if not available
                if (
                    "logits" not in det
                    or det["logits"] is None
                    or i >= len(det["logits"])
                ):
                    # Generate random alternative class
                    new_label = orig_label
                    while new_label == orig_label:
                        new_label = np.random.randint(0, num_classes)
                    new_labels[i] = new_label
                else:
                    # Use actual logits for smart targeting
                    obj_logits = torch.from_numpy(det["logits"][i]).to(device)
                    new_labels[i] = self._new_label_from_logits(
                        orig_label=orig_label,
                        obj_logits=obj_logits,
                        num_classes=num_classes,
                        mode=mode,
                    )
            target_labels_list.append(
                {
                    "boxes": boxes,
                    "labels": new_labels,
                }
            )
        return target_labels_list

    def attack(
        self,
        x: np.ndarray,  # (batch_size, channels, height, width)
        tog_variant: TOGAttackType,
        tog_mislabeling_mode: str = "ml",
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
        if x.dtype == np.uint8:
            x = x.astype(np.float32) / 255.0
        elif np.max(x) > 1.1:  # assume [0, 255]
            x = x.astype(np.float32) / 255.0
        elif np.min(x) < -0.1:  # assume [-1, 1]
            x = (x.astype(np.float32) + 1.0) / 2.0
        else:
            x = x.astype(np.float32)
        x = np.clip(x, 0.0, 1.0)
        match tog_variant:
            case TOGAttackType.VANISHING:
                return self._tog_vanishing(
                    x_query=x,
                    n_iter=self._max_iter,
                    eps=self._eps,
                    eps_iter=self._eps_iter,
                )
            case TOGAttackType.FABRICATION:
                return self._tog_fabrication(
                    x_query=x,
                    n_iter=self._max_iter,
                    eps=self._eps,
                    eps_iter=self._eps_iter,
                )
            case TOGAttackType.MISLABELING:
                return self._tog_mislabeling(
                    x_query=x,
                    mode=tog_mislabeling_mode,
                    n_iter=self._max_iter,
                    eps=self._eps,
                    eps_iter=self._eps_iter,
                )
            case TOGAttackType.UNTARGETED:
                return self._tog_untargeted(
                    x_query=x,
                    n_iter=self._max_iter,
                    eps=self._eps,
                    eps_iter=self._eps_iter,
                )

    def _initialise_x_adv(self, x_query: np.ndarray, eps: float) -> np.ndarray:
        """
        Initialize perturbation and adversarial example within an L-infinity ball.

        Args:
            x_query (np.ndarray): Clean input batch in [0, 1] with shape (N, C, H, W).
            eps (float): L-infinity perturbation budget.

        Returns:
            np.ndarray: The initialized adversarial example in [0, 1].
        """
        eta = np.random.uniform(-eps, eps, size=x_query.shape)
        x_adv = np.clip(x_query + eta, 0.0, 1.0)
        return x_adv

    def _update_x_adv(
        self,
        grad: np.ndarray,
        eps_iter: float,
        x_query: np.ndarray,
        x_adv: np.ndarray,
        eps: float,
    ) -> np.ndarray:
        """
        Perform a single PGD-style update and projection to enforce the L-infinity constraint.
        Applies a sign-gradient step, then projects back into the L-infinity ball and [0, 1] range.

        Args:
            grad (np.ndarray): Per-pixel gradient for the current step, same shape as x_query.
            eps_iter (float): Step size for the current iteration.
            x_query (np.ndarray): Current adversarial batch in [0, 1] with shape (N, C, H, W).
            x_adv (np.ndarray): Adversarial batch in [0, 1] with shape (N, C, H, W).
            eps (float): L-infinity perturbation budget.

        Returns:
            np.ndarray: Updated adversarial batch in [0, 1], same shape as x_query.
        """
        signed_grad = np.sign(grad)
        x_adv -= eps_iter * signed_grad
        eta = np.clip(x_adv - x_query, -eps, eps)
        x_adv = np.clip(x_query + eta, 0.0, 1.0)
        return x_adv

    def _resolve_num_classes(self, initial_detections):
        inf = getattr(self._object_detector, "inference_model", None)
        for obj in [inf, getattr(inf, "model", None)]:
            if obj is None:
                continue
            names = getattr(obj, "names", None)
            if isinstance(names, (list, tuple)) and len(names) > 0:
                return len(names)
            if isinstance(names, dict) and len(names) > 0:
                return len(names.keys())
        for attr_owner in [self._object_detector, inf]:
            if attr_owner is None:
                continue
            nc = getattr(attr_owner, "num_classes", None)
            if isinstance(nc, int) and nc > 0:
                return nc
        detected = [
            int(det["labels"].max())
            for det in initial_detections
            if det.get("labels") is not None and len(det["labels"]) > 0
        ]
        return (max(detected) + 1) if detected else 1

    def _tog_vanishing(
        self,
        x_query: np.ndarray,
        n_iter: int = 10,
        eps: float = 8 / 255.0,
        eps_iter: float = 2 / 255.0,
    ) -> np.ndarray:
        """
        Run the TOG vanishing variant to suppress existing detections via objectness gradients.

        Args:
            x_query (np.ndarray): Clean input batch in [0, 1] with shape (N, C, H, W).
                Will be copied and perturbed; dtype should be float32 or convertible.
            n_iter (int): Number of gradient update steps.
            eps (float): L-infinity perturbation budget (maximum deviation from x_query).
            eps_iter (float): Per-iteration step size for the PGD update.

        Returns:
            np.ndarray: Adversarial batch in [0, 1], same shape as x_query.
        """
        logger.info(
            "Running TOG vanishing attack with n_iter=%d, eps=%.6f, eps_iter=%.6f",
            n_iter,
            eps,
            eps_iter,
        )
        x_adv = self._initialise_x_adv(x_query, eps)
        for _ in range(n_iter):
            grad = self._object_detector.compute_object_vanishing_gradient(
                x_adv, training=False
            )
            x_adv = self._update_x_adv(grad, eps_iter, x_query, x_adv, eps)
        return x_adv

    def _tog_fabrication(
        self,
        x_query: np.ndarray,
        n_iter: int = 10,
        eps: float = 8 / 255.0,
        eps_iter: float = 2 / 255.0,
    ) -> np.ndarray:
        """
        Run the TOG fabrication variant to induce false positives via objectness gradients.

        Args:
            x_query (np.ndarray): Clean input batch in [0, 1] with shape (N, C, H, W).
                Will be copied and perturbed; dtype should be float32 or convertible.
            n_iter (int): Number of gradient update steps.
            eps (float): L-infinity perturbation budget (maximum deviation from x_query).
            eps_iter (float): Per-iteration step size for the PGD update.

        Returns:
            np.ndarray: Adversarial batch in [0, 1], same shape as x_query.
        """
        logger.info(
            "Running TOG fabrication attack with n_iter=%d, eps=%.6f, eps_iter=%.6f",
            n_iter,
            eps,
            eps_iter,
        )
        x_adv = self._initialise_x_adv(x_query, eps)
        for _ in range(n_iter):
            grad = self._object_detector.compute_object_fabrication_gradient(x_adv, training=False)
            x_adv = self._update_x_adv(grad, eps_iter, x_query, x_adv, eps)
        return x_adv

    def _tog_mislabeling(
        self,
        x_query: np.ndarray,
        mode: Literal["ml", "ll"],
        n_iter: int = 10,
        eps: float = 8 / 255.0,
        eps_iter: float = 2 / 255.0,
    ) -> np.ndarray:
        """
        Run the TOG mislabeling variant to change predicted classes using ML/LL target selection.

        Args:
            x_query (np.ndarray): Clean input batch in [0, 1] with shape (N, C, H, W).
                Will be copied and perturbed; dtype should be float32 or convertible.
            mode (Literal["ml", "ll"]): Mislabeling target selection strategy:
                "ml" (most-likely non-original) or "ll" (least-likely).
            n_iter (int): Number of gradient update steps.
            eps (float): L-infinity perturbation budget (maximum deviation from x_query).
            eps_iter (float): Per-iteration step size for the PGD update.

        Returns:
            np.ndarray: Adversarial batch in [0, 1], same shape as x_query.
        """
        logger.info(
            "Running TOG mislabeling attack with n_iter=%d, eps=%.6f, eps_iter=%.6f, mode=%s",
            n_iter,
            eps,
            eps_iter,
            str(mode),
        )
        if mode.lower() not in ["ml", "ll"]:
            warnings.warn(f"Unknown mode '{mode}'. Using 'ml' instead.")
            mode = "ml"
        x_uint8 = (x_query * 255.0).clip(0, 255).astype(np.uint8)
        x_tensor = (
            torch.from_numpy(x_uint8)
            .float()
            .to(next(self._object_detector.model.parameters()).device)
        )
        initial_detections = self._object_detector.predict(x_tensor)
        x_adv = self._initialise_x_adv(x_query, eps)
        num_classes = self._resolve_num_classes(initial_detections)
        targets = self.generate_mislabeling_targets(
            initial_detections, num_classes, mode
        )
        for i in range(n_iter):
            grad = self._object_detector.compute_object_mislabeling_gradient(
                detections=initial_detections, x=x_adv, target_labels_list=targets
            )
            grad_norm = np.linalg.norm(grad)
            if i % 50 == 0:  # Log every 50 iterations
                logger.debug("Iteration %d: Gradient norm = %.6f", i, float(grad_norm))
            if grad_norm < 1e-8:
                warnings.warn(f"Very small gradients detected, stopping early")
                break
            x_adv = self._update_x_adv(grad, eps_iter, x_query, x_adv, eps)
        return x_adv

    def _tog_untargeted(
        self,
        x_query: np.ndarray,
        n_iter: int = 10,
        eps: float = 8 / 255.0,
        eps_iter: float = 2 / 255.0,
    ) -> np.ndarray:
        """
        Run the untargeted TOG variant to reduce detector performance without specifying target classes.

        Args:
            x_query (np.ndarray): Clean input batch in [0, 1] with shape (N, C, H, W).
                Will be copied and perturbed; dtype should be float32 or convertible.
            n_iter (int): Number of gradient update steps.
            eps (float): L-infinity perturbation budget (maximum deviation from x_query).
            eps_iter (float): Per-iteration step size for the PGD update.

        Returns:
            np.ndarray: Adversarial batch in [0, 1], same shape as x_query.
        """
        logger.info(
            "Running TOG untargeted attack with n_iter=%d, eps=%.6f, eps_iter=%.6f",
            n_iter,
            eps,
            eps_iter,
        )
        x_uint8 = (x_query * 255.0).clip(0, 255).astype(np.uint8)
        detections_list = self._object_detector.predict(x_uint8)
        x_adv = self._initialise_x_adv(x_query, eps)
        for _ in range(n_iter):
            grad = self._object_detector.compute_object_untargeted_gradient(
                x_adv, detections=detections_list
            )
            x_adv = self._update_x_adv(grad, eps_iter, x_query, x_adv, eps)
        return x_adv
