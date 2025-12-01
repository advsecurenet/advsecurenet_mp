"""
This code was written based on the following sources:

https://arxiv.org/pdf/1806.02299
https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/dpatch.py
"""

import math
import random
import warnings
import secrets
import numpy as np
import torch
import logging
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import trange, tqdm
from typing import Union, Optional, List, Dict, Tuple
from torch.utils.data import DataLoader

from advsecurenet.shared.types.configs.attack_configs.dpatch_attack_config import (
    DPatchAttackConfig,
)
from advsecurenet.computer_vision.base.adversarial_attack import AdversarialAttack
from advsecurenet.utils.device_utils import move_batch_to_device

logger = logging.getLogger(__name__)


class DPatch(AdversarialAttack):
    """
    DPatch attack. This attack applies an adversarial patch to the input images to deceive the object detector.
    The goal of DPatch attack is to train the adversarial patch pattern that once attached to the input image, RoIs extracted
    by the detector gather in the region where the patch is attached. It is a white-box and iterative attack.

    Args:
        _object_detector: The object detector to attack. It should be an object of a predefined class, e.g. CustomYolov5ODWrapper.
        _patch_shape (tuple): The shape of the adversarial patch.
        _learning_rate (float): The learning rate for the optimizer.
        _max_iterations (int): The maximum number of iterations for the attack.
        _target_label (int): The target label for the attack. When provided, true labels will be ignored.

    References:
        [1] Xin Liu and Huanrui Yang and Ziwei Liu and Linghao Song and Hai Li and Yiran Chen (2019). DPatch: An Adversarial Patch Attack on Object Detectors. arXiv preprint arXiv:1806.02299.

    """

    def __init__(self, config: DPatchAttackConfig) -> None:
        super().__init__(config)
        self._object_detector = config.object_detector
        self._patch_shape = config.patch_shape
        self._learning_rate = config.learning_rate
        self._max_iterations = config.max_iter
        self._target_label = config.target_label
        self.initialize_patch()

    def initialize_patch(self) -> None:
        """
        Initialize the adversarial patch tensor.

        Returns:
            None
        """
        self._patch = torch.zeros(
            (
                tuple(
                    int(x)
                    for x in self._patch_shape.replace("(", "")
                    .replace(")", "")
                    .replace(" ", "")
                    .split(",")
                )
                if isinstance(self._patch_shape, str)
                else self._patch_shape
            ),
            dtype=torch.float32,
        )

    def attack(
        self,
        dataloader: DataLoader,
        mask: Union[np.ndarray, torch.Tensor, None],
        device: Union[str, torch.device],
    ) -> torch.Tensor:
        """
        Run the DPatch optimization loop.

        Args:
            dataloader (DataLoader): PyTorch DataLoader yielding (images, targets_dict) batches.
                        images: torch.Tensor of shape (N, C, H, W) in [0, 1].
                        targets_dict: dict with keys "boxes" and "labels" as lists of tensors.
            mask (Union[np.ndarray, torch.Tensor, None]): Optional spatial mask restricting valid patch centers. Either:
                  - numpy array (H, W) or (N, H, W), or
                  - torch.Tensor (H, W) or (N, H, W), or None for unrestricted placement.
            device (Union[str, torch.device]): CUDA device string or torch.device (e.g., "cuda:0").

        Returns:
            torch.Tensor: The optimized adversarial patch as a torch.Tensor of shape (C, ph, pw) in [0, 255].
        """
        self._object_detector.model.eval()
        ignore_true_labels = self._target_label is not None
        for i_step in trange(self._max_iterations, desc="DPatch iteration"):
            self._maybe_log_training_step(i_step)
            if hasattr(dataloader, "sampler") and isinstance(
                dataloader.sampler, DistributedSampler
            ):
                try:
                    dataloader.sampler.set_epoch(i_step)
                except Exception as e:
                    logger.debug("Failed to set sampler epoch %d: %s", i_step, e)
            patch_gradients_sum, suppress_flag_any = (
                self._accumulate_gradients_over_batches(
                    device=device,
                    dataloader=dataloader,
                    ignore_true_labels=ignore_true_labels,
                    mask=mask,
                    i_step=i_step,
                )
            )
            # Distributed aggregation: sum gradients, OR suppression flag across ranks
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(patch_gradients_sum, op=dist.ReduceOp.SUM)
                suppress_tensor = torch.tensor(
                    1 if suppress_flag_any else 0, device=patch_gradients_sum.device
                )
                dist.all_reduce(suppress_tensor, op=dist.ReduceOp.MAX)
                suppress_flag_any = bool(suppress_tensor.item())
            self._apply_patch_update(patch_gradients_sum, suppress_flag_any)
            self._check_patch_consistency()
        return self._patch

    def _maybe_log_training_step(self, i_step: int) -> None:
        if i_step == 0 or (i_step + 1) % 100 == 0:
            logger.info("Training Step: %d/%d", i_step + 1, self._max_iterations)

    def _accumulate_gradients_over_batches(
        self,
        device: Union[str, torch.device],
        dataloader: DataLoader,
        ignore_true_labels: bool,
        mask: Union[np.ndarray, torch.Tensor, None],
        i_step: int,
    ) -> tuple[torch.Tensor, bool]:
        patch_gradients_sum = torch.zeros_like(self._patch, device=device)
        suppress_flag_any = False
        self._patch = self.device_manager.to_device(self._patch)
        for data_batch in tqdm(
            dataloader,
            desc=f"Epoch {i_step + 1}/{self._max_iterations}",
            leave=False,
        ):
            images, targets_dict = data_batch
            images, targets_dict = move_batch_to_device(images, targets_dict, device)
            images_np_for_dpatch = (images.detach().cpu().numpy() * 255.0).astype(
                np.float32
            )
            images_np_for_dpatch = np.clip(images_np_for_dpatch, 0, 255)
            targets = None
            if not ignore_true_labels:
                targets = []
                for b, l in zip(targets_dict["boxes"], targets_dict["labels"]):
                    raw = l.detach().cpu().numpy().astype(int)
                    targets.append(
                        {
                            "boxes": b.detach().cpu().numpy(),
                            "labels": np.array(raw, dtype=int),
                            "scores": np.ones(len(raw), dtype=float),
                        }
                    )
            patch_gradients, untargeted_should_suppress = self._attack_step(
                x=images_np_for_dpatch,
                y=targets,
                mask=mask,
                device=device,
            )
            patch_gradients = self.device_manager.to_device(patch_gradients)
            patch_gradients_sum += patch_gradients
            if untargeted_should_suppress:
                suppress_flag_any = True
        return patch_gradients_sum, suppress_flag_any

    def _apply_patch_update(
        self, patch_gradients_sum: torch.Tensor, suppress_flag_any: bool
    ) -> None:
        if self._target_label is not None:
            self._patch = self._patch - self._learning_rate * torch.sign(
                patch_gradients_sum
            )
        else:
            if suppress_flag_any:
                self._patch = self._patch - self._learning_rate * torch.sign(
                    patch_gradients_sum
                )
            else:
                self._patch = self._patch + self._learning_rate * torch.sign(
                    patch_gradients_sum
                )
        self._patch = self._patch.clamp(0.0, 255.0)

    def _check_patch_consistency(self) -> None:
        if dist.is_available() and dist.is_initialized():
            with torch.no_grad():
                checksum = torch.sum(self._patch.float()).unsqueeze(0)
                gathered = [
                    torch.zeros_like(checksum) for _ in range(dist.get_world_size())
                ]
                try:
                    dist.all_gather(gathered, checksum)
                    diffs = [abs(checksum.item() - g.item()) for g in gathered]
                    max_diff = max(diffs) if diffs else 0.0
                    if max_diff > 1e-4:
                        logger.error(
                            "[DPATCH]Patch checksum divergence detected across ranks: diffs=%s",
                            diffs,
                        )
                except Exception as e:
                    logger.debug("Patch consistency check failed: %s", e)

    def _attack_step_prepare_x(
        self,
        x: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
    ) -> torch.Tensor:
        """
        Normalize input batch to a torch.Tensor on the configured device.

        Args:
            x (Union[torch.Tensor, np.ndarray, List[np.ndarray]]): Batch as tensor,
                NumPy array (N, C, H, W), or list of NumPy arrays to be stacked.

        Returns:
            torch.Tensor: Batch tensor on the device, shape (N, C, H, W).
        """
        x = DPatch.if_ndarrray_convert_to_tensor(x)
        if isinstance(x, list) and isinstance(x[0], np.ndarray):
            x = torch.tensor(np.stack(x))
        x = self.device_manager.to_device(x)
        return x

    def _attack_step_initial_aug(
        self,
        x: torch.Tensor,
        mask: Optional[Union[np.ndarray, torch.Tensor]],
    ) -> Tuple[
        Optional[Union[np.ndarray, torch.Tensor]], torch.Tensor, List[Dict[str, int]]
    ]:
        """
        Create initial patched images and transforms using a cloned patch.

        Args:
            x (torch.Tensor): Input batch tensor with shape (N, C, H, W).
            mask (Optional[Union[np.ndarray, torch.Tensor]]): Optional spatial mask
                restricting valid patch centers.

        Returns:
            Tuple[Optional[Union[np.ndarray, torch.Tensor]], torch.Tensor, List[Dict[str, int]]]:
                - mask copy (same type as input),
                - patched_images_initial tensor (N, C, H, W),
                - transforms_initial list with keys "i_x_1", "i_x_2", "i_y_1", "i_y_2".
        """
        mask = mask.copy() if mask is not None else None
        initial_patch_for_target_determination = self._patch.clone()
        patched_images_initial, transforms_initial = self.augment_images_with_patch(
            x,
            initial_patch_for_target_determination,
            random_location=False,
            mask=mask,
            transforms=None,
        )
        return mask, patched_images_initial, transforms_initial

    def _attack_step_build_patch_target_and_flag(
        self,
        patched_images: torch.Tensor,
        transforms: List[Dict[str, int]],
        y: Optional[List[Dict[str, np.ndarray]]],
    ) -> Tuple[List[Dict[str, np.ndarray]], bool]:
        """
        Build patch targets and compute suppression flag for untargeted mode.

        Args:
            patched_images (torch.Tensor): Current patched batch, shape (N, C, H, W).
            transforms (List[Dict[str, int]]): Per-image placement info with keys
                "i_x_1", "i_x_2", "i_y_1", "i_y_2".
            y (Optional[List[Dict[str, np.ndarray]]]): Optional per-image ground-truth
                targets with keys "boxes", "labels", and "scores".

        Returns:
            Tuple[List[Dict[str, np.ndarray]], bool]:
                - patch_target: Per-image target dictionaries.
                - _untargeted_attack_should_suppress_from_empty_initial: Suppression flag.
        """
        patch_target = self._prepare_patch_targets(
            patched_images=patched_images, transforms=transforms, y=y
        )
        _untargeted_attack_should_suppress_from_empty_initial = False
        if not self._target_label and y is None:
            all_initial_targets_empty = True
            for pt_entry in patch_target:
                if pt_entry.get("labels") is not None and len(pt_entry["labels"]) > 0:
                    all_initial_targets_empty = False
                    break
            if all_initial_targets_empty:
                _untargeted_attack_should_suppress_from_empty_initial = True
        return patch_target, _untargeted_attack_should_suppress_from_empty_initial

    def _attack_step_accumulate_patch_gradients(
        self,
        gradients: np.ndarray,
        transforms: List[Dict[str, int]],
        i_batch_start: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Accumulate per-image gradients into a single patch-shaped tensor.

        Args:
            gradients (np.ndarray): Loss gradients for the batch slice, shape (B, C, H, W).
            transforms (List[Dict[str, int]]): Per-image placement info with keys
                "i_x_1", "i_x_2", "i_y_1", "i_y_2".
            i_batch_start (int): Start index of the current batch slice.
            device (torch.device): Device for the returned gradient tensor.

        Returns:
            torch.Tensor: Aggregated gradient tensor matching patch shape (C, ph, pw).
        """
        patch_gradients = torch.zeros_like(self._patch, device=device)
        for i_image in range(gradients.shape[0]):
            i_x_1 = transforms[i_batch_start + i_image]["i_x_1"]
            i_x_2 = transforms[i_batch_start + i_image]["i_x_2"]
            i_y_1 = transforms[i_batch_start + i_image]["i_y_1"]
            i_y_2 = transforms[i_batch_start + i_image]["i_y_2"]
            patch_gradients_i = gradients[
                i_image,
                :,
                i_x_1:i_x_2,
                i_y_1:i_y_2,
            ]
            patch_gradients_i = self.device_manager.to_device(
                torch.from_numpy(patch_gradients_i)
            )
            patch_gradients += patch_gradients_i
        patch_gradients = self.device_manager.to_device(patch_gradients)
        return patch_gradients

    def _attack_step(
        self,
        x: Union[torch.Tensor, np.ndarray],  # (batch_size, channels, height, width)
        y: torch.tensor,  # (batch_size, num_boxes, 4) (x1, y1, x2, y2)
        mask: torch.tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, bool]:
        """
        Performs a single gradient update step for the patch based on a single batch of data.

        Args:
            x (Union[torch.Tensor, np.ndarray]): Batch of images with shape (N, C, H, W).
                If np.ndarray, values are expected in [0, 255] float32; if torch.Tensor,
                they will be moved to the configured device.
            y (torch.tensor): Target specification for the batch. For targeted runs, pass
                None. For untargeted runs, pass a list of dicts per image with keys
                "boxes" (np.ndarray), "labels" (np.ndarray), and "scores" (np.ndarray).
            mask (torch.tensor): Optional spatial mask controlling valid patch centers.
                May also be a NumPy array; forwarded unchanged to augmentation helpers.
            device (torch.device): Device context used for intermediate tensors.

        Returns:
            Tuple[torch.Tensor, bool]: A tuple containing:
                - patch_gradients: Aggregated gradient tensor matching patch shape (C, ph, pw).
                - untargeted_should_suppress: Flag indicating the update direction for untargeted mode.

        Raises:
            RuntimeError: If augmentation, target building, prediction, or gradient computation fails.
        """
        x = self._attack_step_prepare_x(x)
        _untargeted_attack_should_suppress_from_empty_initial = False
        patch_gradients = torch.zeros_like(self._patch, device=device)
        try:
            mask, patched_images_initial, transforms_initial = (
                self._attack_step_initial_aug(x, mask)
            )
        except Exception as e:
            logger.exception("Initial augmentation failed in _attack_step.")
        transforms = transforms_initial.copy()
        patched_images = patched_images_initial.clone().detach().requires_grad_(True)
        patched_images = self.device_manager.to_device(patched_images)
        try:
            patch_target, _untargeted_attack_should_suppress_from_empty_initial = (
                self._attack_step_build_patch_target_and_flag(
                    patched_images, transforms, y
                )
            )
            current_step_patched_images, _ = self.augment_images_with_patch(
                x,
                self._patch,
                random_location=False,
                mask=mask,
                transforms=transforms,
            )
        except Exception as e:
            logger.exception(
                "Current-step patch building and/or augmentation failed in _attack_step."
            )
        actual_batch_size = x.shape[0]
        i_batch_start = 0
        i_batch_end = min(actual_batch_size, patched_images.shape[0])
        try:
            gradients = self._prepare_data_and_compute_patch_gradients(
                patched_images,
                current_step_patched_images,
                i_batch_start,
                i_batch_end,
                patch_target,
            )
            patch_gradients = self._attack_step_accumulate_patch_gradients(
                gradients, transforms, i_batch_start, device
            )
        except Exception as e:
            logger.exception("Patch gradients computation failed in _attack_step.")
        return patch_gradients, _untargeted_attack_should_suppress_from_empty_initial

    def apply_patch(
        self,
        x: np.ndarray,
        patch_external: np.ndarray | None = None,
        random_location: bool = False,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply the adversarial patch to images.

        Args:
            x (np.ndarray): Images to be patched with shape (N, C, H, W).
            patch_external (np.ndarray | None): External patch to apply with shape (C, ph, pw).
                If None, the internally optimized patch is used.
            random_location (bool): If True, place the patch at a random valid location per image.
                If False, place it at the origin (top-left).
            mask (np.ndarray | None): Optional boolean mask of shape (H, W) or (N, H, W).
                True entries indicate valid patch centers when random_location is True.

        Returns:
            np.ndarray: The patched images with shape (N, C, H, W).
        """
        if patch_external is not None:
            patch_local = patch_external
        else:
            patch_local = self._patch
        patched_images, _ = self.augment_images_with_patch(
            x=x,
            patch=patch_local,
            random_location=random_location,
            mask=mask,
        )
        return patched_images

    # === Attack Step helpers ===

    def _prepare_patch_targets(
        self,
        patched_images: torch.Tensor,
        transforms: List[Dict[str, int]],
        y: Optional[List[Dict[str, np.ndarray]]],
    ) -> List[Dict[str, np.ndarray]]:
        """
        Build per-image target dicts for the current patch placement.

        Args:
            patched_images (torch.Tensor): Current batch of patched images, shape (N, C, H, W).
            transforms (List[Dict[str, int]]): Per-image placement info with keys
                "i_x_1", "i_x_2", "i_y_1", "i_y_2".
            y (Optional[List[Dict[str, np.ndarray]]]): Optional per-image ground-truth targets
                with keys "boxes", "labels", and "scores". If None, targets are inferred.

        Returns:
            List[Dict[str, np.ndarray]]: Per-image target dictionaries with keys
            "boxes", "labels", and "scores".
        """
        patch_target: list[dict[str, np.ndarray]] = []
        if (self._target_label is not None) and (y is None):
            logger.info(
                "[DPATCH] targeted attack - target_label: %s", str(self._target_label)
            )
            for i_image in range(patched_images.shape[0]):
                if isinstance(self._target_label, int):
                    t_l = self._target_label
                else:
                    t_l = self._target_label[i_image]
                i_x_1 = transforms[i_image]["i_x_1"]
                i_x_2 = transforms[i_image]["i_x_2"]
                i_y_1 = transforms[i_image]["i_y_1"]
                i_y_2 = transforms[i_image]["i_y_2"]
                target_dict = {}
                target_dict["boxes"] = np.asarray([[i_x_1, i_y_1, i_x_2, i_y_2]])
                mapped = np.asarray([t_l], dtype=int)
                target_dict["labels"] = mapped
                target_dict["scores"] = np.asarray(
                    [
                        1.0,
                    ]
                )
                patch_target.append(target_dict)
        elif self._target_label is None:
            predictions = None
            if y is not None:
                logger.info("[DPATCH] untargeted attack - true labels provided")
                predictions = y
            else:
                logger.info("[DPATCH] untargeted attack - no true labels provided")
                targets = []
                patched_images_np = patched_images.detach().cpu().numpy()
                res = self._object_detector.predict(patched_images_np)
                res = [self._object_detector.filter_boxes(t, 0.8) for t in res]
                for preds_dict in res:
                    boxes = preds_dict["boxes"]
                    labels = preds_dict["labels"]
                    scores = preds_dict["scores"]
                    target = {
                        "boxes": boxes,
                        "labels": labels,
                        "scores": scores,
                    }
                    targets.append(target)
                predictions = targets
            for i_image in range(patched_images.shape[0]):
                target_dict = {}
                target_dict["boxes"] = predictions[i_image]["boxes"]
                target_dict["labels"] = predictions[i_image]["labels"]
                target_dict["scores"] = predictions[i_image]["scores"]
                patch_target.append(target_dict)
        return patch_target

    def _prepare_data_and_compute_patch_gradients(
        self,
        patched_images: torch.Tensor,
        current_step_patched_images: torch.Tensor,
        i_batch_start: int,
        i_batch_end: int,
        patch_target: List[Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """
        Prepare inputs and compute loss gradients for the current patch placement.

        Args:
            patched_images (torch.Tensor): Tensor on the correct device used for device
                context and shape checks; shape (N, C, H, W).
            current_step_patched_images (torch.Tensor): Patched images for this step;
                shape (N, C, H, W).
            i_batch_start (int): Start index (inclusive) of the slice to process.
            i_batch_end (int): End index (exclusive) of the slice to process.
            patch_target (List[Dict[str, np.ndarray]]): Per-image target dicts with keys
                "boxes", "labels", and "scores".

        Returns:
            np.ndarray: Loss gradients for the batch slice, shape (slice_N, C, H, W).

        Raises:
            ValueError: If out-of-range labels are detected in patch_target.
        """
        img_batch = (
            current_step_patched_images[i_batch_start:i_batch_end]
            .detach()
            .cpu()
            .numpy()
        )
        input_batch_np = img_batch.astype(np.float32)
        res = self._object_detector.predict(input_batch_np)
        pred_boxes = []
        pred_labels = []
        for i in range(len(res)):
            if "boxes" in res[i] and res[i]["boxes"].size > 0:
                pred_boxes.append(
                    torch.tensor(
                        res[i]["boxes"],
                        device=patched_images.device,
                        dtype=torch.float32,
                    )
                )
                pred_labels.append(
                    torch.tensor(
                        res[i]["labels"],
                        device=patched_images.device,
                        dtype=torch.float32,
                    )
                )
            else:
                pred_boxes.append(
                    torch.empty(
                        (0, 4), device=patched_images.device, dtype=torch.float32
                    )
                )
                pred_labels.append(
                    torch.empty((0,), device=patched_images.device, dtype=torch.float32)
                )
        # Extract predicted class logits or confidence scores
        pred_logits = []
        for result in res:
            if "scores" in result and "labels" in result:
                scores = result["scores"]
                labels = result["labels"].astype(int)
                num_classes = 80
                logits = torch.zeros(
                    (len(labels), num_classes), device=patched_images.device
                )
                for idx, label in enumerate(labels):
                    logits[idx, label] = float(scores[idx])
                pred_logits.append(logits)
        all_labels = np.concatenate([t["labels"] for t in patch_target])
        invalid_mask = (all_labels < 0) | (all_labels >= num_classes)
        if invalid_mask.any():
            bad = all_labels[invalid_mask]
            logger.warning(
                "Invalid labels detected: %s unique: %s", bad, np.unique(bad)
            )
            raise ValueError("Found out-of-range labels in patch_target; see above.")
        gradients = self._object_detector.loss_gradient(
            x=input_batch_np,
            y=patch_target[i_batch_start:i_batch_end],
            standardise_output=True,
        )
        return gradients

    # === Apply Patch To Images static method & its helpers ===

    @staticmethod
    def augment_images_with_patch_transforms_provided(
        transforms: List[Dict[str, int]],
        i_image: int,
        img_width: int,
        img_height: int,
        patch_width: int,
        patch_height: int,
        x_copy: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        """
        Validate and return placement coordinates from provided transforms.

        Args:
            transforms (List[Dict[str, int]]): List of dicts with keys: "i_x_1", "i_x_2", "i_y_1", "i_y_2".
            i_image (int): Index of the image in the current batch.
            img_width (int): Image width (W).
            img_height (int): Image height (H).
            patch_width (int): Patch width.
            patch_height (int): Patch height.
            x_copy (torch.Tensor): Tensor copy of the batch (N, C, H, W), used for error context.

        Returns:
            Tuple[int, int, int, int]: (i_x_1, i_x_2, i_y_1, i_y_2) defining the slice [i_x_1:i_x_2, i_y_1:i_y_2].

        Raises:
            ValueError: If keys are missing, bounds are invalid, or dimensions mismatch.
        """
        required = {"i_x_1", "i_x_2", "i_y_1", "i_y_2"}
        if i_image >= len(transforms):
            raise ValueError(
                f"Missing transform for image {i_image} (len={len(transforms)})."
            )
        if not required.issubset(transforms[i_image].keys()):
            raise ValueError(
                f"Transform for image {i_image} missing keys {required - set(transforms[i_image].keys())}."
            )
        i_x_1 = transforms[i_image]["i_x_1"]
        i_x_2 = transforms[i_image]["i_x_2"]
        i_y_1 = transforms[i_image]["i_y_1"]
        i_y_2 = transforms[i_image]["i_y_2"]
        if not (0 <= i_x_1 < i_x_2 <= img_height and 0 <= i_y_1 < i_y_2 <= img_width):
            raise ValueError(
                f"Invalid transform coordinates for image {i_image}: {transforms[i_image]} with image shape {x_copy.shape}"
            )
        if (i_x_2 - i_x_1) != patch_height or (i_y_2 - i_y_1) != patch_width:
            raise ValueError(
                f"Transform dimensions ({i_x_2 - i_x_1}, {i_y_2 - i_y_1}) do not match patch dimensions ({patch_height}, {patch_width}) for image {i_image}"
            )
        return i_x_1, i_x_2, i_y_1, i_y_2

    @staticmethod
    def augment_images_with_patch_random_location_no_mask(
        img_width: int,
        img_height: int,
        patch_width: int,
        patch_height: int,
    ) -> Tuple[int, int]:
        """
        Sample a random valid top-left corner without using a mask.

        Args:
            img_width (int): Image width (W).
            img_height (int): Image height (H).
            patch_width (int): Patch width.
            patch_height (int): Patch height.

        Returns:
            Tuple[int, int]: (i_x_1, i_y_1) top-left corner for the patch.

        Raises:
            ValueError: If any dimension is non-positive or the patch is larger than the image.
        """
        if min(img_width, img_height, patch_width, patch_height) <= 0:
            raise ValueError(
                f"All dimensions must be positive, got (W={img_width}, H={img_height}, pW={patch_width}, pH={patch_height})."
            )
        # Calculate random top-left corner for height (i_x_1) and width (i_y_1)
        if img_height < patch_height or img_width < patch_width:
            raise ValueError("Patch dimensions are larger than image dimensions.")
        max_h_start = img_height - patch_height
        max_w_start = img_width - patch_width
        i_x_1 = secrets.randbelow(max_h_start + 1)
        i_y_1 = secrets.randbelow(max_w_start + 1)
        return i_x_1, i_y_1

    @staticmethod
    def augment_images_with_patch_random_location_with_mask(
        mask: Union[np.ndarray, torch.Tensor],
        img_width: int,
        img_height: int,
        patch_width: int,
        patch_height: int,
        i_image: int,
    ) -> Tuple[int, int]:
        """
        Sample a random valid top-left corner constrained by a boolean center mask.

        The mask indicates valid centers; this converts centers to a top-left corner
        ensuring the full patch remains inside the image.

        Args:
            mask (Union[np.ndarray, torch.Tensor]): Boolean mask (H, W) or (N, H, W). True marks valid centers.
            img_width (int): Image width (W).
            img_height (int): Image height (H).
            patch_width (int): Patch width.
            patch_height (int): Patch height.
            i_image (int): Batch index (used if mask has batch dimension).

        Returns:
            Tuple[int, int] (i_x_1, i_y_1): Top-left corner for the patch.

        Raises:
            ValueError: If mask shape is invalid or no valid locations exist.
            RuntimeError: If computed coordinates are out of bounds (sanity check).
        """
        # Assuming mask is (H, W) or (N, H, W)
        if mask.ndim == 3:
            if mask.shape[0] == 1:
                mask_2d = mask[0, :, :]
            else:
                mask_2d = mask[i_image, :, :]
        elif mask.ndim == 2:
            mask_2d = mask
        else:
            raise ValueError(f"Unexpected mask dimension: {mask.ndim}")
        if isinstance(mask_2d, torch.Tensor):
            mask_2d = mask_2d.cpu().numpy()
        mask_2d = mask_2d.astype(bool)
        if mask_2d.shape[0] != img_height or mask_2d.shape[1] != img_width:
            raise ValueError(
                f"Mask shape {mask_2d.shape} does not match image spatial dimensions ({img_height}, {img_width})"
            )
        # Calculate patch center offsets
        edge_x_0 = patch_height // 2
        edge_x_1 = patch_height - edge_x_0
        edge_y_0 = patch_width // 2
        edge_y_1 = patch_width - edge_y_0
        valid_center_mask = np.zeros_like(mask_2d, dtype=bool)
        valid_center_mask[
            edge_x_0 : img_height - edge_x_1 + 1, edge_y_0 : img_width - edge_y_1 + 1
        ] = True
        final_mask = mask_2d & valid_center_mask
        valid_indices = np.argwhere(final_mask)
        if valid_indices.shape[0] == 0:
            raise ValueError(
                "No valid locations found in the mask to place the patch center such that the patch remains within image bounds."
            )
        else:
            pos_id = np.random.choice(valid_indices.shape[0], size=1)
            center_x, center_y = valid_indices[pos_id[0]]
            # Calculate top-left corner based on center
            i_x_1 = center_x - edge_x_0
            i_y_1 = center_y - edge_y_0
            if not (
                0 <= i_x_1 <= img_height - patch_height
                and 0 <= i_y_1 <= img_width - patch_width
            ):
                raise RuntimeError(
                    f"Internal error: Calculated invalid patch start ({i_x_1}, {i_y_1}) from center ({center_x}, {center_y})"
                )
            return i_x_1, i_y_1

    @staticmethod
    def augment_images_with_patch_no_transforms(
        random_location: bool,
        mask: Optional[Union[np.ndarray, torch.Tensor]],
        img_width: int,
        img_height: int,
        patch_width: int,
        patch_height: int,
        i_image: int,
    ) -> Tuple[int, int, int, int]:
        """
        Compute placement coordinates either at origin or randomly (with/without mask).

        Args:
            random_location: If True, sample a random valid location.
            mask: Optional mask for valid centers; used only when random_location is True.
            img_width: Image width (W).
            img_height: Image height (H).
            patch_width: Patch width.
            patch_height: Patch height.
            i_image: Batch index (used if mask has batch dimension).

        Returns:
            Tuple (i_x_1, i_x_2, i_y_1, i_y_2): Slice coordinates (top-left and bottom-right, H and W).

        Raises:
            ValueError: If patch is larger than image when placed at origin.
        """
        if random_location:
            if mask is None:
                i_x_1, i_y_1 = DPatch.augment_images_with_patch_random_location_no_mask(
                    img_width, img_height, patch_width, patch_height
                )
            else:
                i_x_1, i_y_1 = (
                    DPatch.augment_images_with_patch_random_location_with_mask(
                        mask, img_width, img_height, patch_width, patch_height, i_image
                    )
                )
        else:
            i_x_1 = 0
            i_y_1 = 0
            if patch_height > img_height or patch_width > img_width:
                raise ValueError(
                    f"Patch (H={patch_height}, W={patch_width}) is larger than image (H={img_height}, W={img_width}) and cannot be placed at origin."
                )
        # Calculate bottom-right corner (exclusive index for slicing)
        i_x_2 = i_x_1 + patch_height
        i_y_2 = i_y_1 + patch_width
        return i_x_1, i_x_2, i_y_1, i_y_2

    @staticmethod
    def if_ndarrray_convert_to_tensor(
        x: Union[torch.Tensor, np.ndarray, List[np.ndarray]],
    ) -> Union[torch.Tensor, np.ndarray, List[np.ndarray]]:
        """
        Convert a NumPy array to a torch.Tensor; return other types unchanged.

        Args:
            x (Union[torch.Tensor, np.ndarray, List[np.ndarray]]): Input that may be a
                tensor, NumPy array, or list of NumPy arrays.

        Returns:
            Union[torch.Tensor, np.ndarray, List[np.ndarray]]: A torch.Tensor when input
            was a NumPy array; otherwise the original input unchanged.
        """
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x

    @staticmethod
    def prepare_tensors_and_shapes(
        x: torch.Tensor | np.ndarray, patch: torch.Tensor | np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, int, int, int, int]:
        """
        Convert inputs to torch tensors, clone them, and return basic shapes.

        Args:
            x (torch.Tensor | np.ndarray): Batch of images as torch.Tensor (N, C, H, W) or np.ndarray with same layout.
            patch (torch.Tensor | np.ndarray): Patch tensor as torch.Tensor (C, H, W) or np.ndarray.

        Returns:
            Tuple containing:
                - x_copy (torch.Tensor): Cloned tensor of x, shape (N, C, H, W).
                - patch_copy (torch.Tensor): Cloned tensor of patch, shape (C, H, W).
                - img_height (int)
                - img_width (int)
                - patch_height (int)
                - patch_width (int)

        Raises:
            ValueError: If channel counts of image and patch do not match.
        """
        x_t = DPatch.if_ndarrray_convert_to_tensor(x)
        patch_t = DPatch.if_ndarrray_convert_to_tensor(patch)
        x_copy = x_t.clone()
        patch_copy = patch_t.clone()
        # Assuming NCHW format for image and CHW for patch
        img_channels, img_height, img_width = (
            x_copy.shape[1],
            x_copy.shape[2],
            x_copy.shape[3],
        )
        patch_channels, patch_height, patch_width = (
            patch_copy.shape[0],
            patch_copy.shape[1],
            patch_copy.shape[2],
        )
        if img_channels != patch_channels:
            raise ValueError(
                f"Image channels ({img_channels}) and patch channels ({patch_channels}) must match."
            )
        return (
            x_copy,
            patch_copy,
            img_height,
            img_width,
            patch_height,
            patch_width,
        )

    @staticmethod
    def place_patch_into_image(
        x_copy: torch.Tensor,
        patch_copy: torch.Tensor,
        i_image: int,
        i_x_1: int,
        i_x_2: int,
        i_y_1: int,
        i_y_2: int,
    ) -> None:
        """
        Write the patch into x_copy at the given slice for a single image, with checks.

        Args:
            x_copy (torch.Tensor): Tensor of images (N, C, H, W).
            patch_copy (torch.Tensor): Patch tensor (C, H, W).
            i_image (int): Image index within the batch.
            i_x_1 (int): Top H index (inclusive).
            i_x_2 (int): Bottom H index (exclusive).
            i_y_1 (int): Left W index (inclusive).
            i_y_2 (int): Right W index (exclusive).

        Returns:
            None

        Raises:
            RuntimeError: If the target slice shape does not match the patch shape.
        """
        try:
            target_slice = x_copy[i_image, :, i_x_1:i_x_2, i_y_1:i_y_2]
            if target_slice.shape != patch_copy.shape:
                raise RuntimeError(
                    f"Shape mismatch before assignment: Slice shape {target_slice.shape}, "
                    f"Patch shape {patch_copy.shape}"
                )
            x_copy[i_image, :, i_x_1:i_x_2, i_y_1:i_y_2] = patch_copy.contiguous()
        except Exception:
            logger.exception(
                "Error during patch application for image %d: "
                "x_copy shape=%s, patch_copy shape=%s, "
                "Indices: H=%d:%d, W=%d:%d, "
                "Calculated slice shape=(%d, %d, %d).",
                i_image,
                tuple(x_copy.shape),
                tuple(patch_copy.shape),
                i_x_1,
                i_x_2,
                i_y_1,
                i_y_2,
                x_copy.shape[1],
                (i_x_2 - i_x_1),
                (i_y_2 - i_y_1),
            )

    @staticmethod
    def augment_images_with_patch(
        x: torch.Tensor,
        patch: torch.Tensor,
        random_location: bool,
        mask: np.ndarray | None = None,
        transforms: list[dict[str, int]] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, int]]]:
        """
        Augment images with an adversarial patch.

        Args:
            x (torch.Tensor): Batch of images with shape (N, C, H, W).
            patch (torch.Tensor): Patch tensor with shape (C, H, W).
            random_location (bool): If True, place the patch at a random valid location; if False, place it at the origin (top-left).
            mask (np.ndarray | None): Optional boolean mask of shape (H, W) or (N, H, W). True entries indicate valid patch centers for random placement.
            transforms (list[dict[str, int]] | None): Optional explicit placement transforms. Requires random_location=False and mask=None.

        Returns:
            tuple[torch.Tensor, list[dict[str, int]]]: A tuple (patched_images, transforms) where patched_images has shape (N, C, H, W) and transforms contains per-image placement metadata.

        """
        if transforms is not None:
            if random_location or mask is not None:
                raise ValueError(
                    "Definition of patch locations in `locations` requires `random_location=False`, and `mask=None`."
                )
        (
            x_copy,
            patch_copy,
            img_height,
            img_width,
            patch_height,
            patch_width,
        ) = DPatch.prepare_tensors_and_shapes(x, patch)
        random_transformations = []
        for i_image in range(x_copy.shape[0]):
            if transforms is None:
                i_x_1, i_x_2, i_y_1, i_y_2 = (
                    DPatch.augment_images_with_patch_no_transforms(
                        random_location,
                        mask,
                        img_width,
                        img_height,
                        patch_width,
                        patch_height,
                        i_image,
                    )
                )
                random_transformations.append(
                    {"i_x_1": i_x_1, "i_y_1": i_y_1, "i_x_2": i_x_2, "i_y_2": i_y_2}
                )
            else:
                i_x_1, i_x_2, i_y_1, i_y_2 = (
                    DPatch.augment_images_with_patch_transforms_provided(
                        transforms,
                        i_image,
                        img_width,
                        img_height,
                        patch_width,
                        patch_height,
                        x_copy,
                    )
                )
            DPatch.place_patch_into_image(
                x_copy,
                patch_copy,
                i_image,
                i_x_1,
                i_x_2,
                i_y_1,
                i_y_2,
            )
        return x_copy, random_transformations
