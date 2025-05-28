"""
This code was written based on the following sources:

https://arxiv.org/pdf/1806.02299
https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/dpatch.py
"""

import math
import random
import numpy as np
import torch
from tqdm.auto import trange

from advsecurenet.shared.types.configs.attack_configs.dpatch_attack_config import DPatchAttackConfig
from advsecurenet.computer_vision.object_detection.attacks.base.object_detection_attack import ObjectDetectionAttack


class DPatch(ObjectDetectionAttack):
    """
    DPatch attack

    Paper: https://arxiv.org/abs/1712.04248
    """

    def __init__(self, config: DPatchAttackConfig) -> None:
        super().__init__(config)
        self.object_detector = config.object_detector
        self.patch_shape = config.patch_shape
        self.learning_rate = config.learning_rate
        self.max_iterations = config.max_iter
        self.batch_size = config.batch_size
        self.verbose = config.verbose
        self._patch = torch.zeros(self.patch_shape, dtype=torch.float32, device=config.device.processor)


    def attack(
            self,
            x: torch.tensor,  # (batch_size, channels, height, width)
            y: torch.tensor,  # (batch_size, num_boxes, 4) (x1, y1, x2, y2)
            target_label: torch.tensor,
            mask: torch.tensor,
            *args,
            **kwargs
    ) -> torch.tensor:
        """
        Generates adversarial examples using the DPatch attack.
        Args:
            model (BaseModel): The model to attack.
            x (torch.tensor): The original input tensor. Expected shape is (batch_size, channels, height, width).
            y (torch.tensor): The true labels for the input tensor. Expected shape is (batch_size, num_boxes, 4) (x1, y1, x2, y2).

        Returns:
            torch.tensor: The adversarial example tensor.
        """
        if target_label is not None and y is not None:
            raise ValueError("Both target_label and y cannot be provided at the same time.")
        self.object_detector.model.eval()
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        elif isinstance(x, list) and isinstance(x[0], np.ndarray):
            x = torch.tensor(np.stack(x))  # Convert list of numpy arrays to tensor 
        x = self.device_manager.to_device(x)
        mask = mask.copy() if mask is not None else None
        initial_patch_for_target_determination = self._patch.clone()
        patched_images_initial, transforms_initial = self._augment_images_with_patch(
                    x,
                    initial_patch_for_target_determination,
                    random_location=False,
                    mask=mask,
                    transforms=None,
        )       
        transforms = transforms_initial.copy()  # Copy the transforms for later use
        patched_images = patched_images_initial.clone().detach().requires_grad_(True)
        patch_target: list[dict[str, np.ndarray]] = []
        if False:
            return
        else:
            if (target_label is not None) and (y is None): # targetted attack
                print(f"[DPATCH] targetted attack - target_label: {target_label}")
                for i_image in range(patched_images.shape[0]):
                    if isinstance(target_label, int):
                        t_l = target_label
                    else:
                        t_l = target_label[i_image]
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
            elif target_label is None: # untargetted attack
                predictions = None
                if y is not None:
                    print(f"[DPATCH] untargetted attack - true labels provided")
                    predictions = y
                else: # untargetted attack where no true labels are provided
                    print(f"[DPATCH] untargetted attack - no true labels provided")
                    targets = []
                    patched_images_np = patched_images.detach().cpu().numpy()
                    res = self.object_detector.predict(patched_images_np)
                    res = [self.object_detector.filter_boxes(t, 0.8) for t in res]
                    for preds_dict in res:  # Iterate over each image in the batch
                        boxes = preds_dict["boxes"]  # Bounding boxes (x1, y1, x2, y2)
                        labels = preds_dict["labels"]  # Class labels
                        scores = preds_dict["scores"]  # Confidence scores   
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
        # Flag to determine if the untargeted attack started with no detections
        # and should therefore aim to suppress any new detections.
        _untargeted_attack_should_suppress_from_empty_initial = False
        if not target_label and y is None:
            # This condition means patch_target was derived from self.object_detector(patched_images)
            # where patched_images were based on the initial_patch_for_target_determination.
            all_initial_targets_empty = True
            for pt_entry in patch_target:
                # Check if the 'labels' key exists and has entries
                if pt_entry.get("labels") is not None and len(pt_entry["labels"]) > 0:
                    all_initial_targets_empty = False
                    break
            if all_initial_targets_empty:
                _untargeted_attack_should_suppress_from_empty_initial = True
                if self.verbose:
                    print("[DPATCH] Untargeted attack mode: Initial state (image + initial patch) had no detections. The attack will aim to keep it that way (suppress new detections).")     
        #CELoss = nn.CrossEntropyLoss()
        #BBoxLoss = nn.BCEWithLogitsLoss()
        for i_step in trange(self.max_iterations, desc="DPatch iteration", disable=not self.verbose):
            if i_step == 0 or (i_step + 1) % 100 == 0:
                print("Training Step: %i", i_step + 1)
            # Generate patched images for the current optimization step
            # using the original images 'x' and the current state of 'self._patch'.
            #current_step_patched_images, current_step_transforms = self._augment_images_with_patch(
            current_step_patched_images, _ = self._augment_images_with_patch(
                x,                       # Original clean images
                self._patch,             # Current adversarial patch
                random_location=False,   # Assuming fixed location based on initial transforms
                mask=mask,               # Original mask
                transforms=transforms,   # Transforms derived from transforms_initial
            )
            num_batches = math.ceil(x.shape[0] / self.batch_size)
            patch_gradients = torch.zeros_like(self._patch)
            for i_batch in range(num_batches):
                i_batch_start = i_batch * self.batch_size
                i_batch_end = min((i_batch + 1) * self.batch_size, patched_images.shape[0])
                # Use patch_target for targets
                #y_list = patch_target[i_batch_start:i_batch_end]
                #target_boxes = [torch.tensor(target["boxes"], device=patched_images.device) for target in y_list]
                #target_labels = [torch.tensor(target["labels"], device=patched_images.device) for target in y_list]
                img_batch = current_step_patched_images[i_batch_start:i_batch_end].detach().cpu().numpy()
                input_batch_np = img_batch.astype(np.float32)
                res = self.object_detector.predict(input_batch_np)
                pred_boxes = []
                pred_labels = []
                for i in range(len(res)):
                    # Check if the dictionary for the current image is not empty
                    if "boxes" in res[i] and res[i]["boxes"].size > 0:
                        pred_boxes.append(torch.tensor(res[i]["boxes"], device=patched_images.device, dtype=torch.float32))
                        pred_labels.append(torch.tensor(res[i]["labels"], device=patched_images.device, dtype=torch.float32))
                    else:
                        # Handle cases with no detections if necessary, e.g., append empty tensors
                        # This depends on how the loss function handles empty predictions
                        pred_boxes.append(torch.empty((0, 4), device=patched_images.device, dtype=torch.float32))
                        pred_labels.append(torch.empty((0,), device=patched_images.device, dtype=torch.float32))
                # Extract predicted class logits or confidence scores (not just class indices)
                pred_logits = []
                for result in res:
                    if "scores" in result and "labels" in result:
                        scores = result["scores"]  # confidence scores
                        labels = result["labels"].astype(int)
                        num_classes = 80 # model.num_classes  # You must know this from your model definition
                        logits = torch.zeros((len(labels), num_classes), device=patched_images.device)
                        for idx, label in enumerate(labels):
                            logits[idx, label] = float(scores[idx])
                        pred_logits.append(logits)
                # Concatenate predictions into a single tensor
                #pred_logits_cat = torch.cat(pred_logits, dim=0)
                # Ensure targets are the correct shape
                #target_labels_cat = torch.cat(target_labels, dim=0).long()
                #pred_labels_cat = torch.cat(pred_labels, dim=0)
                #target_labels_cat = torch.cat(target_labels, dim=0)
                #if pred_boxes and target_boxes:
                    #pred_boxes_cat = torch.cat(pred_boxes, dim=0)
                    #target_boxes_cat = torch.cat(target_boxes, dim=0)
                    #target_boxes_cat = target_boxes_cat.float()
                all_labels = np.concatenate([t["labels"] for t in patch_target])
                invalid_mask = (all_labels < 0) | (all_labels >= num_classes)
                if invalid_mask.any():
                    bad = all_labels[invalid_mask]
                    print("⚠️ Invalid labels detected:", bad, "unique:", np.unique(bad))
                    raise ValueError("Found out-of-range labels in patch_target; see above.")
                gradients = self.object_detector.loss_gradient(
                    x=input_batch_np,
                    y=patch_target[i_batch_start:i_batch_end],
                    standardise_output=True,
                )
                for i_image in range(gradients.shape[0]):
                    i_x_1 = transforms[i_batch_start + i_image]["i_x_1"]
                    i_x_2 = transforms[i_batch_start + i_image]["i_x_2"]
                    i_y_1 = transforms[i_batch_start + i_image]["i_y_1"]
                    i_y_2 = transforms[i_batch_start + i_image]["i_y_2"]
                    patch_gradients_i = gradients[
                        i_image,           # batch index
                        :,                 # channels
                        i_x_1:i_x_2,       # height slice
                        i_y_1:i_y_2        # width slice
                    ]
                    patch_gradients += patch_gradients_i  # now both are (C, patch_h, patch_w)
            patch_gradients /= x.shape[0]
            if target_label is not None:
                self._patch = self._patch - self.learning_rate * torch.sign(patch_gradients)
            else:
                if _untargeted_attack_should_suppress_from_empty_initial:
                    # In this specific case, patch_target was "no objects".
                    # The loss_gradient indicates how to *increase* detections (increase loss against "no objects").
                    # To suppress detections, we need to *minimize* this loss, so perform gradient descent.
                    self._patch = self._patch - self.learning_rate * torch.sign(patch_gradients)
                else:
                    # Standard untargeted attack: patch_target had some objects (from y_true or non-empty initial predictions).
                    # Maximize loss to move away from these targets (gradient ascent).
                    self._patch = self._patch + self.learning_rate * torch.sign(patch_gradients)
            self._patch = self._patch.clamp(0.0, 255.0)
            patched_images, transforms_initial = self._augment_images_with_patch(
                x,
                self._patch,
                random_location=False,
                mask=None,
                transforms=None,
            )
        return self._patch
    

    @staticmethod
    def _augment_images_with_patch(
        x: torch.Tensor,  # Changed type hint to torch.Tensor
        patch: torch.Tensor, # Changed type hint to torch.Tensor
        random_location: bool,
        mask: np.ndarray | None = None,
        transforms: list[dict[str, int]] | None = None,
    ) -> tuple[torch.Tensor, list[dict[str, int]]]: # Changed return type hint
        """
        Augment images with patch.

        :param x: Sample images (N, C, H, W).
        :param patch: The patch to be applied (C, H, W).
        :param random_location: If True apply patch at randomly shifted locations, otherwise place patch at origin
                                (top-left corner).
        :param mask: A boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :param transforms: Patch transforms, requires `random_location=False`, and `mask=None`.
        :type mask: `np.ndarray`
        """
        if transforms is not None:
            if random_location or mask is not None:
                raise ValueError(
                    "Definition of patch locations in `locations` requires `random_location=False`, and `mask=None`."
                )
        random_transformations = []
        # Ensure inputs are tensors
        if isinstance(x, np.ndarray):
             x = torch.from_numpy(x)
        if isinstance(patch, np.ndarray):
             patch = torch.from_numpy(patch)
        x_copy = x.clone()
        patch_copy = patch.clone()
        # Assuming NCHW format for image and CHW for patch
        img_channels, img_height, img_width = x_copy.shape[1], x_copy.shape[2], x_copy.shape[3]
        patch_channels, patch_height, patch_width = patch_copy.shape[0], patch_copy.shape[1], patch_copy.shape[2]
        if img_channels != patch_channels:
            raise ValueError(f"Image channels ({img_channels}) and patch channels ({patch_channels}) must match.")
        for i_image in range(x_copy.shape[0]):
            if transforms is None:
                if random_location:
                    if mask is None:
                        # Calculate random top-left corner for height (i_x_1) and width (i_y_1)
                        if img_height < patch_height or img_width < patch_width:
                             raise ValueError("Patch dimensions are larger than image dimensions.")
                        # Ensure upper bound is not less than lower bound
                        max_h_start = img_height - patch_height
                        max_w_start = img_width - patch_width
                        if max_h_start < 0 or max_w_start < 0:
                             raise ValueError(f"Patch (H={patch_height}, W={patch_width}) is larger than image (H={img_height}, W={img_width}).")
                        i_x_1 = random.randint(0, max_h_start) # Use height
                        i_y_1 = random.randint(0, max_w_start)   # Use width
                    else:
                        # Mask logic needs adjustment for channels-first if used
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
                        # Ensure mask is boolean numpy array
                        if isinstance(mask_2d, torch.Tensor):
                            mask_2d = mask_2d.cpu().numpy()
                        mask_2d = mask_2d.astype(bool)
                        if mask_2d.shape[0] != img_height or mask_2d.shape[1] != img_width:
                            raise ValueError(f"Mask shape {mask_2d.shape} does not match image spatial dimensions ({img_height}, {img_width})")
                        # Calculate patch center offsets
                        edge_x_0 = patch_height // 2
                        edge_x_1 = patch_height - edge_x_0
                        edge_y_0 = patch_width // 2
                        edge_y_1 = patch_width - edge_y_0
                        # Create a valid mask for patch *center* placement
                        # A center at (cx, cy) is valid if the patch fits entirely within the image
                        # Top-left corner: (cx - edge_x_0, cy - edge_y_0)
                        # Bottom-right corner: (cx + edge_x_1 - 1, cy + edge_y_1 - 1)
                        # Conditions:
                        # cx - edge_x_0 >= 0  => cx >= edge_x_0
                        # cy - edge_y_0 >= 0  => cy >= edge_y_0
                        # cx + edge_x_1 - 1 < img_height => cx < img_height - edge_x_1 + 1
                        # cy + edge_y_1 - 1 < img_width  => cy < img_width - edge_y_1 + 1
                        valid_center_mask = np.zeros_like(mask_2d, dtype=bool)
                        valid_center_mask[edge_x_0 : img_height - edge_x_1 + 1, edge_y_0 : img_width - edge_y_1 + 1] = True
                        # Combine with the user-provided mask
                        final_mask = mask_2d & valid_center_mask
                        # Find valid center positions
                        valid_indices = np.argwhere(final_mask)
                        if valid_indices.shape[0] == 0:
                            # Fallback or error if no valid location in mask
                            # Option 1: Raise error
                            raise ValueError("No valid locations found in the mask to place the patch center such that the patch remains within image bounds.")
                            # Option 2: Place randomly without mask (like the mask=None case)
                            # max_h_start = img_height - patch_height
                            # max_w_start = img_width - patch_width
                            # if max_h_start < 0 or max_w_start < 0:
                            #      raise ValueError(f"Patch (H={patch_height}, W={patch_width}) is larger than image (H={img_height}, W={img_width}).")
                            # i_x_1 = random.randint(0, max_h_start)
                            # i_y_1 = random.randint(0, max_w_start)
                        else:
                            # Choose a random valid center position
                            pos_id = np.random.choice(valid_indices.shape[0], size=1)
                            center_x, center_y = valid_indices[pos_id[0]]
                            # Calculate top-left corner based on center
                            i_x_1 = center_x - edge_x_0
                            i_y_1 = center_y - edge_y_0
                            # Ensure calculated top-left is valid (should be guaranteed by valid_center_mask, but good for sanity check)
                            if not (0 <= i_x_1 <= img_height - patch_height and 0 <= i_y_1 <= img_width - patch_width):
                                raise RuntimeError(f"Internal error: Calculated invalid patch start ({i_x_1}, {i_y_1}) from center ({center_x}, {center_y})")
                else: # Not random location
                    i_x_1 = 0
                    i_y_1 = 0
                    if patch_height > img_height or patch_width > img_width:
                         raise ValueError(f"Patch (H={patch_height}, W={patch_width}) is larger than image (H={img_height}, W={img_width}) and cannot be placed at origin.")
                # Calculate bottom-right corner (exclusive index for slicing)
                i_x_2 = i_x_1 + patch_height
                i_y_2 = i_y_1 + patch_width
                random_transformations.append({"i_x_1": i_x_1, "i_y_1": i_y_1, "i_x_2": i_x_2, "i_y_2": i_y_2})
            else: # Use provided transforms
                i_x_1 = transforms[i_image]["i_x_1"]
                i_x_2 = transforms[i_image]["i_x_2"]
                i_y_1 = transforms[i_image]["i_y_1"]
                i_y_2 = transforms[i_image]["i_y_2"]
                # Basic validation
                if not (0 <= i_x_1 < i_x_2 <= img_height and 0 <= i_y_1 < i_y_2 <= img_width):
                     raise ValueError(f"Invalid transform coordinates for image {i_image}: {transforms[i_image]} with image shape {x_copy.shape}")
                if (i_x_2 - i_x_1) != patch_height or (i_y_2 - i_y_1) != patch_width:
                     raise ValueError(f"Transform dimensions ({i_x_2 - i_x_1}, {i_y_2 - i_y_1}) do not match patch dimensions ({patch_height}, {patch_width}) for image {i_image}")
            # Apply patch using channels-first indexing (N, C, H, W)
            try:
                target_slice = x_copy[i_image, :, i_x_1:i_x_2, i_y_1:i_y_2]
                if target_slice.shape != patch_copy.shape:
                     raise RuntimeError(f"Shape mismatch before assignment: Slice shape {target_slice.shape}, Patch shape {patch_copy.shape}")
                x_copy[i_image, :, i_x_1:i_x_2, i_y_1:i_y_2] = patch_copy.contiguous()
            except Exception as e:
                 print(f"Error during patch application for image {i_image}:")
                 print(f"  x_copy shape: {x_copy.shape}")
                 print(f"  patch_copy shape: {patch_copy.shape}")
                 print(f"  Indices: H={i_x_1}:{i_x_2}, W={i_y_1}:{i_y_2}")
                 print(f"  Calculated slice shape: ({x_copy.shape[1]}, {i_x_2 - i_x_1}, {i_y_2 - i_y_1})")
                 raise e
        return x_copy, random_transformations


    def apply_patch(
        self,
        x: np.ndarray,
        patch_external: np.ndarray | None = None,
        random_location: bool = False,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Apply the adversarial patch to images.

        :param x: Images to be patched.
        :param patch_external: External patch to apply to images `x`. If None the attacks patch will be applied.
        :param random_location: True if patch location should be random.
        :param mask: A boolean array of shape equal to the shape of a single samples (1, H, W) or the shape of `x`
                     (N, H, W) without their channel dimensions. Any features for which the mask is True can be the
                     center location of the patch during sampling.
        :return: The patched images.
        """
        if patch_external is not None:
            patch_local = patch_external
        else:
            patch_local = self._patch
        patched_images, _ = self._augment_images_with_patch(
            x=x,
            patch=patch_local,
            random_location=random_location,
            mask=mask,
        )
        return patched_images
