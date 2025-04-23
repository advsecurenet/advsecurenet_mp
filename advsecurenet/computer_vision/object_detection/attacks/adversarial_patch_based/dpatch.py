"""
This code was written based on the following sources:

https://arxiv.org/pdf/1806.02299
https://github.com/Trusted-AI/adversarial-robustness-toolbox/blob/main/art/attacks/evasion/dpatch.py
"""

import datasets
from datasets import load_dataset
import math
import random
import numpy as np
from PIL import Image
import typing
import string
from collections import deque

import click
import torch
from torch import nn
from tqdm.auto import trange

from advsecurenet.shared.types.configs.attack_configs.dpatch_attack_config import DPatchAttackConfig
from advsecurenet.computer_vision.object_detection.attacks.base.object_detection_attack import ObjectDetectionAttack
from advsecurenet.models.base_model import BaseModel


class DPatch(ObjectDetectionAttack):
    """
    DPatch attack

    Paper: https://arxiv.org/abs/1712.04248
    """

    def __init__(self, config: DPatchAttackConfig) -> None:
        super().__init__(config)
        self.object_detector = config.object_detector#self._initialise_object_detector(config.object_detector)
        #self.dataset = self._load_dataset("coco")
        self.patch_shape = config.patch_shape
        self.learning_rate = config.learning_rate
        self.max_iterations = config.max_iter
        self.batch_size = config.batch_size
        self.verbose = config.verbose

        # TODO - potentially add clip_values variant (see ART):
        _patch_np = np.zeros(shape=self.patch_shape, dtype=np.float32)
        self._patch = torch.from_numpy(_patch_np).to(config.device.processor)
        #self._patch = np.zeros(shape=self.patch_shape, dtype=np.float32)
        self.target_label: int | np.ndarray | list[int] | None = []


    # def _initialise_object_detector(self, object_detector: string) -> None:
    #     """
    #     Initialises the object detector.

    #     Args:
    #         object_detector (string): The object detector's name to be used.
    #     """
    #     # Load a YOLOv5 model (options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
    #     object_detector = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)  # Default: yolov5s
    #     #self.object_detector = object_detector #TODO: implement this method
    #     return object_detector


    def attack(
            self,
            model: BaseModel,
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
        model.eval()

        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
        elif isinstance(x, list) and isinstance(x[0], np.ndarray):
            x = torch.tensor(np.stack(x))  # Convert list of numpy arrays to tensor

        # if isinstance(y, np.ndarray):
        #     y = torch.from_numpy(y)
        # elif isinstance(y, list) and y and isinstance(y[0], np.ndarray): # Check if list is not empty
        #     y = torch.from_numpy(np.stack(y))  # Convert list of numpy arrays to tensor
        # elif isinstance(y, list):# and y and isinstance(y[0], torch.Tensor): # Check if list is not empty
        #     y = torch.stack(y) # Convert list of tensors to tensor
           
        x = self.device_manager.to_device(x)
        print(x)
        print(type(x))
        #y = self.device_manager.to_device(y)
        mask = mask.copy() if mask is not None else None

        channel_index = x.ndim - 1 # assuming self.estimator.channels_first is False
        patched_images, transforms = self._augment_images_with_patch(
                    x,
                    self._patch,
                    random_location=True,
                    #channels_first=False,
                    mask=mask,
                    transforms=None,
        )       
        patched_images = patched_images.clone().detach().requires_grad_(True)
        patch_target: list[dict[str, np.ndarray]] = []

        if self.target_label and y is None:

            for i_image in range(patched_images.shape[0]):
                if isinstance(self.target_label, int):
                    t_l = self.target_label
                else:
                    t_l = self.target_label[i_image]

                i_x_1 = transforms[i_image]["i_x_1"]
                i_x_2 = transforms[i_image]["i_x_2"]
                i_y_1 = transforms[i_image]["i_y_1"]
                i_y_2 = transforms[i_image]["i_y_2"]

                target_dict = {}
                target_dict["boxes"] = np.asarray([[i_x_1, i_y_1, i_x_2, i_y_2]])
                target_dict["labels"] = np.asarray(
                    [
                        t_l,
                    ]
                )
                target_dict["scores"] = np.asarray(
                    [
                        1.0,
                    ]
                )

                patch_target.append(target_dict)

        else: 
            if y is not None:
                predictions = y
            else: # if untargetted attack + no true labels are provided
                targets = []
                res = self.object_detector(patched_images)
                for i in range(len(res.xyxy)):  # Iterate over each image in the batch
                    boxes = res.xyxy[i][:, :4].cpu().numpy()  # Bounding boxes (x1, y1, x2, y2)
                    labels = res.xyxy[i][:, 5].cpu().numpy().astype(int)  # Class labels
                    scores = res.xyxy[i][:, 4].cpu().numpy()  # Confidence scores
                    
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

        CELoss = nn.CrossEntropyLoss()
        BBoxLoss = nn.BCEWithLogitsLoss()

        #TODO - replace 1000 with an actual self.max_iterations value
        for i_step in trange(1000, desc="DPatch iteration", disable=not self.verbose):
            if i_step == 0 or (i_step + 1) % 100 == 0:
                print("Training Step: %i", i_step + 1)

            num_batches = math.ceil(x.shape[0] / self.batch_size)
            patch_gradients = np.zeros_like(self._patch)

            for i_batch in range(num_batches):
                i_batch_start = i_batch * self.batch_size
                i_batch_end = min((i_batch + 1) * self.batch_size, patched_images.shape[0])

                # targets_list = []
                # target_boxes = [target["boxes"] for target in targets_list]  # List of bounding boxes for each image
                # target_labels = [target["labels"] for target in targets_list]  # List of labels for each image
                
                # y_list=patch_target[i_batch_start:i_batch_end]
                # pred_boxes = [y["boxes"] for y in y_list]
                # pred_labels = [y["labels"] for y in y_list]

                # Use patch_target for targets
                y_list = patch_target[i_batch_start:i_batch_end]
                target_boxes = [torch.tensor(target["boxes"], device=patched_images.device) for target in y_list]
                target_labels = [torch.tensor(target["labels"], device=patched_images.device) for target in y_list]

                # Predictions from the patched images
                #res = self.object_detector.predict(patched_images[i_batch_start:i_batch_end])
                input_batch_np = patched_images[i_batch_start:i_batch_end].detach().cpu().numpy()
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

                #pred_boxes = [torch.tensor(res.xyxy[i][:, :4].cpu().numpy(), device=patched_images.device) for i in range(len(res.xyxy))]
                #pred_labels = [torch.tensor(res.xyxy[i][:, 5].cpu().numpy().astype(int), device=patched_images.device) for i in range(len(res.xyxy))]

                # Extract predicted class logits or confidence scores (not just class indices)
                pred_logits = []
                for result in res:
                    if "scores" in result and "labels" in result:
                        scores = result["scores"]  # confidence scores
                        labels = result["labels"].astype(int)

                        num_classes = 80 # model.num_classes  # You must know this from your model definition
                        logits = torch.zeros((len(labels), num_classes), device=patched_images.device)

                        for idx, label in enumerate(labels):
                            #logits[idx, label] = scores[idx]  # Assign scores to corresponding class index
                            logits[idx, label] = float(scores[idx])

                        pred_logits.append(logits)

                # Concatenate predictions into a single tensor
                pred_logits_cat = torch.cat(pred_logits, dim=0)

                # Ensure targets are the correct shape
                target_labels_cat = torch.cat(target_labels, dim=0).long()

                pred_labels_cat = torch.cat(pred_labels, dim=0)
                target_labels_cat = torch.cat(target_labels, dim=0)
                #target_labels_cat = target_labels_cat.long()
                #classification_loss = CELoss(pred_logits_cat, target_labels_cat)

                if pred_boxes and target_boxes:
                    pred_boxes_cat = torch.cat(pred_boxes, dim=0)
                    target_boxes_cat = torch.cat(target_boxes, dim=0)
                    target_boxes_cat = target_boxes_cat.float()
                #bbox_loss = BBoxLoss(pred_boxes_cat, target_boxes_cat)

                #total_loss = classification_loss + bbox_loss

                # if patched_images.grad is not None:
                #     patched_images.grad.zero_()

                # total_loss.backward()
                # gradients = patched_images.grad.clone().detach()
                input_batch_np = patched_images[i_batch_start:i_batch_end].detach().cpu().numpy()
                gradients = self.object_detector.loss_gradient(
                    x=input_batch_np,#patched_images[i_batch_start:i_batch_end],
                    y=patch_target[i_batch_start:i_batch_end],
                    standardise_output=True,
                )

                for i_image in range(gradients.shape[0]):

                    i_x_1 = transforms[i_batch_start + i_image]["i_x_1"]
                    i_x_2 = transforms[i_batch_start + i_image]["i_x_2"]
                    i_y_1 = transforms[i_batch_start + i_image]["i_y_1"]
                    i_y_2 = transforms[i_batch_start + i_image]["i_y_2"]

                    # patch_gradients_i = gradients[i_image, i_x_1:i_x_2, i_y_1:i_y_2, :]
                    # patch_gradients = patch_gradients + patch_gradients_i
                    patch_gradients_i = gradients[
                        i_image,           # batch index
                        :,                 # channels
                        i_x_1:i_x_2,       # height slice
                        i_y_1:i_y_2        # width slice
                    ]
                    patch_gradients += patch_gradients_i  # now both are (C, patch_h, patch_w)

            if self.target_label:
                self._patch = self._patch - np.sign(patch_gradients) * self.learning_rate
            else:
                self._patch = self._patch + np.sign(patch_gradients) * self.learning_rate

            #TODO - add value clipping if necessary (see ART)

            patched_images, _ = self._augment_images_with_patch(
                x,
                self._patch,
                random_location=False,
                #channels_first=False,#self.estimator.channels_first,
                mask=None,
                transforms=transforms,
            )

        return np.clip(self._patch, 0, 1)
    

    @staticmethod
    def _augment_images_with_patch(
        x: torch.Tensor,  # Changed type hint to torch.Tensor
        patch: torch.Tensor, # Changed type hint to torch.Tensor
        random_location: bool,
        #channels_first: bool, # Assuming channels_first (N, C, H, W)
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

        # Ensure tensors are on the same device if possible (optional, depends on overall design)
        # if x.device != patch.device:
        #    patch = patch.to(x.device) # Or handle device mismatch appropriately

        x_copy = x.clone()
        patch_copy = patch.clone()

        # Assuming NCHW format for image and CHW for patch
        img_channels, img_height, img_width = x_copy.shape[1], x_copy.shape[2], x_copy.shape[3]
        patch_channels, patch_height, patch_width = patch_copy.shape[0], patch_copy.shape[1], patch_copy.shape[2]
        # print(f"DEBUG: Image shape: {x_copy.shape}")
        # print(f"DEBUG: Patch shape: {patch_copy.shape}")

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
                # print(f"DEBUG: Applying patch to x_copy[{i_image}, :, {i_x_1}:{i_x_2}, {i_y_1}:{i_y_2}]")
                # print(f"DEBUG: Slice shape should be: ({img_channels}, {patch_height}, {patch_width})")
                # print(f"DEBUG: Patch shape is: {patch_copy.shape}")
                target_slice = x_copy[i_image, :, i_x_1:i_x_2, i_y_1:i_y_2]
                # print(f"DEBUG: Actual target_slice shape: {target_slice.shape}")
                if target_slice.shape != patch_copy.shape:
                     raise RuntimeError(f"Shape mismatch before assignment: Slice shape {target_slice.shape}, Patch shape {patch_copy.shape}")

                # Ensure patch is contiguous, might help with weird stride issues
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
            #channels_first=self.estimator.channels_first,
            mask=mask,
        )

        return patched_images
