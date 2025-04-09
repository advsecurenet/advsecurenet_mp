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
        self.object_detector = self._initialise_object_detector(config.object_detector)
        self.dataset = self._load_dataset("coco")
        self.patch_shape: int = config.patch_shape
        self.learning_rate: int = config.learning_rate
        self.max_iterations: int = config.max_iter
        self.batch_size: int = config.batch_size
        self.verbose: int = config.verbose

        # TODO - potentially add clip_values variant (see ART):
        self._patch = np.zeros(shape=self.patch_shape, dtype=np.float32)
        self.target_label: int | np.ndarray | list[int] | None = []

    def _load_dataset(self, dataset: string) -> None:
        dataset = load_dataset("detection-datasets/coco")
        return dataset


    def _initialise_object_detector(self, object_detector: string) -> None:
        """
        Initialises the object detector.

        Args:
            object_detector (string): The object detector's name to be used.
        """
        # Load a YOLOv5 model (options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)
        object_detector = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)  # Default: yolov5s
        #self.object_detector = object_detector #TODO: implement this method
        return object_detector


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
        x = x.detach()
        y = y.detach()
        x = self.device_manager.to_device(x)
        y = self.device_manager.to_device(y)
        mask = mask.copy() if mask is not None else None

        channel_index = x.ndim() - 1 # assuming self.estimator.channels_first is False
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

        for i_step in trange(self.max_iter, desc="DPatch iteration", disable=not self.verbose):
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
                res = self.object_detector(patched_images[i_batch_start:i_batch_end])
                pred_boxes = [torch.tensor(res.xyxy[i][:, :4].cpu().numpy(), device=patched_images.device) for i in range(len(res.xyxy))]
                pred_labels = [torch.tensor(res.xyxy[i][:, 5].cpu().numpy().astype(int), device=patched_images.device) for i in range(len(res.xyxy))]


                classification_loss = CELoss(pred_labels, target_labels)
                bbox_loss = BBoxLoss(pred_boxes, target_boxes)
                total_loss = classification_loss + bbox_loss

                if patched_images.grad is not None:
                    patched_images.grad.zero_()

                total_loss.backward()
                gradients = patched_images.grad.clone().detach()

                # gradients = self.object_detector.loss_gradient(
                #     x=patched_images[i_batch_start:i_batch_end],
                #     y=patch_target[i_batch_start:i_batch_end],
                #     standardise_output=True,
                # )

                for i_image in range(gradients.shape[0]):

                    i_x_1 = transforms[i_batch_start + i_image]["i_x_1"]
                    i_x_2 = transforms[i_batch_start + i_image]["i_x_2"]
                    i_y_1 = transforms[i_batch_start + i_image]["i_y_1"]
                    i_y_2 = transforms[i_batch_start + i_image]["i_y_2"]

                    patch_gradients_i = gradients[i_image, i_x_1:i_x_2, i_y_1:i_y_2, :]
                    patch_gradients = patch_gradients + patch_gradients_i

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
        x: np.ndarray,
        patch: np.ndarray,
        random_location: bool,
        #channels_first: bool,
        mask: np.ndarray | None = None,
        transforms: list[dict[str, int]] | None = None,
    ) -> tuple[np.ndarray, list[dict[str, int]]]:
        """
        Augment images with patch.

        :param x: Sample images.
        :param patch: The patch to be applied.
        :param random_location: If True apply patch at randomly shifted locations, otherwise place patch at origin
                                (top-left corner).
        :param channels_first: Set channels first or last.
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
        x_copy = x.copy()
        patch_copy = patch.copy()

        # if channels_first:
        #     x_copy = np.transpose(x_copy, (0, 2, 3, 1))
        #     patch_copy = np.transpose(patch_copy, (1, 2, 0))

        for i_image in range(x.shape[0]):

            if transforms is None:

                if random_location:
                    if mask is None:
                        i_x_1 = random.randint(0, x_copy.shape[1] - 1 - patch_copy.shape[0])
                        i_y_1 = random.randint(0, x_copy.shape[2] - 1 - patch_copy.shape[1])
                    else:

                        if mask.shape[0] == 1:
                            mask_2d = mask[0, :, :]
                        else:
                            mask_2d = mask[i_image, :, :]

                        edge_x_0 = patch_copy.shape[0] // 2
                        edge_x_1 = patch_copy.shape[0] - edge_x_0
                        edge_y_0 = patch_copy.shape[1] // 2
                        edge_y_1 = patch_copy.shape[1] - edge_y_0

                        mask_2d[0:edge_x_0, :] = False
                        mask_2d[-edge_x_1:, :] = False
                        mask_2d[:, 0:edge_y_0] = False
                        mask_2d[:, -edge_y_1:] = False

                        num_pos = np.argwhere(mask_2d).shape[0]
                        pos_id = np.random.choice(num_pos, size=1)
                        pos = np.argwhere(mask_2d > 0)[pos_id[0]]
                        i_x_1 = pos[0] - edge_x_0
                        i_y_1 = pos[1] - edge_y_0

                else:
                    i_x_1 = 0
                    i_y_1 = 0

                i_x_2 = i_x_1 + patch_copy.shape[0]
                i_y_2 = i_y_1 + patch_copy.shape[1]

                random_transformations.append({"i_x_1": i_x_1, "i_y_1": i_y_1, "i_x_2": i_x_2, "i_y_2": i_y_2})

            else:
                i_x_1 = transforms[i_image]["i_x_1"]
                i_x_2 = transforms[i_image]["i_x_2"]
                i_y_1 = transforms[i_image]["i_y_1"]
                i_y_2 = transforms[i_image]["i_y_2"]

            x_copy[i_image, i_x_1:i_x_2, i_y_1:i_y_2, :] = patch_copy

        # if channels_first:
        #     x_copy = np.transpose(x_copy, (0, 3, 1, 2))

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
