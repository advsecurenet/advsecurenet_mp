from typing import Any, Dict, List
import torch
from abc import ABC, abstractmethod
import numpy as np


class CustomODBaseModel(torch.nn.Module, ABC):
    """
    Abstract contract for object-detection adapters used in your system.

    Conventions
    -----------
    - Images (`x`, `images`, `imgs`) are tensors with shape [B, C, H, W] (channels-first).
      Values may be in [0, 1] or [0, 255]; methods that need normalization should handle it.
    - Targets are per-image dictionaries with:
        'boxes': torch.float32 tensor of shape [N, 4], XYXY pixel coords
        'labels': torch.int64  tensor of shape [N]
      A batch is a `list[dict]` with length == B.
    - “Standardized predictions” (used by `predict_per_batch`) are Python dicts with numpy arrays:
        'boxes':  float32 ndarray [N, 4] (XYXY pixels)
        'scores': float32 ndarray [N]
        'labels': int64  ndarray [N]
      Optional extras:
        'logits': float32 ndarray [N, C] (per-class logits)
        'label_names': object/str ndarray [N]
    """

    @abstractmethod
    def forward(self, x, targets=None):
        """
        Run a model forward pass.

        Args:
            x:
                - torch.Tensor of shape [B, C, H, W] (recommended), or
                - list[torch.Tensor(C, H, W)] for backends that expect per-image tensors.
                Dtype: float32 preferred. Values may be in [0,1] or [0,255].
            targets (optional):
                list[dict] with keys 'boxes' (float32 [N,4] XYXY pixels) and
                'labels' (int64 [N]). Length == batch size. Only used in training.

        Returns:
            If `self.training` is True and `targets` is not None:
                dict[str, torch.Tensor] containing per-component losses, and
                MUST include key 'loss_total' (scalar tensor on the model's device).
            Otherwise (inference / eval):
                *Model-native* predictions (backend-specific):
                  - Torchvision Faster R-CNN: list of dicts with torch tensors
                    {'boxes','scores','labels'} per image.
                  - YOLOv5: AutoShape-style output object with attributes like `.xyxy`.
                (Standardized numpy predictions are produced by `predict_per_batch`.)
        """
        raise NotImplementedError("Subclasses must implement forward method.")

    @abstractmethod
    def predict(self, x, training):
        """
        Convenience prediction wrapper.

        Args:
            x: torch.Tensor [B, C, H, W] (float32). Values in [0,1] or [0,255].
            training (bool):
                If False: should set eval mode and perform inference (no gradients).
                If True: may keep current mode; output format must be the same as
                `forward(x, targets=None)` for consistency.

        Returns:
            Backend-native predictions (same structure as `forward(x, targets=None)`).
            (Use `predict_per_batch` when you need standardized numpy dicts.)
        """
        raise NotImplementedError("Subclasses must implement predict method.")

    @abstractmethod
    def predict_raw(self, x):
        """
        Run a raw forward to obtain pre-post-processing / pre-NMS outputs.

        Args:
            x: torch.Tensor [B, C, H, W] (float32). On the model's device.

        Returns:
            Backend-specific raw outputs (e.g., YOLO head logits / feature maps).
            If the backend does not expose raw logits, returning the same structure
            as `forward(x, targets=None)` is acceptable.
        """
        raise NotImplementedError("Subclasses must implement predict_raw method.")

    @abstractmethod
    def predict_per_batch(self, imgs, inference_model, clip_values):
        """
        Run batched inference and return standardized numpy predictions.

        Args:
            imgs: torch.Tensor [B, C, H, W] (float32). Values may be in [0,1] or [0,255].
                  Implementations may clamp/convert as needed.
            inference_model: object returned by `initialize_inference_model(...)`
                             (e.g., an AutoShape wrapper or eval()'d module).
            clip_values: tuple(min_val, max_val) or similar; upper bound may be used
                         when converting tensors to uint8 images for certain backends.

        Returns:
            list[dict] (length == B). Each dict has:
                'boxes':  np.float32 array [N, 4] (XYXY pixels)
                'scores': np.float32 array [N]
                'labels': np.int64  array [N]
            Optional keys:
                'logits':      np.float32 array [N, C]
                'label_names': np.ndarray [N] of strings
            N may be 0 for images with no detections.
        """
        raise NotImplementedError("Subclasses must implement predict_per_batch method.")

    @abstractmethod
    def initialize_inference_model(self, model, device, conf_thresh=0.7):
        """
        Prepare a model/wrapper for inference.

        Args:
            model: the underlying detection model/module to wrap.
            device: device spec (e.g., 'cpu', 'cuda', torch.device, or 'cuda:0').
            conf_thresh (float): desired confidence threshold for backends that support it.

        Returns:
            An inference-ready object to be passed to `predict_per_batch(...)`.
            Typical steps: move to device, set .eval(), optionally wrap in an
            AutoShape/transformer, and set threshold attributes if available.
        """
        raise NotImplementedError(
            "Subclasses must implement initialize_inference_model method."
        )
    
    @abstractmethod
    def load_model_weights(self, model_weights_path):
        """
        Load model weights from the specified path.

        Args:
            model_weights_path: str, path to the model weights file.
        """
        raise NotImplementedError("Subclasses must implement load_model_weights method.")

    @abstractmethod
    def prepare_training_inputs(self, images: torch.Tensor, targets: list[dict]):
        """
        Adapt caller-provided tensors/targets into backend-specific training inputs.

        Args:
            images: torch.Tensor [B, C, H, W] (float32). May be in [0,1] or [0,255].
                    Should be moved to the model's device and `requires_grad_(True)`
                    if gradients through inputs are needed.
            targets: list of dicts (length == B), each:
                     {'boxes': torch.float32 [N,4] XYXY pixels,
                      'labels': torch.int64  [N]}

        Returns:
            Tuple (images_like_model_expects, targets_like_model_expects):
              - Faster R-CNN example:
                    images_list: list[torch.Tensor(C,H,W)]
                    tv_targets:  list[{'boxes': Tensor[N,4], 'labels': Tensor[N]}]
              - YOLOv5 example:
                    images_bchw: torch.Tensor [B, C, H, W] (float32, on device)
                    yolo_targets: torch.Tensor [M, 6] with rows
                                   [img_idx, cls, xc, yc, w, h] normalized to [0,1]
        """
        raise NotImplementedError(
            "Subclasses must implement prepare_training_inputs method."
        )

    @abstractmethod
    def calculate_loss(self, predictions, target_val):
        """
        Compute a scalar loss used by your outer logic (e.g., adversarial/score-based).

        Args:
            predictions: backend-native predictions (often the output of `predict_raw`
                         or per-image dicts with 'scores' / 'logits').
            target_val: numeric control signal (commonly 0 or 1). Implementations may map
                        it as needed (e.g., {0->-1, 1->+1} for subtract/add objectives).

        Returns:
            torch.Tensor scalar on the model's device representing the loss to optimize.
        """
        raise NotImplementedError("Subclasses must implement calculate_loss method.")

    @abstractmethod
    def preprocess_x_for_loss_calculation(self, x, requires_grad=True):
        """
        Prepare inputs for loss calculation (device/dtype/range/grad).

        Args:
            x: torch.Tensor [B, C, H, W] or np.ndarray [B, C, H, W].
               Values may be in [0,1] or [0,255].
            requires_grad (bool): if True, returned tensor should have requires_grad=True.

        Returns:
            torch.Tensor [B, C, H, W], dtype=torch.float32, on the model's device.
            Values typically normalized to [0,1]. `requires_grad` set per the flag.
        """
        raise NotImplementedError(
            "Subclasses must implement preprocess_x_for_loss_calculation method."
        )

    @abstractmethod
    def translate_labels(
        self, labels: list[dict[str, torch.Tensor | np.ndarray]], batch_size: int
    ):
        """
        Convert user-supplied labels into the backend's training format.

        Args:
            labels: list of dicts (len may differ from batch size). Each dict with:
                    'boxes': torch.Tensor or np.ndarray of shape [N,4] (XYXY pixels),
                    'labels': torch.Tensor or np.ndarray of shape [N] (class ids).
            batch_size: int, number of images in the associated batch.

        Returns:
            Backend-specific labels:
              - Faster R-CNN: list[{'boxes': torch.float32 [N,4],
                                    'labels': torch.int64  [N]}] (length == batch_size).
                Must ensure exactly one target dict per image (pad with empty targets or
                truncate as needed).
              - YOLOv5: torch.float32 tensor [M, 6] where rows are
                        [img_idx, cls, xc, yc, w, h] normalized to [0,1].
        """
        raise NotImplementedError("Subclasses must implement translate_labels method.")
    
    @abstractmethod
    def translate_predictions_for_map_evaluator(self, predictions, dataset_name: str, expects_numpy: bool = True) -> List[Dict[str, Any]]:
        """
        Convert backend-native predictions into a standardized format for mAP evaluation.

        Args:
            predictions: backend-native predictions (often the output of `predict`).
            dataset_name: str, name of the dataset (e.g., 'coco') to determine class mapping.

        Returns:
            Standardized predictions for mAP evaluation.
        """
        raise NotImplementedError("Subclasses must implement translate_predictions_for_map_evaluator method.")