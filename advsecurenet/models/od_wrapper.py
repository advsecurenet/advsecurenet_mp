import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn


class ODWrapper():
    def __init__(
        self,
        model,
        input_shape,
        device_type,
        clip_values,
        attack_losses,
        conf_thresh=0.7,
        weight_dict=None,
    ):
        self.model = model
        self.conf_thresh = conf_thresh
        self.device = device_type
        self.model.device = device_type
        self.clip_values = clip_values
        self.input_shape = input_shape
        self.inference_model = self.model.initialize_inference_model(model, device_type, conf_thresh)
        setattr(self.inference_model, "expects_numpy_images", bool(getattr(self.model, "expects_numpy_images", False)))
        self.attack_losses = attack_losses
        self.weight_dict = weight_dict
        self.channels_first = True
        if hasattr(model, 'num_classes'):
            self.num_classes = self.model.num_classes
        else:
            # Default to COCO classes if not specified
            self.num_classes = 91
        self.model.input_shape = input_shape
        self.model.channels_first = True
        if self._should_freeze_bn():
            self._freeze_bn()

    def filter_boxes(self, predictions, conf_thresh):
        # tolerate missing keys, create consistent empty arrays
        boxes  = predictions.get("boxes")
        scores = predictions.get("scores")
        labels = predictions.get("labels")
        names  = predictions.get("label_names", None)
        if boxes is None:
            boxes = np.empty((0, 4), dtype=np.float32)
        if scores is None:
            scores = np.empty((0,), dtype=np.float32)
        if labels is None:
            labels = np.empty((0,), dtype=np.int64)
        mask = scores >= conf_thresh
        out = {
            "boxes":  boxes[mask] if boxes.size else boxes,    # -> (0,4) when empty
            "scores": scores[mask] if scores.size else scores, # -> (0,)
            "labels": labels[mask] if labels.size else labels, # -> (0,)
        }
        if names is not None:
            out["label_names"] = names[mask] if len(names) else np.empty((0,), dtype=names.dtype)
        return out
    
    def prepare_training_inputs(self, images: torch.Tensor, targets: list[dict]):
        return self.model.prepare_training_inputs(images, targets)

    def compute_object_vanishing_gradient(
        self, x: np.ndarray, training: bool = False
    ) -> np.ndarray:
        x = torch.from_numpy(x)
        x_pre = x.to(self.device, dtype=torch.float32)
        x_pre.requires_grad_(True)
        if training:
            self.model.train()  # ensure we get raw preds, not autoshaped outputs
            if self._should_freeze_bn():
                self._freeze_bn()
        preds = self.model.predict(x_pre, training=training)
        loss = self.model.calculate_loss(preds, target_val=0.0)
        self.model.zero_grad()
        grad_tensor = torch.autograd.grad(
            outputs=loss,
            inputs=x_pre,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        grads = grad_tensor.cpu().numpy()
        # 5) undo any scaling
        if self.clip_values is not None:
            grads = grads / self.clip_values[1]
        return grads

    def compute_object_untargeted_gradient(
        self,
        x: np.ndarray,
        detections: list[dict[str, np.ndarray]] = None,
        training: bool = True,
    ) -> np.ndarray:
        if not detections:
            return np.zeros_like(x)
        x_torch = torch.from_numpy(x).to(self.device, dtype=torch.float32)
        x_torch.requires_grad_(True)
        if training:
            self.model.train()  # Set model to training mode for loss calculation
            if self._should_freeze_bn():
                self._freeze_bn()
        y_list = []
        for det in detections:
            y_list.append(
                {
                    "boxes": torch.from_numpy(det["boxes"]).float().to(self.device),
                    "labels": torch.from_numpy(det["labels"]).long().to(self.device),
                }
            )
        total_loss = self.compute_loss(x_torch, y_list)
        total_loss = -total_loss
        # Compute gradients
        self.model.zero_grad()
        grad_tensor = torch.autograd.grad(
            outputs=total_loss,
            inputs=x_torch,
            retain_graph=False,  # Can be False as we are done with this loss value
            create_graph=False,
            allow_unused=True,  # If original_total_loss was 0 and didn't depend on x_torch
        )[0]
        grads = grad_tensor.cpu().numpy()
        if self.clip_values is not None and grad_tensor is not None:
            grads = grads / self.clip_values[1]
        return grads

    def compute_object_fabrication_gradient(
        self, x: np.ndarray, detections: dict = None, training: bool = False
    ) -> np.ndarray:
        x_pre = torch.from_numpy(x).to(self.device, dtype=torch.float32)
        x_pre.requires_grad_(True)
        if training:
            self.model.train()
            if self._should_freeze_bn():
                self._freeze_bn()
        preds = self.model.predict(x_pre, training=training)
        loss = self.model.calculate_loss(preds, target_val=1.0)
        self.model.zero_grad()
        grad_tensor = torch.autograd.grad(
            outputs=loss,
            inputs=x_pre,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        grads = grad_tensor.cpu().numpy()
        if self.clip_values is not None:
            grads = grads / self.clip_values[1]  # Undo scaling
        return grads

    def compute_object_mislabeling_gradient(
        self, detections, x, target_labels_list=None, training=True
    ):
        """Compute gradients for the mislabeling attack."""
        if not detections or not any(
            len(det.get("labels", [])) > 0 for det in detections
        ):
            return np.zeros_like(x)
        x_torch = torch.from_numpy(x).to(self.device, dtype=torch.float32)
        x_torch.requires_grad_(True)
        if training:
            self.model.train()
            if self._should_freeze_bn():
                self._freeze_bn()
        if not target_labels_list:
            return np.zeros_like(x)
        # Temporarily modify loss weights to isolate classification loss
        # Use the total loss, which now only consists of the classification component
        loss_components, _ = self._get_losses(x=x_torch, y=target_labels_list)
        total_loss = loss_components["loss_total"]
        self.model.zero_grad()
        grad_tensor = torch.autograd.grad(
            outputs=total_loss,
            inputs=x_torch,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]
        if grad_tensor is None:
            return np.zeros_like(x)
        grads = grad_tensor.cpu().numpy()
        if self.clip_values is not None:
            grads = grads / self.clip_values[1]
        return grads

    def extract_total_loss(self, model_output):
        if not isinstance(model_output, dict):
            raise RuntimeError("FasterRCNN expected dict output")
        if "loss_total" in model_output:
            return model_output["loss_total"]
        return sum(v for k, v in model_output.items() if k.startswith("loss_"))

    def _should_freeze_bn(self):
        return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

    def _freeze_bn(self):
        for m in self.model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()
                m.track_running_stats = False
                for p in m.parameters():
                    p.requires_grad = False

    def compute_loss(self, x, y):
        loss_components, _ = self._get_losses(x=x, y=y)
        # Compute the loss
        if self.weight_dict is None:
            loss = sum(
                loss_components[loss_name]
                for loss_name in self.attack_losses
                if loss_name in loss_components
            )
        else:
            loss = sum(
                loss_component * self.weight_dict[loss_name]
                for loss_name, loss_component in loss_components.items()
                if loss_name in self.weight_dict
            )
        assert isinstance(loss, torch.Tensor)
        if isinstance(x, torch.Tensor):
            return loss
        return loss.detach().cpu().numpy()

    def _get_losses(self, x, y):
        self.model.train()
        if self._should_freeze_bn():
            self._freeze_bn() 
        x_preprocessed = self.model.preprocess_x_for_loss_calculation(x, requires_grad=True)
        y_preprocessed = self.model.translate_labels(y, batch_size=x_preprocessed.shape[0])
        loss_components = self.model(x_preprocessed, y_preprocessed)
        return loss_components, x_preprocessed

    def loss_gradient(self, x, y, **kwargs):
        loss_components, x_grad = self._get_losses(x=x, y=y)
        # Compute the loss
        loss = sum(
            loss_components[loss_name]
            for loss_name in self.attack_losses
            if loss_name in loss_components
        )
        # Clean gradients
        self.model.zero_grad()
        # Compute gradients
        loss.backward(retain_graph=True)  # type: ignore
        grads: torch.Tensor | np.ndarray
        if x_grad.grad is not None:
            if isinstance(x, np.ndarray):
                grads = x_grad.grad.cpu().numpy()
            else:
                grads = x_grad.grad.clone()
        else:
            raise ValueError("Gradient term in PyTorch model is `None`.")
        if self.clip_values is not None:
            grads = grads / self.clip_values[1]
        assert grads.shape == x.shape
        return grads

    def predict(
        self, x_preprocessed: np.ndarray, batch_size: int = 8, **kwargs
    ) -> list[dict[str, np.ndarray]]:
        self.inference_model.eval()
        # Standardize: must return one Tensor (N,C,H,W) on the correct device
        x_tensor = self.model.preprocess_x_for_loss_calculation(x_preprocessed, requires_grad=False)
        assert isinstance(x_tensor, torch.Tensor) and x_tensor.dim() == 4, \
        "preprocess_x_for_loss_calculation must return (N,C,H,W) tensor"
        predictions: list[dict[str, np.ndarray]] = []
        N = x_tensor.shape[0]
        for start in range(0, N, batch_size):
            x_batch = x_tensor[start:start + batch_size].to(self.device)
            batch_preds = self.model.predict_per_batch(x_batch, self.inference_model, self.clip_values)
            predictions.extend(self.filter_boxes(p, self.conf_thresh) for p in batch_preds)
        return predictions