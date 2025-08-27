import torch
import torch.nn.functional as F
import numpy as np
from advsecurenet.models.CustomODWrappers.ODWrapper import ODWrapper
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights


class CustomFasterRCNNODWrapper(ODWrapper):
    def __init__(
        self,
        model: torch.nn.Module,
        input_shape: tuple[int, int, int],
        clip_values: tuple[float, float],
        attack_losses: tuple[str, ...],
        device_type: str,
        conf_thresh: float = 0.7,
        weight_dict: dict[str, float] | None = None,
    ):
        super().__init__(
            model=model,
            conf_thresh=conf_thresh,
            device_type=device_type,
            clip_values=clip_values,
            input_shape=input_shape,
        )
        self.inference_model = model.to(self.device)
        self.inference_model.eval()
        self.attack_losses = attack_losses
        self.weight_dict = weight_dict
        self.channels_first = True
        if hasattr(model, 'num_classes'):
            self.num_classes = model.num_classes
        else:
            # Default to COCO classes if not specified
            self.num_classes = 91  # COCO has 90 classes + background
        self.categories = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]
        self.expects_numpy_images = False


    def _empty_target_np(self):
    # An empty target the torchvision detector accepts
        return {
            "boxes":  np.empty((0, 4), dtype=np.float32),
            "labels": np.empty((0,),   dtype=np.int64),
        }


    def _align_targets_to_batch(self, y, batch_size: int):
        """
        Ensure we have exactly one target dict per image.
        Pads with empty targets or truncates if needed.
        """
        if y is None:
            return [self._empty_target_np() for _ in range(batch_size)]
        y = list(y)
        if len(y) < batch_size:
            y = y + [self._empty_target_np() for _ in range(batch_size - len(y))]
        elif len(y) > batch_size:
            y = y[:batch_size]
        return y


    def _translate_labels(self, labels: list[dict[str, torch.Tensor | np.ndarray]]):
        """From your labels [{'boxes': Nx4, 'labels': N}, …] to
           torchvision's targets: list of dicts with Tensors."""
        targets = []
        for lab in labels:
            boxes = lab["boxes"]
            classes = lab["labels"]
            if isinstance(boxes, np.ndarray):
                boxes = torch.from_numpy(boxes).float()
            if isinstance(classes, np.ndarray):
                classes = torch.from_numpy(classes).long()
            targets.append({
                "boxes": boxes.to(self.device),
                "labels": classes.to(self.device),
            })
        return targets
    

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
        # boolean mask; safe when empty
        mask = scores >= conf_thresh #if scores.size else np.zeros((boxes.shape[0],), dtype=bool)
        out = {
            "boxes":  boxes[mask] if boxes.size else boxes,    # -> (0,4) when empty
            "scores": scores[mask] if scores.size else scores, # -> (0,)
            "labels": labels[mask] if labels.size else labels, # -> (0,)
        }
        if names is not None:
            # keep key but return empty array when no detections
            out["label_names"] = names[mask] if len(names) else np.empty((0,), dtype=names.dtype)
        return out


    def _translate_predictions(self, outputs: list[dict[str, torch.Tensor]]):
        """From torchvision outputs (list of dicts) back to your np format."""
        preds = []
        for out in outputs:
            boxes = out["boxes"].detach().cpu().numpy()
            scores = out["scores"].detach().cpu().numpy()
            labels = out["labels"].detach().cpu().numpy()
            pred = {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
            pred["label_names"] = np.array([self.categories[int(l)] for l in labels])
            preds.append(pred)
        return preds


    def _get_losses(self, x, y):
        """Run model in train mode to get the loss dict."""
        self.model.train()
        # build list[Tensor(C,H,W)] float32 in [0,1]
        imgs = []
        if isinstance(x, np.ndarray):
            for i in range(x.shape[0]):
                t = torch.from_numpy(x[i]).to(self.device).float()
                if t.max() > 1:
                    t = t / 255.0
                imgs.append(t)
        else:
            if x.dim() == 4:
                for i in range(x.shape[0]):
                    t = x[i].to(self.device).float()
                    if t.max() > 1:
                        t = t / 255.0
                    imgs.append(t)
            else:
                t = x.to(self.device).float()
                if t.max() > 1:
                    t = t / 255.0
                imgs = [t]
        x_tensor = torch.stack(imgs, dim=0)
        y = self._align_targets_to_batch(y, batch_size=x_tensor.shape[0])
        targets = self._translate_labels(y)
        x_tensor.requires_grad_(True)
        imgs = [x_tensor[i] for i in range(x_tensor.shape[0])]
        loss_dict = self.model(imgs, targets)
        return loss_dict, x_tensor


    def loss_gradient(self, x, y, **kwargs):
        loss_dict, x_tensor = self._get_losses(x, y)
        loss = sum(
            loss_dict[k]
            for k in self.attack_losses
            if k in loss_dict
        )
        self.model.zero_grad()
        loss.backward(retain_graph=True)
        grads = x_tensor.grad.detach().cpu().numpy()
        if self.clip_values:
            grads = grads / self.clip_values[1]
        return grads


    def compute_loss(self, x, y):
        loss_dict, _ = self._get_losses(x, y)
        if self.weight_dict:
            return sum(
                loss_dict[k] * self.weight_dict[k]
                for k in loss_dict
                if k in self.weight_dict
            )
        else:
            return sum(
                loss_dict[k]
                for k in self.attack_losses
                if k in loss_dict
            )


    def predict(self, x_preprocessed: np.ndarray, batch_size: int = 8, **kwargs):
        """Batch inference: takes (N,C,H,W) numpy, returns list of dicts."""
        self.inference_model.eval()
        if isinstance(x_preprocessed, np.ndarray):
            imgs = []
            for i in range(x_preprocessed.shape[0]):
                t = torch.from_numpy(x_preprocessed[i]).to(self.device).float()
                if t.max() > 1:
                    t = t / 255.0
                imgs.append(t)
        else:
            imgs = []
            for img in x_preprocessed:
                t = img.to(self.device).float()
                if t.max() > 1:
                    t = t / 255.0
                imgs.append(t)
        outputs = []
        with torch.no_grad():
            for i in range(0, len(imgs), batch_size):
                batch = imgs[i : i + batch_size]
                outs = self.inference_model(batch)
                outputs.extend(outs)
        preds = self._translate_predictions(outputs)
        filtered_preds = [self.filter_boxes(p, self.conf_thresh) for p in preds]
        return filtered_preds


    def compute_object_vanishing_gradient(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        """Compute vanishing gradient by minimizing objectness scores."""
        x_torch = torch.from_numpy(x).to(self.device, dtype=torch.float32)
        x_torch.requires_grad_(True)
        if training:
            self.model.train()
        else:
            self.model.eval()        
        predictions = self.model(x_torch)
        vanishing_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for pred in predictions:
            if "scores" in pred:
                scores = pred["scores"]
                vanishing_loss += torch.sum(scores)
            else:
                if "logits" in pred:
                    logits = pred["logits"]
                    # Minimize the maximum logit (make all classes less confident)
                    vanishing_loss += torch.sum(torch.max(logits, dim=1)[0])
        self.model.zero_grad()
        grad_tensor = torch.autograd.grad(
            outputs=vanishing_loss,
            inputs=x_torch,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        grads = grad_tensor.cpu().numpy()
        if self.clip_values is not None:
            grads = grads / self.clip_values[1]
        return grads


    def compute_object_untargeted_gradient(
        self,
        x: np.ndarray,
        detections: list[dict[str, np.ndarray]] = None,
        training: bool = True,
    ):
        if not detections:
            return np.zeros_like(x)
        x_torch = torch.from_numpy(x).to(self.device, dtype=torch.float32)
        x_torch.requires_grad_(True)
        if training:
            self.model.train()
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
        self.model.zero_grad()
        grad_tensor = torch.autograd.grad(
            outputs=total_loss,
            inputs=x_torch,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]
        grads = grad_tensor.cpu().numpy()
        if self.clip_values is not None and grad_tensor is not None:
            grads = grads / self.clip_values[1]
        return grads


    def compute_object_fabrication_gradient(self, x: np.ndarray, detections=None, training: bool = False) -> np.ndarray:
        """Compute fabrication gradient by maximizing false positive detections."""
        x_torch = torch.from_numpy(x).to(self.device, dtype=torch.float32)
        x_torch.requires_grad_(True)
        if training:
            self.model.train()
        else:
            self.model.eval()
        predictions = self.model(x_torch)
        fabrication_loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for pred in predictions:
            if "scores" in pred:
                scores = pred["scores"]
                fabrication_loss -= torch.sum(scores)
            else:
                if "logits" in pred:
                    logits = pred["logits"]
                    fabrication_loss -= torch.sum(torch.max(logits, dim=1)[0])
        self.model.zero_grad()
        grad_tensor = torch.autograd.grad(
            outputs=fabrication_loss,
            inputs=x_torch,
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]
        grads = grad_tensor.cpu().numpy()
        if self.clip_values is not None:
            grads = grads / self.clip_values[1]
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
        if not target_labels_list:
            return np.zeros_like(x)
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