import torch
import torchvision
import torch.nn.functional as F
import yolov5
from yolov5.models.common import AutoShape
from yolov5.utils.general import xywh2xyxy
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from advsecurenet.models.CustomODWrappers.ODWrapper import ODWrapper

class CustomYolov5ODWrapper(ODWrapper):
    def __init__(
        self,
        model,
        input_shape,
        clip_values,
        attack_losses,
        device_type,
        conf_thresh = 0.7,
        weight_dict=None,
    ):
        super().__init__(
            model=model, 
            conf_thresh=conf_thresh, 
            device_type=device_type, 
            clip_values=clip_values,
            input_shape=input_shape,
            )
        self.inference_model = yolov5.load('model_weights\yolov5s.pt', device=device_type, autoshape=True)
        self.inference_model.conf = conf_thresh
        self.input_shape=input_shape
        self.channels_first=True
        self.attack_losses=attack_losses
        self.weight_dict = weight_dict


    def _translate_labels(self, labels: list[dict[str, "torch.Tensor"]]) -> "torch.Tensor":
        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]
        labels_xcycwh_list = []
        for i, label_dict in enumerate(labels):
            # For each image in the batch, make a [N,6] tensor:
            # [ image_index, class_label, x_center, y_center, w, h ]
            # Number of objects in this image
            N = len(label_dict["boxes"])
            # create 2D tensor to encode labels and bounding boxes
            label_xcycwh = torch.zeros((N, 6), device=self.device)
            label_xcycwh[:, 0] = i # image index
            # class labels
            raw_lbls = label_dict["labels"]
            if isinstance(raw_lbls, np.ndarray):
                lbls = torch.from_numpy(raw_lbls).to(self.device)
            else:
                lbls = raw_lbls.to(self.device)
            label_xcycwh[:, 1] = lbls
            # bounding boxes
            raw_boxes = label_dict["boxes"]
            if isinstance(raw_boxes, np.ndarray):
                boxes = torch.from_numpy(raw_boxes).float().to(self.device)
            else:
                boxes = raw_boxes.to(self.device)
            # boxes are [x1, y1, x2, y2]
            label_xcycwh[:, 2:6] = boxes
            # normalize bounding boxes to [0, 1]
            label_xcycwh[:, 2:6:2] /= width
            label_xcycwh[:, 3:6:2] /= height
            # convert from x1y1x2y2 to xcycwh
            label_xcycwh[:, 4] -= label_xcycwh[:, 2]
            label_xcycwh[:, 5] -= label_xcycwh[:, 3]
            label_xcycwh[:, 2] += label_xcycwh[:, 4] / 2
            label_xcycwh[:, 3] += label_xcycwh[:, 5] / 2
            labels_xcycwh_list.append(label_xcycwh)
        labels_xcycwh = torch.vstack(labels_xcycwh_list)
        return labels_xcycwh


    def _translate_predictions(self, predictions: "torch.Tensor") -> list[dict[str, np.ndarray]]:
        if self.channels_first:
            height = self.input_shape[1]
            width = self.input_shape[2]
        else:
            height = self.input_shape[0]
            width = self.input_shape[1]
        predictions_x1y1x2y2: list[dict[str, np.ndarray]] = []
        for pred in predictions:
            # pred is [num_detections, 5 + num_classes]:
            #  [ x_center, y_center, w, h, conf, class_logits... ]
            if isinstance(pred, np.ndarray):
                pred = torch.from_numpy(pred).to(self.device).float()
            elif isinstance(pred, list):
                pred = torch.tensor(pred, dtype=torch.float32, device=self.device)
            boxes = torch.vstack(
                [
                    torch.maximum((pred[:, 0] - pred[:, 2] / 2), torch.tensor(0, device=self.device)),
                    torch.maximum((pred[:, 1] - pred[:, 3] / 2), torch.tensor(0, device=self.device)),
                    torch.minimum((pred[:, 0] + pred[:, 2] / 2), torch.tensor(height, device=self.device)),
                    torch.minimum((pred[:, 1] + pred[:, 3] / 2), torch.tensor(width, device=self.device)),
                ]
            ).permute((1, 0))
            logits = pred[:, 5:]
            labels = torch.argmax(pred[:, 5:], dim=1)
            scores = pred[:, 4]
            pred_dict = {
                "boxes": boxes.detach().cpu().numpy(),
                "labels": labels.detach().cpu().numpy(),
                "scores": scores.detach().cpu().numpy(),
                "logits": logits.detach().cpu().numpy(),
            }
            predictions_x1y1x2y2.append(pred_dict)
        return predictions_x1y1x2y2


    def _get_losses(self, x, y):
        self.model.train()
        # 1) Ensure x is a torch.Tensor on the right device
        if isinstance(x, np.ndarray):
            # x shape is (B, C, H, W), values already in [0..255] or [0..1]
            x_preprocessed = torch.from_numpy(x).float().to(self.device)
        else:
            x_preprocessed = x.float().to(self.device)
        # Move inputs to device
        x_preprocessed.requires_grad = True
        y_preprocessed = self._translate_labels(y)
        loss_components = self.model(x_preprocessed, y_preprocessed)
        return loss_components, x_preprocessed


    def loss_gradient(self, x, y, **kwargs):
        loss_components, x_grad = self._get_losses(x=x, y=y)
        # Compute the loss
        loss = sum(loss_components[loss_name] for loss_name in self.attack_losses if loss_name in loss_components)
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


    def _predict_with_logits(self, x_tensor: torch.Tensor) -> list[dict[str, np.ndarray]]:
        """
        Internal prediction method that guarantees full logits are returned.
        It uses the raw model and performs manual NMS.
        """
        self.model.eval()
        final_predictions = []
        with torch.no_grad():
            raw_pred = self.model(x_tensor)[0]
            conf_thres = self.conf_thresh
            iou_thres = 0.45  # Standard NMS IoU threshold
            max_det = 1000

            bs = raw_pred.shape[0]
            nc = raw_pred.shape[2] - 5
            xc = raw_pred[..., 4] > conf_thres

            for i in range(bs):
                x = raw_pred[i][xc[i]]
                if not x.shape[0]:
                    final_predictions.append({
                        "boxes": np.empty((0, 4)), "scores": np.empty((0,)),
                        "labels": np.empty((0,), dtype=int), "logits": np.empty((0, nc))
                    })
                    continue

                x[:, 5:] *= x[:, 4:5]
                box = xywh2xyxy(x[:, :4])
                conf, j = x[:, 5:].max(1, keepdim=True)
                
                nms_indices = torchvision.ops.nms(box, conf.view(-1), iou_thres)
                if nms_indices.shape[0] > max_det:
                    nms_indices = nms_indices[:max_det]
                
                final_dets = x[nms_indices]
                final_boxes = xywh2xyxy(final_dets[:, :4])
                final_logits = final_dets[:, 5:]
                final_conf, final_labels = final_logits.max(1)

                final_predictions.append({
                    "boxes": final_boxes.cpu().numpy(),
                    "scores": final_conf.cpu().numpy(),
                    "labels": final_labels.cpu().numpy().astype(int),
                    "logits": final_logits.cpu().numpy(),
                })
        return final_predictions


    def predict(self, x_preprocessed: np.ndarray, batch_size: int = 128, **kwargs) -> list[dict[str, np.ndarray]]:
        self.inference_model.eval()
        # Create dataloader
        if isinstance(x_preprocessed, np.ndarray):
        # shape should be (N, C, H, W)
            x_tensor = torch.from_numpy(x_preprocessed).float()
        else:
            x_tensor = x_preprocessed
        dataset = TensorDataset(x_tensor)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False)
        predictions: list[dict[str, np.ndarray]] = []
        for (x_batch,) in dataloader:
            # Move inputs to device
            x_batch = x_batch.to(self.device)
            imgs = x_batch.detach().cpu().numpy()
            imgs = [(img.transpose(1, 2, 0)).clip(0,self.clip_values[1]).astype(np.uint8) for img in imgs]
            with torch.no_grad():
                outputs = self.inference_model(imgs, size=self.input_shape[1])
                for i, det in enumerate(outputs.xyxy):
                    arr = det.cpu().numpy() if isinstance(det, torch.Tensor) else det
                    if arr.size == 0:
                        predictions.append({"boxes": np.empty((0, 4)), "scores": np.empty((0,)), "labels": np.empty((0,), dtype=int)})
                    else:
                        # Get raw logits from outputs.pred
                        raw_pred = outputs.pred[i].cpu().numpy()  # shape: [num_detections, 5 + num_classes]
                        logits = raw_pred[:, 5:]  # shape: [num_detections, num_classes]
                        predictions.append({
                            "boxes": arr[:, :4],  # x1, y1, x2, y2
                            "scores": arr[:, 4],  
                            "labels": arr[:, 5].astype(int),
                            "logits": logits,  # Add logits here
                        })
        return predictions


    def compute_loss(self, x, y, **kwargs):
        loss_components, _ = self._get_losses(x=x, y=y)
        # Compute the loss
        if self.weight_dict is None:
            loss = sum(loss_components[loss_name] for loss_name in self.attack_losses if loss_name in loss_components)
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
    

    def compute_object_vanishing_gradient(self, x: np.ndarray, training: bool = False) -> np.ndarray:
        x = torch.from_numpy(x)
        x_pre = x.to(self.device, dtype=torch.float32)
        x_pre.requires_grad_(True)
        if training:
            self.model.train()                 # ensure we get raw preds, not autoshaped outputs
        preds = self.model(x_pre)[0]          # list of 3 tensors: (bs, na, gh, gw, 5+nc)
        # 3) Compute the TF-identical vanishing loss:
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for p in preds:
            # p[..., 4] is the objectness logit
            obj_logit = p[..., 4]        # shape (bs, na, gh, gw)
            zeros     = torch.zeros_like(obj_logit, device=self.device)
            # sum reduction matches K.sum(...)/nothing
            loss     += F.binary_cross_entropy_with_logits(
                             obj_logit, zeros,
                             reduction='sum'
                         )
        grad_tensor = torch.autograd.grad(
            outputs=loss,
            inputs=x_pre,
            retain_graph=False,
            create_graph=False,
            allow_unused=False
        )[0]
        grads = grad_tensor.cpu().numpy()
        # 5) undo any scaling
        if self.clip_values is not None:
            grads = grads / self.clip_values[1]
        return grads
    

    def compute_object_untargeted_gradient(self, x: np.ndarray, detections: list[dict[str, np.ndarray]] = None, training: bool = True) -> np.ndarray:
        if not detections:
            return np.zeros_like(x)
        x_torch = torch.from_numpy(x).to(self.device, dtype=torch.float32)
        x_torch.requires_grad_(True)
        if training:
            self.model.train()  # Set model to training mode for loss calculation
        y_list = []
        for det in detections:
            y_list.append({
                "boxes": torch.from_numpy(det["boxes"]).float().to(self.device),
                "labels": torch.from_numpy(det["labels"]).long().to(self.device),
            })
        total_loss = self.compute_loss(x_torch, y_list)
        total_loss = -total_loss
        # Compute gradients
        self.model.zero_grad() 
        grad_tensor = torch.autograd.grad(
            outputs=total_loss,
            inputs=x_torch,
            retain_graph=False, # Can be False as we are done with this loss value
            create_graph=False,
            allow_unused=True # If original_total_loss was 0 and didn't depend on x_torch
        )[0]
        grads = grad_tensor.cpu().numpy()
        if self.clip_values is not None and grad_tensor is not None:
            grads = grads / self.clip_values[1]
        return grads


    def compute_object_fabrication_gradient(self, x: np.ndarray, detections: dict = None, training: bool = False) -> np.ndarray:
        x_pre = torch.from_numpy(x).to(self.device, dtype=torch.float32)
        x_pre.requires_grad_(True)
        if training:
            self.model.train()
        preds = self.model(x_pre)[0]
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for p in preds: # Iterate over batch items
            obj_logit = p[..., 4]  # Objectness logit for current batch item
            ones = torch.ones_like(obj_logit, device=self.device)
            loss += F.binary_cross_entropy_with_logits(obj_logit, ones, 
                                                       reduction='sum',
                                                       )
        grad_tensor = torch.autograd.grad(
            outputs=loss,
            inputs=x_pre,
            retain_graph=False,
            create_graph=False,
            allow_unused=False
        )[0]
        grads = grad_tensor.cpu().numpy()
        if self.clip_values is not None:
            grads = grads / self.clip_values[1] # Undo scaling
        return grads


    def compute_object_mislabeling_gradient(self, x, detections=None, mode="ml", training=True):
        """Compute gradients for the mislabeling attack."""
        if not detections or not any(len(det.get('labels', [])) > 0 for det in detections):
            return np.zeros_like(x)
        mode = mode.lower()
        if mode not in ['ml', 'll']:
            print(f"Warning: Unknown mode '{mode}'. Using 'ml' instead.")
            mode = 'ml'
        x_torch = torch.from_numpy(x).to(self.device, dtype=torch.float32)
        x_torch.requires_grad_(True)
        if training:
            self.model.train()
        # Get number of classes from model
        num_classes = len(self.inference_model.model.names)
        target_labels_list = []
        for det in detections:
            if len(det.get('labels', [])) == 0:
                continue
            boxes = torch.tensor(det["boxes"], dtype=torch.float32, device=self.device)
            original_labels = torch.from_numpy(det['labels']).long().to(self.device)
            # Choose target labels based on the specified mode
            new_labels = original_labels.clone()
            for i in range(len(original_labels)):
                orig_label = original_labels[i].item()
                # Safe random selection if orig_label is out of range
                if orig_label >= num_classes:
                    print(f"Warning: Label {orig_label} is out of range (num_classes={num_classes})")
                    new_labels[i] = np.random.randint(0, num_classes)
                    continue
                # Use prediction confidence to create pseudo-logits if not available
                if 'logits' not in det or det['logits'] is None or i >= len(det['logits']):
                    # Generate random alternative class
                    new_label = orig_label
                    while new_label == orig_label:
                        new_label = np.random.randint(0, num_classes)
                    new_labels[i] = new_label
                else:
                    # Use actual logits for smart targeting
                    obj_logits = torch.from_numpy(det['logits'][i]).to(self.device)
                    # Make sure logits array has enough elements
                    if len(obj_logits) <= orig_label:
                        # Generate pseudo-logits to accommodate the original label
                        pseudo_logits = torch.randn(max(num_classes, orig_label+1), device=self.device)
                        # Copy existing values
                        pseudo_logits[:len(obj_logits)] = obj_logits
                        obj_logits = pseudo_logits
                    if mode == 'll':
                        # Least likely: find class with lowest logit value
                        obj_logits_mod = obj_logits.clone()
                        obj_logits_mod[orig_label] = float('inf')  # Exclude original class
                        new_labels[i] = torch.argmin(obj_logits_mod).item()
                    else:  # mode == 'ml'
                        # Most likely: find class with highest logit value (excluding original)
                        obj_logits_mod = obj_logits.clone()
                        obj_logits_mod[orig_label] = float('-inf')  # Exclude original class
                        new_labels[i] = torch.argmax(obj_logits_mod).item()
            target_labels_list.append({
                "boxes": boxes,
                "labels": new_labels,
            })
        if not target_labels_list:
            return np.zeros_like(x)
        # Temporarily modify loss weights to isolate classification loss
        original_hyp = self.model._model.hyp.copy() if hasattr(self.model, '_model') else None
        try:
            # Use the total loss, which now only consists of the classification component
            loss_components, _ = self._get_losses(x=x_torch, y=target_labels_list)
            total_loss = loss_components['loss_total']
            print(f"[DEBUG] Loss Components: {loss_components}")
            self.model.zero_grad()
            grad_tensor = torch.autograd.grad(
                outputs=total_loss,
                inputs=x_torch,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )[0]
        finally:
            # Restore original hyperparameters
            if original_hyp:
                self.model._model.hyp = original_hyp
        if grad_tensor is None:
            return np.zeros_like(x)
        grads = grad_tensor.cpu().numpy()
        if self.clip_values is not None:
            grads = grads / self.clip_values[1]
        return grads