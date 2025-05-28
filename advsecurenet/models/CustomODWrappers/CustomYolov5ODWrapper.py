import torch
import torch.nn.functional as F
import yolov5
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
            labels = torch.argmax(pred[:, 5:], dim=1)
            scores = pred[:, 4]
            pred_dict = {
                "boxes": boxes.detach().cpu().numpy(),
                "labels": labels.detach().cpu().numpy(),
                "scores": scores.detach().cpu().numpy(),
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
                for det in outputs.xyxy:
                    arr = det.cpu().numpy() if isinstance(det, torch.Tensor) else det
                    if arr.size == 0:
                        predictions.append({"boxes": np.empty((0, 4)), "scores": np.empty((0,)), "labels": np.empty((0,), dtype=int)})
                    else:
                        predictions.append({
                            "boxes": arr[:, :4],  # x1, y1, x2, y2
                            "scores": arr[:, 4],  
                            "labels": arr[:, 5].astype(int),
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