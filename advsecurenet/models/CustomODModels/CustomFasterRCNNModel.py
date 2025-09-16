import torch
import torchvision
import numpy as np
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights

from advsecurenet.models.CustomODModels.CustomODBaseModel import CustomODBaseModel


class CustomFasterRCNNModel(CustomODBaseModel):
    def __init__(
            self, 
            num_classes: int = 91, 
            pretrained: bool = True, 
            pretrained_backbone: bool = True, 
            device: str | int | torch.device | None = None,
            ):
        super().__init__()
        self.expects_numpy_images = False
        self.num_classes = num_classes
        if pretrained:
            self._model = fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained_backbone else None,
                num_classes=None if pretrained_backbone else num_classes
            )
        else:
            self._model = fasterrcnn_resnet50_fpn_v2(
                weights=None,
                num_classes=num_classes
            )
        self._model_name = "CustomFasterRCNNModel"
        self.categories = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.meta["categories"]
        # Resolve and move the model
        if device is None:
            device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self._model.to(self.device)


    def forward(self, x, targets=None):
        """
        Forward pass for Faster R-CNN.
        
        Args:
            x: Input images tensor of shape (N, C, H, W)
            targets: List of dicts with 'boxes' and 'labels' keys (for training)
            
        Returns:
            During training: dict with loss components
            During inference: list of dicts with 'boxes', 'scores', 'labels'
        """
        if isinstance(x, torch.Tensor):
            x = [x[i] for i in range(x.shape[0])]
        if self.training and targets is not None:
            loss_dict = self._model(x, targets)
            total_loss = sum(loss_dict.values())
            loss_dict["loss_total"] = total_loss
            return loss_dict
        else:
            return self._model(x)

    def predict(self, x, training):
        if not training:
            self.eval()
        return self.forward(x, None)
    

    def predict_raw(self, x):
        """
        Predicts raw outputs without post-processing.
        """
        return self._model(x)


    def predict_per_batch(self, imgs, inference_model, clip_values):
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs[i] for i in range(imgs.shape[0])]
        outs = inference_model(imgs)
        preds = self._translate_predictions(outs)
        return preds
    

    def initialize_inference_model(self, model, device, conf_thresh=0.7):
        inference_model = model.to(device)
        inference_model.eval()
        return inference_model
        
    
    def prepare_training_inputs(self, images: torch.Tensor, targets: list[dict]):
        img_list = [images[i] for i in range(images.size(0))]
        for t in img_list:
            t.requires_grad_(True)
        return img_list, targets


    def calculate_loss(self, predictions, target_val):
        # target_val = 0 - subtract, target_val = 1 - add
        target_val = 2 * target_val - 1 # map 0 -> -1, 1 -> 1
        loss = torch.tensor(0.0, device=self.device, dtype=torch.float32)
        for pred in predictions:
            if "scores" in pred:
                scores = pred["scores"]
                loss -= (torch.sum(scores) * target_val)
            else:
                if "logits" in pred:
                    logits = pred["logits"]
                    loss -= (torch.sum(torch.max(logits, dim=1)[0]) * target_val)
        return loss


    def preprocess_x_for_loss_calculation(self, x, requires_grad=True):
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
        if requires_grad:
            x_tensor.requires_grad_(True)
        #return [x_tensor[i] for i in range(x_tensor.shape[0])]
        return x_tensor
    

    def translate_labels(self, labels: list[dict[str, torch.Tensor | np.ndarray]], batch_size: int):
        """From your labels [{'boxes': Nx4, 'labels': N}, …] to
           torchvision's targets: list of dicts with Tensors."""
        y = self._align_targets_to_batch(labels, batch_size=batch_size)
        targets = []
        for lab in y:
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


# Model specific methods / helpers:

    def _empty_target_np(self):
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
    