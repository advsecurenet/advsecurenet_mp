import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


class CustomFasterRCNNModel(torch.nn.Module):
    def __init__(self, num_classes: int = 91, pretrained: bool = True, pretrained_backbone: bool = True):
        super().__init__()
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
        if self.training and targets is not None:
            loss_dict = self._model(x, targets)
            total_loss = sum(loss_dict.values())
            loss_dict["loss_total"] = total_loss
            return loss_dict
        else:
            return self._model(x)
    

    def predict_raw(self, x):
        """
        Predicts raw outputs without post-processing.
        """
        return self._model(x)