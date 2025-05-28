import numpy as np
import torch
from torch.nn import functional as F

class ODWrapper:
    def __init__(
        self,
        model,
        conf_thresh,
        device_type,
        clip_values,
        input_shape,
    ):
        self.model=model
        self.conf_thresh=conf_thresh
        self.device=device_type
        self.clip_values=clip_values
        self.input_shape = input_shape

    def filter_boxes(self, predictions, conf_thresh):
        dictionary = {}
        boxes_list = []
        scores_list = []
        labels_list = []
        for i in range(len(predictions["boxes"])):
            score = predictions["scores"][i]
            if score >= conf_thresh:
                boxes_list.append(predictions["boxes"][i])
                scores_list.append(predictions["scores"][[i]])
                labels_list.append(predictions["labels"][[i]])
        if len(boxes_list)>0 and len(scores_list)>0 and len(labels_list)>0:
            dictionary["boxes"] = np.vstack(boxes_list)
            dictionary["scores"] = np.hstack(scores_list)
            dictionary["labels"] = np.hstack(labels_list)
        y = dictionary
        return y
    
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