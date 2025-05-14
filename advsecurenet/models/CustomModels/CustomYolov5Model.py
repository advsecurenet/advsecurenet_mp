import torch
import yolov5
from yolov5.utils.loss import ComputeLoss
from yolov5.models.common import AutoShape

class MyYolo(torch.nn.Module):
        def __init__(self, model_weights_path="/home/user/arutkiewicz/advsecurenet/code/advsecurenet_mp/yolov5s.pt"):
            super().__init__()
            det = yolov5.load(model_weights_path, autoshape=False)
            self._model = det.model
            self._model.hyp.update({'box': 0.05,
                            'obj': 1.0,
                            'cls': 0.5,
                            'anchor_t': 4.0,
                            'cls_pw': 1.0,
                            'obj_pw': 1.0,
                            'fl_gamma': 0.0
                            })
            self.compute_loss = ComputeLoss(self._model)
            self._detector = AutoShape(self._model)

        def forward(self, x, targets=None):
            if self.training:
                outputs = self._model(x)
                loss, loss_items = self.compute_loss(outputs, targets)
                loss_components_dict = {"loss_total": loss}
                loss_components_dict['loss_box'] = loss_items[0]
                loss_components_dict['loss_obj'] = loss_items[1]
                loss_components_dict['loss_cls'] = loss_items[2]
                return loss_components_dict
            else:
                return self._detector(x)