import logging
import random
from typing import Optional, Union

from advsecurenet.computer_vision.object_detection.attacks.pixel_perturbation_based.tog.tog_attack_type import TOGAttackType
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from advsecurenet.computer_vision.image_classification.attacks import AdversarialAttack
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import (
    AdversarialTrainingConfig,
)
from advsecurenet.computer_vision.base.base_adversarial_training import BaseAdversarialTraining
logger = logging.getLogger(__name__)


def coco_targets_to_yolov5(targets_dict, input_shape, device):
    # input_shape: (B, C, H, W)
    height = input_shape[2]
    width  = input_shape[3]
    boxes_list  = targets_dict["boxes"]
    labels_list = targets_dict["labels"]
    labels_xcycwh_list = []
    for i, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
        boxes  = boxes.to(device).float()
        labels = labels.to(device).long()
        N = boxes.shape[0]
        if N == 0:
            # no targets in this image → append an empty (0,6) tensor so cat() works
            labels_xcycwh_list.append(torch.zeros((0, 6), device=device))
            continue
        lab = torch.zeros((N, 6), device=device)
        lab[:, 0] = i                       # image index
        lab[:, 1] = labels                  # class id
        # x1y1x2y2 → normalized x_center,y_center,w,h in [0,1]
        x1, y1, x2, y2 = boxes.T
        lab[:, 2] = (x1 + x2) / 2 / width
        lab[:, 3] = (y1 + y2) / 2 / height
        lab[:, 4] = (x2 - x1)     / width
        lab[:, 5] = (y2 - y1)     / height
        labels_xcycwh_list.append(lab)
    if len(labels_xcycwh_list) == 0:
        return torch.zeros((0, 6), device=device)
    return torch.cat(labels_xcycwh_list, dim=0)



class AdversarialODTraining(BaseAdversarialTraining):
    """
    Adversarial Training class for object detection (OD).

    Args:
        config (AdversarialTrainingConfig): The configuration for the Adversarial Training defense.

    """

    def __init__(self, config: AdversarialTrainingConfig) -> None:
        self._check_config(config)
        super().__init__(config)
        self._trainable = getattr(self._model, "model", self._model)
        for p in self._trainable.parameters():
            p.requires_grad_(True)
        if not any(g["params"] for g in self._optimizer.param_groups):
            kwargs = self._config.optimizer_kwargs or {}
            self._optimizer = self._get_optimizer(
                self._config.optimizer, self._trainable, self._config.learning_rate, **kwargs
            )
            self._scheduler = self._get_scheduler(self._config.scheduler, self._optimizer)

    def _check_config(self, config: AdversarialTrainingConfig) -> None:
        self._check_config_base(config)
        # Shape check on the first sample to ensure OD targets are list[dict]
        try:
            sample = config.train_loader.dataset[0]
            if not (isinstance(sample, tuple) and len(sample) == 2):
                raise ValueError
            _img, _tgt = sample
            # Accept single-target dict (typical detection dataset) OR list[dict]
            if isinstance(_tgt, dict):
                return
            if isinstance(_tgt, list) and all(isinstance(d, dict) for d in _tgt):
                return
            raise ValueError
        except Exception:
            raise ValueError(
                "AdversarialODTraining expects dataset samples like "
                "(image_tensor, targets_list_of_dicts)."
            )
        
    def _shuffle_data(
        self, data: Union[torch.Tensor, list], target: Union[torch.Tensor, list, dict]
    ) -> tuple[Union[torch.Tensor, list], Union[torch.Tensor, list, dict]]:
        if isinstance(target, dict):
            assert isinstance(data, torch.Tensor), "images must be a Tensor when targets are dict-of-lists"
            perm = torch.randperm(data.size(0), device=data.device)
            data = data[perm]
            idx = perm.tolist()
            target = {k: [v[i] for i in idx] for k, v in target.items()}
            return data, target
        if isinstance(data, list):
            perm = torch.randperm(len(data))
            data_shuffled = [data[i] for i in perm]
            if isinstance(target, torch.Tensor):
                target_shuffled = target[perm]
            else:
                target_shuffled = [target[i] for i in perm]
            return data_shuffled, target_shuffled
        permutation = torch.randperm(data.size(0))
        if isinstance(target, torch.Tensor):
            return data[permutation], target[permutation]
        shuffled_target = [target[i] for i in permutation]
        return data[permutation], shuffled_target

    def _combine_clean_and_adversarial_data(
        self,
        images: torch.Tensor,
        adv_images: torch.Tensor,
        targets: list[dict],
    ) -> tuple[torch.Tensor, list[dict]]:
        assert images.shape == adv_images.shape, "images and adv_images must match shape"
        combined_data = torch.cat([images, adv_images], dim=0)
        combined_targets = {k: (v + v) for k, v in targets.items()}
        combined_data, combined_targets = self._shuffle_data(combined_data, combined_targets)
        return combined_data, combined_targets
    
    def _generate_adversarial_batch(
        self,
        images: torch.Tensor,
        targets: list[dict],
        target_images: Optional[torch.Tensor] = None,
        target_targets: Optional[list[dict]] = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        model = random.choice(self.config.models).to(self._device).eval()
        attack = random.choice(self.config.attacks)
        was_training = self._trainable.training
        self._trainable.eval()
        try:
            adv_images = self._perform_attack(
                attack=attack,
                model=self._trainable,
                images=images,
                targets=targets,
                target_images=target_images,
                target_targets=target_targets,
            )
        finally:
            if was_training:
                self._trainable.train()
        return adv_images, targets

    def _move_to_device(
        self, images: torch.Tensor, targets: list[dict]
    ) -> tuple[torch.Tensor, list[dict]]:
        if isinstance(images, list):
            images = torch.stack([img.to(self._device) for img in images], dim=0)
        else:
            images = images.to(self._device)
        targets = {k: [t.to(self._device) for t in v] for k, v in targets.items()}
        return images, targets

    def _perform_attack(
        self,
        attack: AdversarialAttack,
        model: BaseModel,
        images: torch.Tensor,
        targets: list[dict],
        target_images: Optional[torch.Tensor] = None,  # kept for forward compat
        target_targets: Optional[list[dict]] = None,   # kept for forward compat
    ) -> torch.Tensor:
        """
        Runs a detection-friendly attack and returns adversarial images (same shape as images).
        """
        attack_name = attack.__class__.__name__
        if (not getattr(attack, "_detector_resolved", False)) and hasattr(attack, "_object_detector") and (
             isinstance(attack._object_detector, str)
             or not hasattr(attack._object_detector, "predict")
         ):
            try:
                from advsecurenet.models.detector_factory import get_object_detector
                det_name = attack._object_detector if isinstance(attack._object_detector, str) else "yolov5"
                detector_wrapper = get_object_detector(det_name.lower())
                underlying = getattr(self._trainable, "model", self._trainable)
                detector_wrapper.model = underlying
                attack._object_detector = detector_wrapper
                attack._detector_resolved = True
            except Exception:
                if hasattr(model, "predict"):
                    attack._object_detector = model
                else:
                    raise RuntimeError(
                        f"Cannot resolve object detector for attack (value={attack._object_detector})."
                    )
        if attack_name == "DPatch":
            if not hasattr(attack, "_optimized_patch"):
                attack._optimized_patch = attack.attack(
                    dataloader=self.config.train_loader, mask=None, device=self._device
                )
            # Applying the cached patch doesn't need grads
            images_np = images.detach().cpu().numpy()
            patched_np = attack.apply_patch(
                x=images_np,
                patch_external=attack._optimized_patch.detach().cpu().numpy(),
                random_location=True,
            )
            return torch.from_numpy(patched_np).to(self._device)

        if attack_name == "TOG":
            images_np = images.detach().cpu().numpy()
            tog_variant = getattr(attack, "attack_type", TOGAttackType.UNTARGETED)
            if isinstance(tog_variant, str):
                tog_variant = TOGAttackType(tog_variant.lower())
            adv_np = attack.attack(x=images_np, tog_variant=tog_variant, tog_mislabeling_mode=getattr(attack, "tog_mislabeling_mode", "ml"))
            return torch.from_numpy(adv_np).to(self._device)

        # Default: leave grads enabled so gradient-based attacks work
        adv_images = attack.attack(model, images, targets)
        return adv_images.detach().to(self._device)
    
    def _run_epoch(self, epoch: int) -> None:
        total_loss = 0.0
        for images, targets in self._get_train_loader(epoch):
            images, targets = self._move_to_device(images, targets)
            adv_images, adv_targets = self._generate_adversarial_batch(images, targets)
            combined_images, combined_targets = self._combine_clean_and_adversarial_data(
                images, adv_images, adv_targets
            )
            loss = self._run_batch(combined_images, combined_targets)
            total_loss += loss

        total_loss /= self._get_loss_divisor()
        self._log_loss(epoch, total_loss)

    def _run_batch(self, source: torch.Tensor, targets: list[dict]) -> float:
        """
        Runs a batch for object detection adversarial training.
        Uses the model's own loss (e.g. YOLOv5's composite loss).
        """
        self._trainable.train()
        self._optimizer.zero_grad()
        yolo_targets = coco_targets_to_yolov5(targets, input_shape=source.shape, device=source.device)
        source.requires_grad = True
        output = self._trainable(source, yolo_targets)
        # Use the model's loss dict
        if isinstance(output, dict) and "loss_total" in output:
            loss = output["loss_total"]
            loss = loss.squeeze()
        else:
            raise RuntimeError("OD model did not return a loss dict with 'loss_total'.")
        loss.backward()
        self._optimizer.step()
        if self._scheduler:
            self._scheduler.step()
        return loss.item()
