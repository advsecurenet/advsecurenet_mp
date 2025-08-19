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


class AdversarialODTraining(BaseAdversarialTraining):
    """
    Adversarial Training class for object detection (OD).

    Args:
        config (AdversarialTrainingConfig): The configuration for the Adversarial Training defense.

    """

    def __init__(self, config: AdversarialTrainingConfig) -> None:
        self._check_config(config)
        super().__init__(config)

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
        self, data: Union[torch.Tensor, list], target: Union[torch.Tensor, list]
    ) -> tuple[Union[torch.Tensor, list], Union[torch.Tensor, list]]:
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
        combined_targets = {k: v + v for k, v in targets.items()}
        return combined_data, combined_targets
    
    def _generate_adversarial_batch(
        self,
        images: torch.Tensor,
        targets: list[dict],
        target_images: Optional[torch.Tensor] = None,
        target_targets: Optional[list[dict]] = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        # Randomly pick a model/attack for diversity (same idea as your classification trainer).
        model = random.choice(self.config.models).to(self._device).eval()
        attack = random.choice(self.config.attacks)

        adv_images = self._perform_attack(
            attack=attack,
            model=model,
            images=images,
            targets=targets,
            target_images=target_images,
            target_targets=target_targets,
        )

        # For OD, ground-truth targets remain the same (untargeted by default).
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
        if attack_name == "DPatch":
            # one-time optimization might need grads internally; let the attack handle that.
            if not hasattr(attack, "_optimized_patch"):
                attack._optimized_patch = attack.attack(
                    dataloader=self.config.train_loader, mask=None, device=self._device
                )
            # Applying the cached patch doesn't need grads
            with torch.no_grad():
                images_np = images.detach().cpu().numpy()
                patched_np = attack.apply_patch(
                    x=images_np,
                    patch_external=attack._optimized_patch.detach().cpu().numpy(),
                    random_location=True,
                )
                return torch.from_numpy(patched_np).to(self._device)

        if attack_name == "TOG":
            # TOG's numpy-based transform doesn't need grads
            with torch.no_grad():
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
