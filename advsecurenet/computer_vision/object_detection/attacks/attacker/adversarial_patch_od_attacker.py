# advsecurenet/computer_vision/object_detection/attacks/attacker/default_od_attacker.py

import logging
import numpy as np
import torch
import click

from advsecurenet.evaluation.od_adversarial_evaluator import ObjectDetectorAdversarialEvaluator
from advsecurenet.shared.types.configs.attack_configs.od_attacker_config import ODAttackerConfig
from advsecurenet.computer_vision.object_detection.attacks.attacker.od_attacker import ODAttacker
from advsecurenet.utils.device_utils import move_batch_to_device

logger = logging.getLogger(__name__)

class AdversarialPatchODAttacker(ODAttacker):
    """
    Attacks an object detection model using an adversarial patch.
    """
    def __init__(self, config: ODAttackerConfig):
        super().__init__(config)
        self._trained_patch = None

    def execute(self):
        adversarial_images = []
        with ObjectDetectorAdversarialEvaluator(
            evaluators    = self._config.evaluators,
            target_models = [self._eval_model],      # we evaluate on the eval_model
        ) as evaluator:
            # 1) GENERATE the patch
            self._trained_patch = self._config.attack.attack(
                mask = getattr(self._config.attack, "mask", None),
                dataloader = self._dataloader,
                device = self._device,
            )

            for data_batch in self._dataloader:
                images, targets_dict = data_batch
                images, targets_dict = move_batch_to_device(images, targets_dict, self._device)
                images_np_for_dpatch = (images.detach().cpu().numpy() * 255.0).astype(np.float32)
                images_np_for_dpatch = np.clip(images_np_for_dpatch, 0, 255)
                boxes  = targets_dict["boxes"]
                labels = targets_dict["labels"]
                targets = []
                for b, l in zip(boxes, labels):
                    raw = l.detach().cpu().numpy().astype(int)      # e.g. [1, 3, 18, …]
                    mapped = np.array(raw, dtype=int)
                    targets.append({
                        "boxes":  b.detach().cpu().numpy(),
                        "labels": mapped,
                        "scores": np.ones(len(mapped), dtype=float),
                    })
                # 2) APPLY the patch to *this* batch of images
                patched_np = self._config.attack.apply_patch(
                    x               = images_np_for_dpatch,
                    patch_external  = self._trained_patch.detach().cpu().numpy(),
                    random_location = False
                ).detach().cpu().numpy()
                patched = torch.from_numpy(patched_np / 255.0).to(self._device)
                # #3) EVALUATE on the patched images
                evaluator.update(
                    model              = self._eval_model,
                    original_images    = images,
                    adversarial_images = patched,
                    targets            = targets,
                )
                if self._config.return_adversarial_images:
                    adversarial_images.append(patched.detach().cpu())
                # free up GPU memory if needed
                if torch.cuda.is_available() and self._device.type == 'cuda':
                    torch.cuda.empty_cache()
            # summary logging
            results = evaluator.get_results()
            click.secho(
                "Adversarial Patch Attack summary: {}".format(results),
                fg="green",
            )
            logger.info("Object Detection Attack summary: %s", results)
        return adversarial_images if self._config.return_adversarial_images else None
