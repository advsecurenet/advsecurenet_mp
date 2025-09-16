# advsecurenet/computer_vision/object_detection/attacks/attacker/default_od_attacker.py

import logging
import numpy as np
import torch
import click
from advsecurenet.evaluation.od_adversarial_evaluator import (
    ObjectDetectorAdversarialEvaluator,
)
from advsecurenet.shared.types.configs.attack_configs.od_attacker_config import (
    ODAttackerConfig,
)
from advsecurenet.computer_vision.object_detection.attacks.attacker.od_attacker import (
    ODAttacker,
)
from advsecurenet.utils.device_utils import move_batch_to_device

logger = logging.getLogger(__name__)


class AdversarialPatchODAttacker(ODAttacker):
    """
    Attacks an object detection model using an adversarial patch.
    """

    def __init__(self, config: ODAttackerConfig):
        super().__init__(config)
        self._trained_patch = None

    def _apply_patch(
        self, images_np_for_dpatch: np.ndarray, patch_np: np.ndarray
    ) -> torch.Tensor:
        """Apply the trained patch to a batch (expects [0,255] np, returns torch float32 [0,1])."""
        patched_np = (
            self._config.attack.apply_patch(
                x=images_np_for_dpatch,
                patch_external=patch_np,
                random_location=False,
            )
            .detach()
            .cpu()
            .numpy()
        )
        return torch.from_numpy(patched_np).to(self._device).float().div_(255.0)

    def execute(self):
        adversarial_images = []
        with ObjectDetectorAdversarialEvaluator(
            evaluators=self._config.evaluators,
            target_models=[
                self._config.attack._object_detector.inference_model
            ],  # we evaluate on the eval_model
        ) as evaluator:
            logger.info("Starting adversarial patch training and evaluation")
            self._trained_patch = self._config.attack.attack(
                mask=getattr(self._config.attack, "mask", None),
                dataloader=self._dataloader,
                device=self._device,
            )
            try:
                logger.info(
                    "Patch trained: shape=%s, min=%.2f, max=%.2f",
                    tuple(self._trained_patch.shape),
                    float(self._trained_patch.min().item()),
                    float(self._trained_patch.max().item()),
                )
            except Exception:
                logger.info("Patch trained: stats unavailable")
            for data_batch in self._dataloader:
                images_preprocessed, targets, images = self.process_batch(data_batch)
                patched = self._apply_patch(
                    images_preprocessed, self._trained_patch.detach().cpu().numpy()
                )
                logger.debug(
                    "Applied patch to batch: patched_shape=%s, targets=%d",
                    tuple(patched.shape) if hasattr(patched, "shape") else "unknown",
                    len(targets),
                )
                evaluator.update(
                    model=self._config.attack._object_detector.inference_model,
                    original_images=images,
                    adversarial_images=patched,
                    targets=targets,
                )
                logger.debug("Evaluator updated for current batch")
                if self._config.return_adversarial_images:
                    adversarial_images.append(patched.detach().cpu())
                # free up GPU memory
                if torch.cuda.is_available() and self._device.type == "cuda":
                    torch.cuda.empty_cache()
            results = evaluator.get_results()
            click.secho(
                "Adversarial Patch Attack summary: {}".format(results),
                fg="green",
            )
            logger.info("Object Detection Attack summary: %s", results)
        return adversarial_images if self._config.return_adversarial_images else None
