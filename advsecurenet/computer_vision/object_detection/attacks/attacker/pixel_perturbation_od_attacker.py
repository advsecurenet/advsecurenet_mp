# advsecurenet/computer_vision/object_detection/attacks/attacker/default_od_attacker.py

import logging
import numpy as np
import torch
import click
from tqdm.auto import tqdm

from advsecurenet.evaluation.od_adversarial_evaluator import (
    ObjectDetectorAdversarialEvaluator,
)
from advsecurenet.shared.types.configs.attack_configs.od_attacker_config import (
    ODAttackerConfig,
)
from advsecurenet.computer_vision.object_detection.attacks.attacker.od_attacker import (
    ODAttacker,
)
from advsecurenet.computer_vision.object_detection.attacks.pixel_perturbation_based.tog.tog_attack_type import (
    TOGAttackType,
)
from advsecurenet.utils.device_utils import move_batch_to_device

logger = logging.getLogger(__name__)


class PixelPerturbationODAttacker(ODAttacker):
    """
    Run pixel-perturbation attacks (e.g., TOG) against an object detector.

    This attacker orchestrates per-batch preprocessing, invokes the configured
    pixel-perturbation attack to generate adversarial images, and evaluates
    them with the OD adversarial evaluator.

    Args:
        config (ODAttackerConfig): Runtime configuration containing the object
            detector to evaluate, dataloader, evaluators, and the attack instance.
        attack_type (TOGAttackType, optional): TOG variant to apply. One of
            TOGAttackType.VANISHING, TOGAttackType.FABRICATION,
            TOGAttackType.MISLABELING, TOGAttackType.UNTARGETED.
            Defaults to TOGAttackType.VANISHING.
        tog_mislabeling_mode (str, optional): Mode for TOG mislabeling when
            attack_type is MISLABELING. One of {"ml", "ll"} where:
            - "ml": most-likely non-original class,
            - "ll": least-likely class.
            Defaults to "ml".
    """

    def __init__(
        self,
        config: ODAttackerConfig,
        attack_type: TOGAttackType = TOGAttackType.VANISHING,
        tog_mislabeling_mode: str = "ml",
    ) -> None:
        super().__init__(config)
        self._attack_type = attack_type
        self._tog_mislabeling_mode = tog_mislabeling_mode

    def _perturb_images(
        self,
        images_np_for_tog: np.ndarray,
    ) -> np.ndarray:
        adv_imgs = self._config.attack.attack(
            x=images_np_for_tog,
            tog_variant=self._attack_type,
            tog_mislabeling_mode=self._tog_mislabeling_mode or "ml",
        )
        return torch.from_numpy(adv_imgs).to(self._device)

    def execute(self):
        adversarial_images = []
        with ObjectDetectorAdversarialEvaluator(
            evaluators=self._config.evaluators,
            target_models=[self._eval_model],  # we evaluate on the eval_model
        ) as evaluator:
            for data_batch in tqdm(
                self._dataloader,
                desc="Generating adversarial samples",
                colour="red",
            ):
                images_np_for_tog, targets, images = self.process_batch(data_batch)
                modified_imgs = self._perturb_images(images_np_for_tog)
                # 2) EVALUATE on the patched images
                evaluator.update(
                    model=self._eval_model,
                    original_images=images,
                    adversarial_images=modified_imgs,
                    targets=targets,
                )
                if self._config.return_adversarial_images:
                    adversarial_images.append(modified_imgs.detach().cpu())
                    # adversarial_images.append(images.detach().cpu())
                # free up GPU memory
                if torch.cuda.is_available() and self._device.type == "cuda":
                    torch.cuda.empty_cache()
            # summary logging
            results = evaluator.get_results()
            click.secho(
                "Adversarial Patch Attack summary: {}".format(results),
                fg="green",
            )
            logger.info("Object Detection Attack summary: %s", results)
        return adversarial_images if self._config.return_adversarial_images else None
