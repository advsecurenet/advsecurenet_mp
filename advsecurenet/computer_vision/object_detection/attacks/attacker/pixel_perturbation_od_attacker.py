# advsecurenet/computer_vision/object_detection/attacks/attacker/default_od_attacker.py

import logging
import numpy as np
import torch
from tqdm.auto import tqdm

from advsecurenet.evaluation.adversarial_evaluator import AdversarialEvaluator
from advsecurenet.shared.types.configs.attack_configs.od_attacker_config import ODAttackerConfig
from advsecurenet.computer_vision.object_detection.attacks.attacker.od_attacker import ODAttacker
from advsecurenet.computer_vision.object_detection.attacks.pixel_perturbation_based.tog import TOGAttackType

logger = logging.getLogger(__name__)

class PixelPerturbationODAttacker(ODAttacker):
    """
    Attacks an object detection model using pixel perturbation techniques.
    """
    def __init__(self, config: ODAttackerConfig, attack_type: TOGAttackType = TOGAttackType.VANISHING, tog_mislabeling_mode: str = "ml") -> None:
        super().__init__(config)
        self._attack_type = attack_type
        self._tog_mislabeling_mode = tog_mislabeling_mode

    def execute(self):
        adversarial_images = []
        with AdversarialEvaluator(
            evaluators    = self._config.evaluators,
            target_models = [self._eval_model],      # we evaluate on the eval_model
        ) as evaluator:
            for data_batch in tqdm(
                self._dataloader,
                desc="Generating adversarial samples",
                colour="red",
            ):
                images, targets_dict = data_batch
                images_np_for_dpatch = (images.detach().cpu().numpy() * 255.0).astype(np.float32)
                images_np_for_dpatch = np.clip(images_np_for_dpatch, 0, 255)
                boxes  = [b.to(self._device) for b in targets_dict["boxes"]]
                labels = [l.to(self._device) for l in targets_dict["labels"]]
                # 1) GENERATE the patch (no real images returned here)
                my_target_label = None#self._config.target_label
                print("Target label for attack: ", my_target_label)
                targets = []
                for b, l in zip(boxes, labels):
                    raw = l.detach().cpu().numpy().astype(int)      # e.g. [1, 3, 18, …]
                    mapped = np.array(raw, dtype=int)
                    targets.append({
                        "boxes":  b.detach().cpu().numpy(),
                        "labels": mapped,
                        "scores": np.ones(len(mapped), dtype=float),
                    })
                adv_imgs = self._config.attack.attack(
                    x            = images_np_for_dpatch,
                    y            = targets,
                    target_label = my_target_label,
                    mask         = getattr(self._config.attack, "mask", None),
                    tog_variant  = self._attack_type,
                    tog_mislabeling_mode = self._tog_mislabeling_mode or "ml",
                )
                patched = torch.from_numpy(adv_imgs).to(self._device)
                # 2) EVALUATE on the patched images
                # evaluator.update(
                #     model              = self._eval_model,
                #     original_images    = images,
                #     true_labels        = {"boxes": boxes, "labels": labels},
                #     adversarial_images = patched,
                #     is_targeted        = self._config.attack.targeted,
                #     target_labels      = getattr(self._config.attack, "target_label", None),
                # )
                if self._config.return_adversarial_images:
                    adversarial_images.append(patched.detach().cpu())
                    #adversarial_images.append(images.detach().cpu())
                # free up GPU memory if needed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            # summary logging
            # results = evaluator.get_results()
            # logger.info("Object Detection Attack summary: %s", results)
        return adversarial_images if self._config.return_adversarial_images else None
