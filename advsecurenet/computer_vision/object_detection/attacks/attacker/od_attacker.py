import logging
import torch
from tqdm.auto import tqdm

from advsecurenet.evaluation.adversarial_evaluator import AdversarialEvaluator
from advsecurenet.dataloader import DataLoaderFactory
from advsecurenet.shared.types.configs.attack_configs.od_attacker_config import ODAttackerConfig

logger = logging.getLogger(__name__)

class ODAttacker:
    """
    Orchestrates an object-detection attack where:
      - `config.attack.object_detector` is used to *generate* the patch
      - `config.model` is used to *detect* on the patched images
    """
    def __init__(self, config: ODAttackerConfig):
        self._config       = config
        self._device       = self._setup_device()
        # this model is only used inside your attack.attack(...) for gradients
        self._attack_model = config.attack.object_detector.to(self._device)
        # this model is used for final inference/evaluation on patched images
        self._eval_model   = config.model.to(self._device).eval()
        self._dataloader   = self._create_dataloader()

    def _setup_device(self):
        if self._config.device.processor:
            return torch.device(self._config.device.processor)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu") # replace with setup device utility function - TODO

    def _create_dataloader(self):
        dl = self._config.dataloader
        if isinstance(dl, torch.utils.data.DataLoader):
            return dl
        return DataLoaderFactory.create_dataloader(dl)

    def execute(self):
        adversarial_images = []

        with AdversarialEvaluator(
            evaluators    = self._config.evaluators,
            target_models = [self._eval_model],      # we evaluate on the eval_model
        ) as evaluator:

            for images, boxes, labels in tqdm(
                self._dataloader,
                desc="Generating adversarial samples",
                colour="red",
            ):
                # move to device
                images = images.to(self._device)
                boxes  = [b.to(self._device) for b in boxes]
                labels = [l.to(self._device) for l in labels]

                # 1) GENERATE the patch (no real images returned here)
                my_target_label = self._config.target_label
                print("Target label for attack: ", my_target_label)
                learned_patch = self._config.attack.attack(
                    model        = self._attack_model,
                    x            = images,
                    y            = {"boxes": boxes, "labels": labels},
                    target_label = my_target_label,#getattr(self._config.attack, "target_label", None),
                    mask         = getattr(self._config.attack, "mask", None),
                )

                # 2) APPLY the patch to *this* batch of images
                patched_np = self._config.attack.apply_patch(
                    x               = images.detach().cpu().numpy(),
                    patch_external  = learned_patch.detach().cpu().numpy(),
                    random_location = False
                )
                patched = patched_np.to(self._device)#torch.from_numpy(patched_np).to(self._device)

                #3) EVALUATE on the patched images
                evaluator.update(
                    model              = self._eval_model,
                    original_images    = images,
                    true_labels        = {"boxes": boxes, "labels": labels},
                    adversarial_images = patched,
                    is_targeted        = self._config.attack.targeted,
                    target_labels      = getattr(self._config.attack, "target_label", None),
                )

                if self._config.return_adversarial_images:
                    #adversarial_images.append(patched.detach().cpu())
                    adversarial_images.append(images.detach().cpu())

                # free up GPU memory if needed
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # summary logging
            results = evaluator.get_results()
            logger.info("Object Detection Attack summary: %s", results)

        return adversarial_images if self._config.return_adversarial_images else None
