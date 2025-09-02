import logging
from typing import Optional, Union

import click
import torch
from torch.utils.data import Subset, random_split

from advsecurenet.computer_vision.object_detection.attacks.attacker.adversarial_patch_od_attacker import (
    AdversarialPatchODAttacker,
)
from advsecurenet.computer_vision.object_detection.attacks.attacker.pixel_perturbation_od_attacker import (
    PixelPerturbationODAttacker,
)
from advsecurenet.computer_vision.object_detection.attacks.attacker.od_attacker import (
    ODAttackerConfig,
)
from cli.shared.types.attack import BaseAttackCLIConfigType
from advsecurenet.distributed.ddp_coordinator import DDPCoordinator
from advsecurenet.utils.ddp import set_visible_gpus
from advsecurenet.computer_vision.object_detection.attacks.attacker.ddp_od_attacker import DDPODAttacker
from advsecurenet.dataloader.data_loader_factory import od_collate_fn, DataLoaderFactory
from advsecurenet.shared.types.configs.dataloader_config import DataLoaderConfig
from cli.shared.utils.dataset import get_datasets
from cli.shared.utils.helpers import save_images
from cli.shared.utils.model import create_model
from advsecurenet.computer_vision.object_detection.attacks.adversarial_patch_based.dpatch import (
    DPatch,
)
from advsecurenet.computer_vision.object_detection.attacks.pixel_perturbation_based.tog.tog import (
    TOG,
)
from advsecurenet.computer_vision.object_detection.attacks.pixel_perturbation_based.tog.tog_attack_type import (
    TOGAttackType,
)
from advsecurenet.models.detector_factory import get_object_detector
from advsecurenet.models.CustomModels.CustomYolov5Model import CustomYolov5Model

logger = logging.getLogger(__name__)


class CLIODAttacker:
    """
    Attacker class for the CLI for object detection. This module parses the CLI arguments and executes the OD attack.
    """

    def __init__(self, config: BaseAttackCLIConfigType, od_main_attack_type, **kwargs):
        self._config = config
        self.od_main_attack_type = od_main_attack_type
        self._kwargs = kwargs
        self._dataset = self._prepare_dataset()

    def execute(self):
        logger.info(
            "Starting %s attack (object detection).", self.od_main_attack_type.name
            )
        if getattr(self._config.device, "use_ddp", False):
            logger.info("Using DDP for attack with GPUs: %s", self._config.device.gpu_ids)
            self._execute_ddp_attack()
        else:
            self._execute_attack()
        click.secho("Attack completed successfully.", fg="green")
        logger.info("%s attack completed successfully.", self.od_main_attack_type.name)

    def _execute_ddp_attack(self):
        if not self._config.device.gpu_ids or len(self._config.device.gpu_ids) == 0:
            self._config.device.gpu_ids = list(range(torch.cuda.device_count()))
        world_size = len(self._config.device.gpu_ids)
        set_visible_gpus(self._config.device.gpu_ids)
        ddp_attacker = DDPCoordinator(self._ddp_attack_fn, world_size)
        ddp_attacker.run()
        # Only rank0 process collects results (after spawn join) – gather stored per-rank files
        if self._config.attack_procedure.save_result_images:
            try:
                adv_imgs = DDPODAttacker.gather_results(world_size)
                if adv_imgs:
                    self._save_images_if_needed(adv_imgs)
            except Exception as e:
                logger.error("Failed to gather DDP OD adversarial images: %s", e)

    def _ddp_attack_fn(self, rank: int, world_size: int) -> None:
        try:
            if getattr(self._config.device, "gpu_ids", None):
                torch.cuda.set_device(rank)
                self._config.device.processor = f"cuda:{rank}"
            else:
                # Fallback: ensure processor string reflects local rank even if gpu_ids absent
                torch.cuda.set_device(rank)
                self._config.device.processor = f"cuda:{rank}"
            logger.info("[DDP OD] Rank %d using device %s (physical GPU %s)", rank, self._config.device.processor, getattr(self._config.device, 'gpu_ids', [None])[rank] if getattr(self._config.device, 'gpu_ids', None) else rank)
        except Exception as e:
            logger.error("[DDP OD] Failed to set device for rank %d: %s", rank, e)
        config, extra_kwargs = self._prepare_attack_config()
        ddp_wrapper = DDPODAttacker(attacker_class=self._build_concrete_attacker_class(), config=config, **extra_kwargs)
        ddp_wrapper.setup()
        ddp_wrapper.run_task()

    def _build_concrete_attacker_class(self):
        if self.od_main_attack_type.name.upper() == "DPATCH":
            return AdversarialPatchODAttacker
        elif self.od_main_attack_type.name.upper() == "TOG":
            return PixelPerturbationODAttacker
        else:
            raise ValueError(f"Unknown attack type: {self.od_main_attack_type}")

    def _execute_attack(self):
        config, extra_kwargs = self._prepare_attack_config()
        if self.od_main_attack_type.name.upper() == "DPATCH":
            attacker = AdversarialPatchODAttacker(config=config)
        elif self.od_main_attack_type.name.upper() == "TOG":
            attacker = PixelPerturbationODAttacker(
                config=config,
                attack_type=extra_kwargs.get("attack_type", TOGAttackType.VANISHING),
                tog_mislabeling_mode=extra_kwargs.get("tog_mislabeling_mode", "ml"),
            )
        else:
            raise ValueError(f"Unknown attack type: {self.od_main_attack_type}")
        adv_imgs = attacker.execute()
        self._save_images_if_needed(adv_imgs)

    def _save_images_if_needed(self, adv_imgs: Optional[list] = None):
        if self._config.attack_procedure.save_result_images and adv_imgs:
            logger.info("Saving adversarial images.")
            save_images(
                images=adv_imgs,
                path=self._config.attack_procedure.result_images_dir or "results",
                prefix=self._config.attack_procedure.result_images_prefix or "adv",
            )
            logger.info("Adversarial images saved successfully.")
        else:
            logger.info("No adversarial images to save.")

    def _prepare_attack_config(self):
        model = create_model(self._config.model)
        dataloader_config = self._create_dataloader_config()
        attack_config = self._config.attack_config.attack_parameters
        # Extract object_detector_config from model config if present
        detector_config = {}
        if hasattr(self._config.model, "object_detector_config"):
            detector_config = self._config.model.object_detector_config
        detector = get_object_detector(attack_config.object_detector, detector_config)
        attack_config.object_detector = detector
        attack_config.device = self._config.device

        extra_kwargs = {}
        if self.od_main_attack_type.name.upper() == "DPATCH":
            # Set verbose from attack_procedure
            setattr(attack_config, "verbose", self._config.attack_procedure.verbose)
            attack = DPatch(attack_config)
        elif self.od_main_attack_type.name.upper() == "TOG":
            # Set verbose from attack_procedure
            setattr(attack_config, "verbose", self._config.attack_procedure.verbose)
            attack = TOG(attack_config)
            # Map string to TOGAttackType enum
            attack_type_str = getattr(attack_config, "attack_type", "vanishing")
            tog_attack_type = TOGAttackType(attack_type_str.lower())
            mislabeling_mode = getattr(attack_config, "mislabeling_mode", "ml")
            extra_kwargs["attack_type"] = tog_attack_type
            extra_kwargs["tog_mislabeling_mode"] = mislabeling_mode
        else:
            raise ValueError(f"Unknown attack type: {self.od_main_attack_type}")

        config = ODAttackerConfig(
            model=model,
            dataloader=dataloader_config,
            device=self._config.device,
            attack=attack,
            return_adversarial_images=self._config.attack_procedure.save_result_images,
            evaluators=self._get_evaluators(),
        )
        return config, extra_kwargs

    def _create_dataloader_config(self):
        dataloader_config = DataLoaderConfig(
            dataset=self._dataset,
            batch_size=self._config.dataloader.default.batch_size,
            num_workers=self._config.dataloader.default.num_workers,
            shuffle=self._config.dataloader.default.shuffle,
            drop_last=self._config.dataloader.default.drop_last,
            pin_memory=self._config.dataloader.default.pin_memory,
            sampler=None,
        )
        if self._is_coco_dataset():
            setattr(dataloader_config, "collate_fn", od_collate_fn)
        return dataloader_config

    def _is_coco_dataset(self):
        return getattr(self._config.dataset, "dataset_name", "").upper() == "COCO"

    def _prepare_dataset(self):
        train_data, test_data = get_datasets(self._config.dataset)
        load_splits = self._config.dataset.load_splits
        if load_splits == ["train"]:
            data = train_data
        elif load_splits == ["test"]:
            data = test_data
        else:
            data = (
                train_data + test_data
                if train_data and test_data
                else train_data or test_data
            )

        data = self._sample_data_if_required(data)
        return data

    def _sample_data_if_required(self, all_data):
        """
        Sample data from the dataset if random_sample_size is specified in the config.
        """
        sample_size = self._config.dataset.random_sample_size
        if sample_size is not None and sample_size > 0:
            logger.info("Sampling %d data points from the dataset.", sample_size)
            return self._sample_data(all_data, sample_size)
        return all_data

    def _sample_data(self, data, sample_size):
        """
        Sample data from the dataset.

        Args:
            data (torch.utils.data.Dataset): The dataset.
            sample_size (int): The sample size.

        Returns:
            torch.utils.data.Subset: The sampled data.
        """
        if len(data) < sample_size:
            logger.warning(
                "The dataset size (%d) is smaller than the requested sample size (%d). Using the entire dataset.",
                len(data),
                sample_size,
            )
            sample_size = len(data)

        random_samples = min(sample_size, len(data))
        lengths = [random_samples, len(data) - random_samples]
        subset, _ = random_split(data, lengths)
        random_data = Subset(data, subset.indices)
        return random_data

    def _get_evaluators(self):
        """
        Get the evaluators for the attack from the CLI parameters.

        Returns:
            list[str]: List of evaluator names.
        """
        # Get evaluators from CLI kwargs (same as image classification attacks)
        return self._kwargs.get("evaluators", ["mean_average_precision"])
