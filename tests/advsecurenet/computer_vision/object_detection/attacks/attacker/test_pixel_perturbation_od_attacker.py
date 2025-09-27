import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from torch.utils.data import Dataset, DataLoader

from advsecurenet.computer_vision.object_detection.attacks.attacker.pixel_perturbation_od_attacker import (
    PixelPerturbationODAttacker,
)
from advsecurenet.shared.types.configs.attack_configs.od_attacker_config import (
    ODAttackerConfig,
)
from advsecurenet.computer_vision.object_detection.attacks.pixel_perturbation_based.tog.tog_attack_type import (
    TOGAttackType,
)


# Minimal dummy dataset
class DummyDataset(Dataset):
    def __init__(self, n=2):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        images = torch.zeros((3, 10, 10))
        targets_dict = {
            "boxes": torch.zeros((2, 4)),
            "labels": torch.zeros((2,)),
        }
        return images, targets_dict


# Dummy attack with .attack()
class DummyPixelAttack:
    def attack(self, x, y=None, mask=None, tog_variant=None, tog_mislabeling_mode=None):
        return np.ones_like(x)


@pytest.fixture
def pixel_config():
    # Create the minimal attack and attach a minimal _object_detector with an inference_model
    attack = DummyPixelAttack()
    attack._object_detector = MagicMock()
    attack._object_detector.inference_model = MagicMock(name="inference_model")

    config = ODAttackerConfig(
        model=MagicMock(),
        attack=attack,
        dataloader=DataLoader(DummyDataset()),
        device=MagicMock(processor="cpu", use_ddp=False),
        return_adversarial_images=True,
        evaluators=["mean_average_precision"],
    )
    config.model.to.return_value = config.model
    config.model.eval.return_value = config.model
    return config


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.pixel_perturbation_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_execute_return_adv_images(mock_evaluator, pixel_config):
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    attacker = PixelPerturbationODAttacker(
        pixel_config, attack_type=TOGAttackType.VANISHING
    )
    adv_images = attacker.execute()
    assert isinstance(adv_images, list)
    assert all(isinstance(img, torch.Tensor) for img in adv_images)
    evaluator_instance.update.assert_called()
    evaluator_instance.get_results.assert_called()


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.pixel_perturbation_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_execute_dont_return_adv_images(mock_evaluator, pixel_config):
    pixel_config.return_adversarial_images = False
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    attacker = PixelPerturbationODAttacker(
        pixel_config, attack_type=TOGAttackType.FABRICATION
    )
    adv_images = attacker.execute()
    assert adv_images is None


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.pixel_perturbation_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_execute_empty_dataloader(mock_evaluator, pixel_config):
    pixel_config.dataloader = DataLoader(DummyDataset(n=0))
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    attacker = PixelPerturbationODAttacker(
        pixel_config, attack_type=TOGAttackType.UNTARGETED
    )
    adv_images = attacker.execute()
    assert isinstance(adv_images, list)
    assert adv_images == []


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.pixel_perturbation_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_patch_application_shape(mock_evaluator, pixel_config):
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    attacker = PixelPerturbationODAttacker(
        pixel_config, attack_type=TOGAttackType.VANISHING
    )
    adv_images = attacker.execute()
    for img in adv_images:
        assert img.shape[-3:] == (
            3,
            10,
            10,
        )  # Accept any batch shape, but last three dims must match
