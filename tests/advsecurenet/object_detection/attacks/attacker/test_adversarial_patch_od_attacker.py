import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from torch.utils.data import Dataset, DataLoader

from advsecurenet.computer_vision.object_detection.attacks.attacker.adversarial_patch_od_attacker import (
    AdversarialPatchODAttacker,
)
from advsecurenet.shared.types.configs.attack_configs.od_attacker_config import (
    ODAttackerConfig,
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


# Dummy attack with .attack() and .apply_patch()
class DummyAttack:
    def attack(self, mask, dataloader, device):
        return torch.ones((3, 10, 10))

    def apply_patch(self, x, patch_external, random_location):
        return torch.from_numpy(x / 255.0)


@pytest.fixture
def patch_config():
    config = ODAttackerConfig(
        model=MagicMock(),
        attack=DummyAttack(),
        dataloader=DataLoader(DummyDataset()),
        device=MagicMock(processor="cpu", use_ddp=False),
        return_adversarial_images=True,
        evaluators=["mean_average_precision"],
    )
    config.model.to.return_value = config.model
    config.model.eval.return_value = config.model
    return config


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.adversarial_patch_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_execute_return_adv_images(mock_evaluator, patch_config):
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    attacker = AdversarialPatchODAttacker(patch_config)
    adv_images = attacker.execute()
    assert isinstance(adv_images, list)
    assert all(isinstance(img, torch.Tensor) for img in adv_images)
    evaluator_instance.update.assert_called()
    evaluator_instance.get_results.assert_called()


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.adversarial_patch_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_execute_dont_return_adv_images(mock_evaluator, patch_config):
    patch_config.return_adversarial_images = False
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    attacker = AdversarialPatchODAttacker(patch_config)
    adv_images = attacker.execute()
    assert adv_images is None


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.adversarial_patch_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_execute_empty_dataloader(mock_evaluator, patch_config):
    patch_config.dataloader = DataLoader(DummyDataset(n=0))
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    attacker = AdversarialPatchODAttacker(patch_config)
    adv_images = attacker.execute()
    assert isinstance(adv_images, list)
    assert adv_images == []


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.adversarial_patch_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_patch_application_shape(mock_evaluator, patch_config):
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    attacker = AdversarialPatchODAttacker(patch_config)
    adv_images = attacker.execute()
    for img in adv_images:
        assert img.shape[-3:] == (
            3,
            10,
            10,
        )  # Accept any batch shape, but last three dims must match
