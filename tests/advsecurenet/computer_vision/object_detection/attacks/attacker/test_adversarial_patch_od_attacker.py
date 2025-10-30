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
    attack = DummyAttack()
    # Attach minimal object detector with an inference_model as required by attacker
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


# Additional tests for 100% coverage
@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.adversarial_patch_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_execute_with_cuda_available(mock_evaluator, patch_config, monkeypatch):
    """Test execute with CUDA available."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    patch_config.device.type = "cuda"
    
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    
    attacker = AdversarialPatchODAttacker(patch_config)
    adv_images = attacker.execute()
    
    assert isinstance(adv_images, list)


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.adversarial_patch_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_apply_patch_method(mock_evaluator, patch_config):
    """Test _apply_patch method."""
    attacker = AdversarialPatchODAttacker(patch_config)
    
    images_np = np.ones((2, 3, 10, 10), dtype=np.float32) * 128
    patch_np = np.ones((3, 10, 10), dtype=np.float32) * 200
    
    result = attacker._apply_patch(images_np, patch_np)
    
    assert isinstance(result, torch.Tensor)
    assert result.dtype == torch.float32
    assert torch.all(result >= 0.0) and torch.all(result <= 1.0)


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.adversarial_patch_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_execute_patch_stats_logging(mock_evaluator, patch_config, caplog):
    """Test that patch stats are logged."""
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    
    attacker = AdversarialPatchODAttacker(patch_config)
    adv_images = attacker.execute()
    
    assert "Patch trained" in caplog.text


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.adversarial_patch_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_execute_patch_stats_exception(mock_evaluator, patch_config, caplog):
    """Test patch stats logging when exception occurs."""
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    
    # Make patch without min/max/item methods
    class BadPatch:
        shape = (3, 10, 10)
        def min(self):
            raise AttributeError("no min")
        def max(self):
            raise AttributeError("no max")
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return np.ones((3, 10, 10), dtype=np.float32)
    
    patch_config.attack.attack = lambda **kwargs: BadPatch()
    
    attacker = AdversarialPatchODAttacker(patch_config)
    adv_images = attacker.execute()
    
    assert "stats unavailable" in caplog.text


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.adversarial_patch_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_execute_debug_logging(mock_evaluator, patch_config, caplog):
    """Test debug logging during execution."""
    import logging
    caplog.set_level(logging.DEBUG)
    
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    
    attacker = AdversarialPatchODAttacker(patch_config)
    adv_images = attacker.execute()
    
    assert "Applied patch to batch" in caplog.text
    assert "Evaluator updated" in caplog.text


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.adversarial_patch_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_execute_with_mask_attribute(mock_evaluator, patch_config):
    """Test execute when attack has mask attribute."""
    patch_config.attack.mask = np.ones((10, 10), dtype=bool)
    
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    
    attacker = AdversarialPatchODAttacker(patch_config)
    adv_images = attacker.execute()
    
    assert isinstance(adv_images, list)


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.adversarial_patch_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_execute_without_mask_attribute(mock_evaluator, patch_config):
    """Test execute when attack doesn't have mask attribute."""
    # Remove mask attribute if it exists
    if hasattr(patch_config.attack, 'mask'):
        delattr(patch_config.attack, 'mask')
    
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    
    attacker = AdversarialPatchODAttacker(patch_config)
    adv_images = attacker.execute()
    
    assert isinstance(adv_images, list)


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.adversarial_patch_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_trained_patch_initialization(mock_evaluator, patch_config):
    """Test that _trained_patch is initialized to None."""
    attacker = AdversarialPatchODAttacker(patch_config)
    assert attacker._trained_patch is None
    
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    
    attacker.execute()
    assert attacker._trained_patch is not None


@patch(
    "advsecurenet.computer_vision.object_detection.attacks.attacker.adversarial_patch_od_attacker.ObjectDetectorAdversarialEvaluator"
)
def test_execute_info_logging(mock_evaluator, patch_config, caplog):
    """Test info level logging."""
    import logging
    caplog.set_level(logging.INFO)
    
    evaluator_instance = MagicMock()
    mock_evaluator.return_value.__enter__.return_value = evaluator_instance
    evaluator_instance.get_results.return_value = {"mAP": 0.5}
    
    attacker = AdversarialPatchODAttacker(patch_config)
    adv_images = attacker.execute()
    
    assert "Starting adversarial patch training" in caplog.text
    assert "Object Detection Attack summary" in caplog.text
