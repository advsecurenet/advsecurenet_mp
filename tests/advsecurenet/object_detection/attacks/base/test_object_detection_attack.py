import pytest
from unittest.mock import MagicMock
import torch

from advsecurenet.computer_vision.object_detection.attacks.base.object_detection_attack import (
    ObjectDetectionAttack,
)
from advsecurenet.shared.types.configs.attack_configs.attack_config import AttackConfig


# Dummy config for testing
class DummyConfig(AttackConfig):
    def __init__(self):
        super().__init__(
            device=MagicMock(processor="cpu", use_ddp=False), targeted=True
        )


# Minimal concrete subclass for testing
class DummyObjectDetectionAttack(ObjectDetectionAttack):
    def attack(self, model, x, y, *args, **kwargs):
        return x  # Just return input for test


def test_instantiation_sets_attributes():
    config = DummyConfig()
    attack = DummyObjectDetectionAttack(config)
    assert hasattr(attack, "device_manager")
    assert hasattr(attack, "name")
    assert hasattr(attack, "targeted")
    assert attack.name == "DummyObjectDetectionAttack"
    assert attack.targeted is True


def test_attack_method_runs():
    config = DummyConfig()
    attack = DummyObjectDetectionAttack(config)
    x = torch.zeros((2, 3, 10, 10))
    y = torch.zeros((2, 1))
    out = attack.attack(None, x, y)
    assert torch.equal(out, x)


def test_abstract_attack_enforced():
    config = DummyConfig()
    # Direct instantiation of ObjectDetectionAttack should fail
    with pytest.raises(TypeError):
        ObjectDetectionAttack(config)
