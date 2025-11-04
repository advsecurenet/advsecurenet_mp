import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from advsecurenet.computer_vision.base.adversarial_attack import AdversarialAttack
from advsecurenet.computer_vision.base.base_adversarial_training import (
    BaseAdversarialTraining,
)
from advsecurenet.models.base_model import BaseModel


class MockModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.model_name = "mock_model"
        self._train_called = 0
        self._to_called_with = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return x

    def load_model(self) -> None:  # satisfy abstract method
        return None

    def models(self):  # satisfy abstract method expected by BaseModel
        return [self]

    def train(self, mode: bool = True):  # type: ignore[override]
        self._train_called += 1
        return self

    def to(self, device):  # type: ignore[override]
        self._to_called_with = device
        return self


class MockAttack(AdversarialAttack):
    def __init__(self):
        # do not call super().__init__ to avoid requiring AttackConfig in tests
        self.name = "mock"
        self.targeted = False

    def attack(self, model: BaseModel, x: torch.Tensor, y: torch.Tensor, *args, **kwargs):  # type: ignore[override]
        return x

    __test__ = False


def _make_dataloader(num_samples: int = 3) -> DataLoader:
    x = torch.randn(num_samples, 3, 4, 4)
    y = torch.zeros(num_samples, dtype=torch.long)
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=1, shuffle=False)


@pytest.fixture
def valid_config():
    return SimpleNamespace(
        model=MockModel(),
        models=[],
        attacks=[MockAttack()],
        train_loader=_make_dataloader(4),
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_config_base_valid(valid_config):
    bat = object.__new__(BaseAdversarialTraining)
    bat.config = valid_config  # type: ignore[attr-defined]
    # Should not raise
    bat._check_config_base(valid_config)


@pytest.mark.advsecurenet
@pytest.mark.essential
@pytest.mark.parametrize(
    "bad_cfg,err_msg",
    [
        (
            SimpleNamespace(
                model=object(),
                models=[],
                attacks=[MockAttack()],
                train_loader=_make_dataloader(),
            ),
            "Target model must be a subclass of BaseModel!",
        ),
        (
            SimpleNamespace(
                model=MockModel(),
                models=[object()],
                attacks=[MockAttack()],
                train_loader=_make_dataloader(),
            ),
            "All models must be a subclass of BaseModel!",
        ),
        (
            SimpleNamespace(
                model=MockModel(),
                models=[MockModel()],
                attacks=[object()],
                train_loader=_make_dataloader(),
            ),
            "All attacks must be a subclass of AdversarialAttack!",
        ),
        (
            SimpleNamespace(
                model=MockModel(),
                models=[MockModel()],
                attacks=[MockAttack()],
                train_loader=object(),
            ),
            "train_dataloader must be a DataLoader!",
        ),
    ],
)
def test_check_config_base_invalid(bad_cfg, err_msg):
    bat = object.__new__(BaseAdversarialTraining)
    with pytest.raises(ValueError, match=err_msg):
        bat._check_config_base(bad_cfg)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_pre_training_adds_model_and_sets_train_and_device(valid_config):
    bat = object.__new__(BaseAdversarialTraining)
    bat.config = valid_config  # type: ignore[attr-defined]
    bat._device = torch.device("cpu")

    # Initially, target model not in models list
    assert valid_config.model not in valid_config.models

    bat._pre_training()

    # Target model appended exactly once
    assert valid_config.models.count(valid_config.model) == 1

    # train() called on each model
    assert all(
        isinstance(m, MockModel) and m._train_called >= 1 for m in valid_config.models
    )

    # to(device) called on each model
    assert all(
        isinstance(m, MockModel) and m._to_called_with == bat._device
        for m in valid_config.models
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_pre_training_no_duplicate_when_model_already_present(valid_config):
    # Pre-populate with the same target model
    valid_config.models = [valid_config.model]

    bat = object.__new__(BaseAdversarialTraining)
    bat.config = valid_config  # type: ignore[attr-defined]
    bat._device = torch.device("cpu")

    bat._pre_training()

    # Still only one instance
    assert valid_config.models.count(valid_config.model) == 1


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_train_loader_wraps_dataloader(valid_config):
    bat = object.__new__(BaseAdversarialTraining)
    bat.config = valid_config  # type: ignore[attr-defined]

    wrapped = bat._get_train_loader(epoch=1)

    # Consuming the iterator should yield the same number of batches as the dataloader
    assert sum(1 for _ in wrapped) == len(valid_config.train_loader)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_loss_divisor(valid_config):
    bat = object.__new__(BaseAdversarialTraining)
    bat.config = valid_config  # type: ignore[attr-defined]

    assert bat._get_loss_divisor() == len(valid_config.train_loader)
