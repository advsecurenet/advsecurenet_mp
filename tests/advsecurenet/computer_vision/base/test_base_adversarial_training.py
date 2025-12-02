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
    """Provides a valid, nested config for BaseAdversarialTraining."""
    model = MockModel()
    return SimpleNamespace(
        models=[model],
        attacks=[MockAttack()],
        train_config=SimpleNamespace(
            model_config=SimpleNamespace(model=model),
            training_process_config=SimpleNamespace(
                train_loader=_make_dataloader(4),
            ),
            device_config=SimpleNamespace(processor="cpu"),
        ),
    )


@pytest.fixture
def base_instance(valid_config):
    """Provides a partially initialized BaseAdversarialTraining instance."""
    # We use object.__new__ to bypass __init__ which calls super().__init__
    # that we don't want to test here.
    inst = object.__new__(BaseAdversarialTraining)
    inst.config = valid_config
    inst._device = torch.device("cpu")
    return inst


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_config_base_valid(base_instance):
    # Should not raise
    base_instance._check_config_base(base_instance.config)


@pytest.mark.advsecurenet
@pytest.mark.essential
@pytest.mark.parametrize(
    "bad_cfg_update,err_msg",
    [
        (
            {"model_config": SimpleNamespace(model=object())},
            "Target model must be a subclass of BaseModel!",
        ),
        ({"models": [object()]}, "All models must be a subclass of BaseModel!"),
        (
            {"attacks": [object()]},
            "All attacks must be a subclass of AdversarialAttack!",
        ),
        (
            {
                "training_process_config": SimpleNamespace(
                    train_loader=object(),
                )
            },
            "train_dataloader must be a DataLoader!",
        ),
    ],
)
def test_check_config_base_invalid(valid_config, bad_cfg_update, err_msg):
    # Create a deep copy to avoid modifying the original fixture
    from copy import deepcopy

    bad_cfg = deepcopy(valid_config)

    # Update the config with the bad value
    if "model_config" in bad_cfg_update:
        bad_cfg.train_config.model_config = bad_cfg_update["model_config"]
    elif "training_process_config" in bad_cfg_update:
        bad_cfg.train_config.training_process_config = bad_cfg_update[
            "training_process_config"
        ]
    else:
        for key, value in bad_cfg_update.items():
            setattr(bad_cfg, key, value)

    bat = object.__new__(BaseAdversarialTraining)
    with pytest.raises(ValueError, match=err_msg):
        bat._check_config_base(bad_cfg)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_pre_training_adds_model_and_sets_train_and_device(base_instance):
    # Initially, target model is in models list
    assert (
        base_instance.config.train_config.model_config.model
        in base_instance.config.models
    )

    # Make a copy to check if a duplicate is added
    initial_models = list(base_instance.config.models)
    base_instance.config.models = initial_models

    base_instance._pre_training()

    # Target model should not be added again
    assert (
        base_instance.config.models.count(
            base_instance.config.train_config.model_config.model
        )
        == 1
    )

    # train() called on each model
    for model in base_instance.config.models:
        assert model._train_called > 0
        assert model._to_called_with == base_instance._device


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_pre_training_no_duplicate_when_model_already_present(base_instance):
    # Ensure model is already in the list
    target_model = base_instance.config.train_config.model_config.model
    base_instance.config.models = [target_model]

    base_instance._pre_training()

    # Should not add a duplicate
    assert len(base_instance.config.models) == 1
    assert base_instance.config.models[0] == target_model


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_train_loader_wraps_dataloader(base_instance):
    from tqdm import tqdm

    wrapped_loader = base_instance._get_train_loader(epoch=1)
    assert isinstance(wrapped_loader, tqdm)
    assert (
        wrapped_loader.iterable
        == base_instance.config.train_config.training_process_config.train_loader
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_loss_divisor(base_instance):
    expected_len = len(
        base_instance.config.train_config.training_process_config.train_loader
    )
    assert base_instance._get_loss_divisor() == expected_len
