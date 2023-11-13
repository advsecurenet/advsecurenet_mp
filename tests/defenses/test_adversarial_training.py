import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss

from advsecurenet.models.base_model import BaseModel
from advsecurenet.attacks.adversarial_attack import AdversarialAttack

from advsecurenet.defenses import AdversarialTraining
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import AdversarialTrainingConfig


class DummyModel(BaseModel):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(1, 1)  # simple linear layer

    def forward(self, x):
        return self.fc(x)

    def load_model(self, *args, **kwargs):
        pass

    def models(self):
        return [self]

    def _hook_fn(self, module, input, output):
        pass


class DummyAttack(AdversarialAttack):
    def attack(self, model, data, target):
        return data


class DummyDataset(Dataset):
    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return torch.tensor([1.0]).unsqueeze(0), torch.tensor([1.0]).unsqueeze(0)


@pytest.fixture(scope="module")
def setup():
    # This setup function will be called before any test is executed
    return {
        'optimizer': Adam(DummyModel().parameters()),
        'criterion': CrossEntropyLoss(),
        'data_loader': DataLoader(DummyDataset())
    }


def test_target_model_is_not_base_model(setup):
    config = AdversarialTrainingConfig(
        model="NotAModel",
        models=[],
        attacks=[],
        train_loader=setup['data_loader'],
        optimizer=setup['optimizer'],
        criterion=setup['criterion']
    )
    with pytest.raises(ValueError, match="Target model must be a subclass of BaseModel!"):
        at_training = AdversarialTraining(config)
        at_training.adversarial_training()


def test_models_list_contains_non_base_model(setup):
    config = AdversarialTrainingConfig(
        model=DummyModel(),
        models=["NotAModel"],
        attacks=[],
        train_loader=setup['data_loader'],
        optimizer=setup['optimizer'],
        criterion=setup['criterion']
    )
    with pytest.raises(ValueError, match="All models must be a subclass of BaseModel!"):
        at_training = AdversarialTraining(config)
        at_training.adversarial_training()


def test_attacks_list_contains_non_adversarial_attack(setup):
    config = AdversarialTrainingConfig(
        model=DummyModel(),
        models=[DummyModel()],
        attacks=["NotAnAttack"],
        train_loader=setup['data_loader'],
        optimizer=setup['optimizer'],
        criterion=setup['criterion']
    )
    with pytest.raises(ValueError, match="All attacks must be a subclass of AdversarialAttack!"):
        at_training = AdversarialTraining(config)
        at_training.adversarial_training()


def test_train_dataloader_is_not_dataloader(setup):
    config = AdversarialTrainingConfig(
        model=DummyModel(),
        models=[DummyModel()],
        attacks=[DummyAttack()],
        train_loader="NotADataLoader",
        optimizer=setup['optimizer'],
        criterion=setup['criterion']
    )
    with pytest.raises(ValueError, match="must be a DataLoader!"):
        at_training = AdversarialTraining(config)
        at_training.adversarial_training()


def test_adversarial_training_runs_successfully(setup):
    config = AdversarialTrainingConfig(
        model=DummyModel(),
        models=[DummyModel()],
        attacks=[DummyAttack()],
        train_loader=setup['data_loader'],
        optimizer=setup['optimizer'],
        criterion=setup['criterion']
    )
    # Should not raise any exceptions
    at_training = AdversarialTraining(config)
    at_training.adversarial_training()


def test_adversarial_training_runs_successfully_with_bce_loss(setup):
    config = AdversarialTrainingConfig(
        model=DummyModel(),
        models=[DummyModel()],
        attacks=[DummyAttack()],
        train_loader=setup['data_loader'],
        optimizer=setup['optimizer'],
        criterion=BCEWithLogitsLoss()
    )
    # Should not raise any exceptions
    at_training = AdversarialTraining(config)
    at_training.adversarial_training()
