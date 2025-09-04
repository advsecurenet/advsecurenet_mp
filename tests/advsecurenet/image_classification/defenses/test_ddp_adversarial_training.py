from unittest.mock import MagicMock, patch

import pytest
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from advsecurenet.computer_vision.image_classification.defenses.ddp_adversarial_training import (
    DDPAdversarialTraining,
)
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import (
    AdversarialTrainingConfig,
)

from advsecurenet.shared.types.configs.train_config import (
    ModelConfig,
    TrainingProcessConfig,
)
from shared.types.configs.base import (
    OptimizationBase,
    CheckpointBase,
    FinalModelBase,
)

from advsecurenet.shared.types.configs.device_config import DeviceConfig
from advsecurenet.shared.types.configs.train_config import TrainConfig


def create_mock_adversarial_training_config(mock_train_loader):
    """Helper function to create a properly structured mock config."""
    mock_model = MagicMock()
    mock_attack = MagicMock()

    # Create a TrainConfig with the new nested structure
    train_config = TrainConfig(
        model_config=ModelConfig(model=mock_model),
        training_process_config=TrainingProcessConfig(
            train_loader=mock_train_loader, epochs=1
        ),
        optimization_config=OptimizationBase(optimizer="adam"),
        checkpoint_config=CheckpointBase(save_checkpoint=False),
        final_model_config=FinalModelBase(save_final_model=False),
        device_config=DeviceConfig(processor="cpu"),
    )

    return AdversarialTrainingConfig(
        train_config=train_config, models=[mock_model], attacks=[mock_attack]
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch(
    "advsecurenet.computer_vision.image_classification.defenses.ddp_adversarial_training.DDPTrainer.__init__"
)
@patch(
    "advsecurenet.computer_vision.image_classification.defenses.ddp_adversarial_training.AdversarialTraining.__init__"
)
def test_ddp_adversarial_training_init(
    mock_adversarial_training_init, mock_ddp_trainer_init
):
    mock_train_loader = MagicMock(spec=DataLoader)
    config = create_mock_adversarial_training_config(mock_train_loader)
    rank = 0
    world_size = 1

    DDPAdversarialTraining(config, rank, world_size)

    mock_ddp_trainer_init.assert_called_once()
    mock_adversarial_training_init.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(DDPAdversarialTraining, "__init__", return_value=None)
def test_get_train_loader(mock_ddp_adversarial_training_init):
    # Create mock config and dataset
    mock_sampler = MagicMock(spec=DistributedSampler)
    mock_train_loader = MagicMock(spec=DataLoader)
    mock_train_loader.sampler = mock_sampler

    mock_config = create_mock_adversarial_training_config(mock_train_loader)

    # Initialize the DDPAdversarialTraining instance
    ddp_adversarial_training_instance = DDPAdversarialTraining.__new__(
        DDPAdversarialTraining
    )
    ddp_adversarial_training_instance.config = mock_config
    ddp_adversarial_training_instance._rank = 0

    # Test the _get_train_loader method
    epoch = 1
    with patch(
        "advsecurenet.computer_vision.image_classification.defenses.ddp_adversarial_training.tqdm",
        return_value=mock_train_loader,
    ) as mock_tqdm:
        train_loader = ddp_adversarial_training_instance._get_train_loader(epoch)

    # Assertions
    mock_sampler.set_epoch.assert_called_once_with(epoch)
    mock_tqdm.assert_called_once_with(
        mock_train_loader,
        desc="Adversarial Training",
        leave=False,
        position=1,
        unit="batch",
        colour="blue",
    )
    assert train_loader == mock_train_loader


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(DDPAdversarialTraining, "__init__", return_value=None)
def test_get_train_loader_non_zero_rank(mock_ddp_adversarial_training_init):
    # Create mock config and dataset
    mock_sampler = MagicMock(spec=DistributedSampler)
    mock_train_loader = MagicMock(spec=DataLoader)
    mock_train_loader.sampler = mock_sampler

    mock_config = create_mock_adversarial_training_config(mock_train_loader)

    # Initialize the DDPAdversarialTraining instance
    ddp_adversarial_training_instance = DDPAdversarialTraining.__new__(
        DDPAdversarialTraining
    )
    ddp_adversarial_training_instance.config = mock_config
    ddp_adversarial_training_instance._rank = 1

    # Test the _get_train_loader method
    epoch = 1
    train_loader = ddp_adversarial_training_instance._get_train_loader(epoch)

    # Assertions
    mock_sampler.set_epoch.assert_called_once_with(epoch)
    assert train_loader == mock_train_loader


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(DDPAdversarialTraining, "__init__", return_value=None)
def test_get_loss_divisor(mock_ddp_adversarial_training_init):
    # Create mock config and dataset
    mock_train_loader = MagicMock()
    # Mock the length of the train loader
    mock_train_loader.__len__.return_value = 10

    mock_config = create_mock_adversarial_training_config(mock_train_loader)

    # Initialize the DDPAdversarialTraining instance
    ddp_adversarial_training_instance = DDPAdversarialTraining.__new__(
        DDPAdversarialTraining
    )
    ddp_adversarial_training_instance.config = mock_config
    ddp_adversarial_training_instance._world_size = 4

    # Test the _get_loss_divisor method
    loss_divisor = ddp_adversarial_training_instance._get_loss_divisor()

    # Assertions
    assert loss_divisor == 40, "Loss divisor calculation is incorrect"
