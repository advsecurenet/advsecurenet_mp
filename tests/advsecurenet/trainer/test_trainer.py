import logging
import os
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.train_config import TrainConfig
from advsecurenet.trainer.trainer import Trainer
from advsecurenet.trainer import trainer_logic

from advsecurenet.shared.types.configs.train_config import (
    ModelConfig,
    TrainingProcessConfig,
    OptimizationConfig,
    CheckpointConfig,
    FinalModelConfig,
)
from advsecurenet.shared.types.configs.device_config import DeviceConfig

logger = logging.getLogger("advsecurenet.trainer.trainer")


@pytest.fixture
def device(request):
    device_arg = request.config.getoption("--device")
    return torch.device(device_arg if device_arg else "cpu")


@pytest.fixture
def train_config(device):
    model = ModelFactory.create_model(
        model_name="CustomCifar10Model",
        pretrained=False,
        architecture={"num_classes": 1000},
    )

    dataset = TensorDataset(torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,)))
    train_loader = DataLoader(dataset, batch_size=10)

    # Define the training config using the new nested structure
    config = TrainConfig(
        model_config=ModelConfig(model=model),
        training_process_config=TrainingProcessConfig(
            train_loader=train_loader, learning_rate=0.001, epochs=1
        ),
        optimization_config=OptimizationConfig(optimizer="adam"),
        checkpoint_config=CheckpointConfig(
            save_checkpoint=True,
            checkpoint_interval=1,
            save_checkpoint_path="./checkpoints",
        ),
        final_model_config=FinalModelConfig(
            save_final_model=True, save_model_path="./models"
        ),
        device_config=DeviceConfig(processor=device),
    )
    return config


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_trainer_initialization(train_config, device):
    trainer = Trainer(train_config)
    assert trainer._config == train_config
    assert trainer._device == device
    assert trainer.model == train_config.model_config.model
    assert trainer.optimizer is not None
    assert trainer._loss_fn is not None
    assert trainer._scheduler is None


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_setup_device(train_config, device):
    trainer = Trainer(train_config)
    trainer_device = trainer._device
    assert trainer_device == device


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_setup_model(train_config, device):
    trainer = Trainer(train_config)
    model = trainer.model
    assert model == train_config.model_config.model
    assert next(model.parameters()).device.type == device.type


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_setup_optimizer(train_config):
    trainer = Trainer(train_config)
    optimizer = trainer.optimizer
    assert isinstance(optimizer, optim.Adam)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_setup_optimizer_with_kwargs(device):
    model = ModelFactory.create_model(
        model_name="CustomCifar10Model",
        pretrained=False,
        architecture={"num_classes": 1000},
    )
    dataset = TensorDataset(torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,)))
    train_loader = DataLoader(dataset, batch_size=10)

    config = TrainConfig(
        model_config=ModelConfig(model=model),
        training_process_config=TrainingProcessConfig(
            train_loader=train_loader, learning_rate=0.1, epochs=1
        ),
        optimization_config=OptimizationConfig(
            optimizer="adam", optimizer_kwargs={"betas": (0.9, 0.999)}
        ),
        checkpoint_config=CheckpointConfig(save_checkpoint=False),
        final_model_config=FinalModelConfig(save_final_model=False),
        device_config=DeviceConfig(processor=device),
    )

    trainer = Trainer(config)
    optimizer = trainer.optimizer

    assert isinstance(optimizer, optim.Adam)
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["lr"] == 0.1


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_setup_scheduler(device):
    model = ModelFactory.create_model(
        model_name="CustomCifar10Model",
        pretrained=False,
        architecture={"num_classes": 1000},
    )
    dataset = TensorDataset(torch.randn(100, 3, 32, 32), torch.randint(0, 10, (100,)))
    train_loader = DataLoader(dataset, batch_size=10)

    config = TrainConfig(
        model_config=ModelConfig(model=model),
        training_process_config=TrainingProcessConfig(
            train_loader=train_loader, epochs=1
        ),
        optimization_config=OptimizationConfig(optimizer="adam", scheduler="LINEAR_LR"),
        checkpoint_config=CheckpointConfig(save_checkpoint=False),
        final_model_config=FinalModelConfig(save_final_model=False),
        device_config=DeviceConfig(processor=device),
    )

    trainer = Trainer(config)
    scheduler = trainer._scheduler
    assert isinstance(scheduler, torch.optim.lr_scheduler.LinearLR)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("torch.save")
def test_save_checkpoint(mock_save, train_config):
    from advsecurenet.trainer import trainer_logic

    trainer = Trainer(train_config)
    trainer_logic.save_checkpoint(
        epoch=1,
        optimizer=trainer.optimizer,
        model=trainer.model,
        checkpoint_path="test_path",
    )
    assert mock_save.called


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_should_save_checkpoint(train_config):
    from advsecurenet.trainer import trainer_logic

    trainer = Trainer(train_config)
    assert (
        trainer_logic.should_save_checkpoint(
            epoch=1, save_checkpoint=True, checkpoint_interval=1
        )
        is True
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("torch.save")
def test_save_final_model(mock_save, train_config):
    from advsecurenet.trainer import trainer_logic

    trainer = Trainer(train_config)
    trainer_logic.save_final_model(
        model_to_save=trainer.model,
        save_path="./models",
        save_name="test_model",
        model_name="TestModel",
        dataset_name="TestDataset",
        use_ddp=False,
    )
    assert mock_save.called


@pytest.mark.advsecurenet
@pytest.mark.comprehensive
def test_model_training_mode(train_config, device):
    trainer = Trainer(train_config)
    assert trainer.model.training is True


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_log_loss(train_config, tmp_path):
    from advsecurenet.trainer import trainer_logic

    trainer = Trainer(train_config)
    trainer_logic.log_loss(epoch=1, loss=0.61, dir=tmp_path, filename="loss.log")
    assert os.path.exists(tmp_path / "loss.log")
    with open(tmp_path / "loss.log", "r") as f:
        assert f.readline() == "epoch,loss\n"
        assert f.readline() == "1,0.61\n"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_optimizer_with_string():
    """Test get_optimizer method with string optimizer."""
    from torch.optim import Adam
    from advsecurenet.trainer import trainer_logic

    model = torch.nn.Linear(10, 2)

    optimizer = trainer_logic.get_optimizer(
        optimizer="adam", model=model, learning_rate=0.001
    )

    assert isinstance(optimizer, Adam)
    assert optimizer.defaults["lr"] == 0.001


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_optimizer_with_instance():
    """Test get_optimizer method with optimizer instance."""
    from torch.optim import Adam
    from advsecurenet.trainer import trainer_logic

    model = torch.nn.Linear(10, 2)
    adam_optimizer = Adam(model.parameters(), lr=0.01)

    optimizer = trainer_logic.get_optimizer(
        optimizer=adam_optimizer,
        model=model,
        learning_rate=0.001,  # This should be ignored when passing instance
    )

    assert optimizer is adam_optimizer


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_optimizer_with_kwargs():
    """Test get_optimizer method with kwargs."""
    from torch.optim import Adam
    from advsecurenet.trainer import trainer_logic

    model = torch.nn.Linear(10, 2)

    optimizer = trainer_logic.get_optimizer(
        optimizer="adam",
        model=model,
        learning_rate=0.01,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    assert isinstance(optimizer, Adam)
    assert optimizer.defaults["lr"] == 0.01
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    assert optimizer.defaults["weight_decay"] == 0.01


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_train_method_runs():
    """Test that the train method runs without errors."""
    model = ModelFactory.create_model(
        model_name="CustomCifar10Model",
        pretrained=False,
        architecture={"num_classes": 10},
    )
    dataset = TensorDataset(torch.randn(10, 3, 32, 32), torch.randint(0, 10, (10,)))
    train_loader = DataLoader(dataset, batch_size=5)

    config = TrainConfig(
        model_config=ModelConfig(model=model),
        training_process_config=TrainingProcessConfig(
            train_loader=train_loader,
            learning_rate=0.001,
            epochs=1,  # Just one epoch for testing
        ),
        optimization_config=OptimizationConfig(optimizer="adam"),
        checkpoint_config=CheckpointConfig(save_checkpoint=False),
        final_model_config=FinalModelConfig(save_final_model=False),
        device_config=DeviceConfig(processor="cpu"),
    )

    trainer = Trainer(config)

    # Mock the training loop to avoid actually training
    with patch.object(trainer, "_execute_training_loop") as mock_execute:
        trainer.train()
        mock_execute.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("os.path.isfile", return_value=True)
@patch("torch.load")
def test_load_checkpoint_data_success(mock_torch_load, mock_isfile, train_config):
    from advsecurenet.trainer import trainer_logic

    mock_torch_load.return_value = {
        "model_state_dict": "mock_model_state_dict",
        "optimizer_state_dict": "mock_optimizer_state_dict",
        "epoch": 10,
    }
    trainer = Trainer(train_config)
    device = torch.device("cpu")

    result = trainer_logic.load_checkpoint_data("/path/to/checkpoint", device)

    assert result is not None
    assert result["epoch"] == 10
    mock_isfile.assert_called_with("/path/to/checkpoint")
    mock_torch_load.assert_called_with("/path/to/checkpoint", map_location=device)


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("os.path.isfile", return_value=False)
def test_load_checkpoint_data_file_not_found(mock_isfile, train_config):
    from advsecurenet.trainer import trainer_logic

    trainer = Trainer(train_config)
    device = torch.device("cpu")

    result = trainer_logic.load_checkpoint_data("/path/to/checkpoint", device)

    assert result is None
    mock_isfile.assert_called_with("/path/to/checkpoint")


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("os.path.isfile", return_value=True)
@patch("torch.load", side_effect=FileNotFoundError)
def test_load_checkpoint_data_load_error(mock_torch_load, mock_isfile, train_config):
    from advsecurenet.trainer import trainer_logic

    trainer = Trainer(train_config)
    device = torch.device("cpu")

    result = trainer_logic.load_checkpoint_data("/path/to/checkpoint", device)

    assert result is None
    mock_isfile.assert_called_with("/path/to/checkpoint")
    mock_torch_load.assert_called_with("/path/to/checkpoint", map_location=device)


# Additional test cases for Trainer class to improve coverage


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.trainer.trainer_logic.load_checkpoint_data")
def test_trainer_with_checkpoint_loading(mock_load_checkpoint, train_config, device):
    """Test trainer initialization with checkpoint loading."""
    # Configure checkpoint loading
    train_config.checkpoint_config.load_checkpoint = True
    train_config.checkpoint_config.load_checkpoint_path = "/path/to/checkpoint.pth"

    # Create a temporary optimizer to get a proper state dict structure
    temp_optimizer = trainer_logic.get_optimizer(
        train_config.optimization_config.optimizer,
        train_config.model_config.model,
        train_config.training_process_config.learning_rate,
    )

    # Mock the checkpoint data
    mock_checkpoint = {
        "model_state_dict": train_config.model_config.model.state_dict(),
        "optimizer_state_dict": temp_optimizer.state_dict(),
        "epoch": 5,
    }
    mock_load_checkpoint.return_value = mock_checkpoint

    with patch(
        "advsecurenet.trainer.trainer_logic.assign_device_to_optimizer_state"
    ) as mock_assign_device:
        trainer = Trainer(train_config)

        # Verify checkpoint loading was called
        mock_load_checkpoint.assert_called_once_with(
            checkpoint_path="/path/to/checkpoint.pth", device=device
        )
        mock_assign_device.assert_called_once()
        assert trainer.start_epoch == 6  # epoch + 1


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.trainer.trainer_logic.load_checkpoint_data")
def test_trainer_with_failed_checkpoint_loading(mock_load_checkpoint, train_config):
    """Test trainer initialization with failed checkpoint loading."""
    # Configure checkpoint loading
    train_config.checkpoint_config.load_checkpoint = True
    train_config.checkpoint_config.load_checkpoint_path = "/path/to/nonexistent.pth"

    # Mock failed checkpoint loading
    mock_load_checkpoint.return_value = None

    trainer = Trainer(train_config)

    # Should fallback to start_epoch = 1
    assert trainer.start_epoch == 1


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("opacus.validators.ModuleValidator.is_valid")
@patch("opacus.validators.ModuleValidator.fix")
@patch("advsecurenet.trainer.trainer.setup_privacy_engine")
def test_trainer_with_differential_privacy_invalid_model(
    mock_setup_privacy, mock_fix, mock_is_valid, train_config
):
    """Test trainer with differential privacy and invalid model."""
    from advsecurenet.shared.types.configs.train_config import DifferentialPrivacyConfig

    # Set up differential privacy config
    dp_config = DifferentialPrivacyConfig(
        enable=True, noise_multiplier=1.0, delta=1e-5, max_grad_norm=1.0
    )
    train_config.differential_privacy_config = dp_config

    # Mock ModuleValidator
    mock_is_valid.return_value = False
    fixed_model = MagicMock()
    # Set up the fixed model to have parameters for the optimizer
    mock_param = torch.nn.Parameter(torch.randn(10, 10))
    fixed_model.parameters.return_value = iter([mock_param])
    fixed_model.to.return_value = fixed_model
    mock_fix.return_value = fixed_model

    # Mock setup_privacy_engine return values
    mock_privacy_engine = MagicMock()
    mock_private_loss_fn = MagicMock()
    mock_setup_privacy.return_value = (
        fixed_model,  # model
        MagicMock(),  # optimizer
        MagicMock(),  # data_loader
        mock_privacy_engine,  # privacy_engine
        mock_private_loss_fn,  # private_loss_fn
    )

    trainer = Trainer(train_config)

    # Verify model was fixed and privacy engine was set up
    mock_is_valid.assert_called_once()
    mock_fix.assert_called_once()
    mock_setup_privacy.assert_called_once()
    assert trainer._privacy_engine == mock_privacy_engine
    assert trainer._loss_fn == mock_private_loss_fn


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("opacus.validators.ModuleValidator.is_valid")
@patch("advsecurenet.trainer.trainer.setup_privacy_engine")
def test_trainer_with_differential_privacy_valid_model_with_inplace_false(
    mock_setup_privacy, mock_is_valid, train_config
):
    """Test trainer with differential privacy and valid model with inplace_false method."""
    from advsecurenet.shared.types.configs.train_config import DifferentialPrivacyConfig

    # Set up differential privacy config
    dp_config = DifferentialPrivacyConfig(
        enable=True, noise_multiplier=1.0, delta=1e-5, max_grad_norm=1.0
    )
    train_config.differential_privacy_config = dp_config

    # Mock ModuleValidator
    mock_is_valid.return_value = True

    # Add inplace_false method to model
    train_config.model_config.model.inplace_false = MagicMock()

    # Mock setup_privacy_engine return values
    mock_privacy_engine = MagicMock()
    mock_setup_privacy.return_value = (
        train_config.model_config.model,  # model
        MagicMock(),  # optimizer
        MagicMock(),  # data_loader
        mock_privacy_engine,  # privacy_engine
        None,  # private_loss_fn
    )

    trainer = Trainer(train_config)

    # Verify inplace_false was called and no global patch is needed
    train_config.model_config.model.inplace_false.assert_called_once()
    assert trainer._needs_global_patch is False
    mock_setup_privacy.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("opacus.validators.ModuleValidator.is_valid")
@patch("advsecurenet.trainer.trainer.setup_privacy_engine")
def test_trainer_with_differential_privacy_valid_model_without_inplace_false(
    mock_setup_privacy, mock_is_valid, train_config
):
    """Test trainer with differential privacy and valid model without inplace_false method."""
    from advsecurenet.shared.types.configs.train_config import DifferentialPrivacyConfig

    # Set up differential privacy config
    dp_config = DifferentialPrivacyConfig(
        enable=True, noise_multiplier=1.0, delta=1e-5, max_grad_norm=1.0
    )
    train_config.differential_privacy_config = dp_config

    # Mock ModuleValidator
    mock_is_valid.return_value = True

    # Ensure model doesn't have inplace_false method
    if hasattr(train_config.model_config.model, "inplace_false"):
        delattr(train_config.model_config.model, "inplace_false")

    # Mock setup_privacy_engine return values
    mock_privacy_engine = MagicMock()
    mock_setup_privacy.return_value = (
        train_config.model_config.model,  # model
        MagicMock(),  # optimizer
        MagicMock(),  # data_loader
        mock_privacy_engine,  # privacy_engine
        None,  # private_loss_fn
    )

    trainer = Trainer(train_config)

    # Verify global patch is needed
    assert trainer._needs_global_patch is True
    mock_setup_privacy.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_trainer_without_differential_privacy(train_config):
    """Test trainer without differential privacy configuration."""
    # Ensure no differential privacy config
    train_config.differential_privacy_config = None

    trainer = Trainer(train_config)

    # Verify normal initialization without privacy
    assert trainer._privacy_engine is None
    assert trainer._needs_global_patch is False
    assert trainer.model == train_config.model_config.model
    assert trainer._train_loader == train_config.training_process_config.train_loader


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_trainer_differential_privacy_disabled(train_config):
    """Test trainer with differential privacy config but disabled."""
    from advsecurenet.shared.types.configs.train_config import DifferentialPrivacyConfig

    # Set up disabled differential privacy config
    dp_config = DifferentialPrivacyConfig(
        enable=False, noise_multiplier=1.0, delta=1e-5, max_grad_norm=1.0
    )
    train_config.differential_privacy_config = dp_config

    trainer = Trainer(train_config)

    # Verify normal initialization without privacy
    assert trainer._privacy_engine is None
    assert trainer._needs_global_patch is False
    assert trainer.model == train_config.model_config.model


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.trainer.trainer.non_inplace_operations")
@patch.object(Trainer, "_execute_training_loop")
def test_trainer_train_with_global_patch(
    mock_execute_loop, mock_non_inplace, train_config
):
    """Test trainer train method with global patch needed."""
    trainer = Trainer(train_config)
    trainer._needs_global_patch = True

    # Mock the context manager
    mock_context = MagicMock()
    mock_non_inplace.return_value = mock_context

    trainer.train()

    # Verify context manager was used
    mock_non_inplace.assert_called_once()
    mock_context.__enter__.assert_called_once()
    mock_context.__exit__.assert_called_once()
    mock_execute_loop.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch.object(Trainer, "_execute_training_loop")
def test_trainer_train_without_global_patch(mock_execute_loop, train_config):
    """Test trainer train method without global patch needed."""
    trainer = Trainer(train_config)
    trainer._needs_global_patch = False

    trainer.train()

    # Verify training loop was called directly
    mock_execute_loop.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.trainer.trainer_logic.run_epoch")
@patch("advsecurenet.trainer.trainer_logic.should_save_checkpoint")
@patch("advsecurenet.trainer.trainer_logic.define_save_checkpoint_path")
@patch("advsecurenet.trainer.trainer_logic.save_checkpoint")
@patch("advsecurenet.trainer.trainer_logic.post_training")
def test_trainer_execute_training_loop_with_checkpoints(
    mock_post_training,
    mock_save_checkpoint,
    mock_define_path,
    mock_should_save,
    mock_run_epoch,
    train_config,
):
    """Test trainer execute_training_loop with checkpoint saving."""
    # Configure for checkpoint saving
    train_config.training_process_config.epochs = 2
    train_config.checkpoint_config.save_checkpoint = True
    train_config.checkpoint_config.checkpoint_interval = 1

    # Mock return values
    mock_should_save.return_value = True
    mock_define_path.return_value = "/path/to/checkpoint.pth"

    trainer = Trainer(train_config)
    trainer._execute_training_loop()

    # Verify checkpoint operations
    assert mock_run_epoch.call_count == 2  # epochs = 2
    assert mock_should_save.call_count == 2
    assert mock_define_path.call_count == 2
    assert mock_save_checkpoint.call_count == 2
    mock_post_training.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.trainer.trainer_logic.run_epoch")
@patch("advsecurenet.trainer.trainer_logic.should_save_checkpoint")
@patch("advsecurenet.trainer.trainer_logic.post_training")
def test_trainer_execute_training_loop_without_checkpoints(
    mock_post_training, mock_should_save, mock_run_epoch, train_config
):
    """Test trainer execute_training_loop without checkpoint saving."""
    # Configure for no checkpoint saving
    train_config.training_process_config.epochs = 1
    mock_should_save.return_value = False

    trainer = Trainer(train_config)
    trainer._execute_training_loop()

    # Verify no checkpoint operations
    mock_run_epoch.assert_called_once()
    mock_should_save.assert_called_once()
    mock_post_training.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_trainer_with_scheduler(train_config):
    """Test trainer initialization with scheduler."""
    # Add scheduler to config
    train_config.optimization_config.scheduler = "steplr"
    train_config.optimization_config.scheduler_kwargs = {"step_size": 10}

    trainer = Trainer(train_config)

    # Verify scheduler was created
    assert trainer._scheduler is not None
    assert isinstance(trainer._scheduler, optim.lr_scheduler.StepLR)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_trainer_with_optimizer_kwargs(train_config):
    """Test trainer initialization with optimizer kwargs."""
    # Add optimizer kwargs to config
    train_config.optimization_config.optimizer_kwargs = {"weight_decay": 0.001}

    trainer = Trainer(train_config)

    # Verify optimizer was created with kwargs
    assert trainer.optimizer is not None
    assert isinstance(trainer.optimizer, optim.Adam)
