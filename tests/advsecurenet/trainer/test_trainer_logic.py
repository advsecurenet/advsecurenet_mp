import os
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from advsecurenet.trainer import trainer_logic


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def simple_model():
    """Simple model for testing."""
    return nn.Linear(10, 2)


@pytest.fixture
def simple_data_loader():
    """Simple data loader for testing."""
    dataset = TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,)))
    return DataLoader(dataset, batch_size=10)


@pytest.fixture
def optimizer(simple_model):
    """Simple optimizer for testing."""
    return optim.Adam(simple_model.parameters())


@pytest.fixture
def loss_fn():
    """Simple loss function for testing."""
    return nn.CrossEntropyLoss()


# Mock model with logits attribute for testing
class ModelWithLogits(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        output = MagicMock()
        output.logits = self.linear(x)
        return output


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_run_batch(simple_model, optimizer, loss_fn, device):
    """Test the run_batch function."""
    source = torch.randn(5, 10)
    targets = torch.randint(0, 2, (5,))

    loss = trainer_logic.run_batch(
        source=source,
        targets=targets,
        model=simple_model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=None,
    )

    assert isinstance(loss, float)
    assert loss > 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_run_batch_with_logits():
    """Test run_batch with model that has logits attribute."""
    model = ModelWithLogits()
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    source = torch.randn(5, 10)
    targets = torch.randint(0, 2, (5,))

    loss = trainer_logic.run_batch(
        source=source,
        targets=targets,
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=None,
    )

    assert isinstance(loss, float)
    assert loss > 0


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_run_batch_with_scheduler(simple_model, optimizer, loss_fn, device):
    """Test the run_batch function with scheduler."""
    source = torch.randn(5, 10)
    targets = torch.randint(0, 2, (5,))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1)

    loss = trainer_logic.run_batch(
        source=source,
        targets=targets,
        model=simple_model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,  # type: ignore
    )

    assert isinstance(loss, float)
    assert loss > 0


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.trainer.trainer_logic.run_batch")
@patch("click.echo")
def test_run_epoch(
    mock_echo,
    mock_run_batch,
    simple_data_loader,
    simple_model,
    optimizer,
    loss_fn,
    device,
):
    """Test the run_epoch function."""
    mock_run_batch.return_value = 0.5

    trainer_logic.run_epoch(
        epoch=1,
        train_loader=simple_data_loader,
        device=device,
        model=simple_model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=None,
    )

    # Check that run_batch was called for each batch in the loader
    assert mock_run_batch.call_count == len(simple_data_loader)
    mock_echo.assert_called()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_should_save_checkpoint():
    """Test the should_save_checkpoint function."""
    # Should save when conditions are met
    assert (
        trainer_logic.should_save_checkpoint(
            epoch=5, save_checkpoint=True, checkpoint_interval=5
        )
        is True
    )

    # Should not save when save_checkpoint is False
    assert (
        trainer_logic.should_save_checkpoint(
            epoch=5, save_checkpoint=False, checkpoint_interval=5
        )
        is False
    )

    # Should not save when interval is 0
    assert (
        trainer_logic.should_save_checkpoint(
            epoch=5, save_checkpoint=True, checkpoint_interval=0
        )
        is False
    )

    # Should not save when epoch is not divisible by interval
    assert (
        trainer_logic.should_save_checkpoint(
            epoch=3, save_checkpoint=True, checkpoint_interval=5
        )
        is False
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("torch.save")
@patch("click.echo")
def test_save_checkpoint(mock_echo, mock_save, simple_model, optimizer):
    """Test the save_checkpoint function."""
    checkpoint_path = "/path/to/checkpoint.pth"

    trainer_logic.save_checkpoint(
        epoch=5,
        optimizer=optimizer,
        model=simple_model,
        checkpoint_path=checkpoint_path,
    )

    # Check that torch.save was called with correct arguments
    mock_save.assert_called_once()
    save_args = mock_save.call_args[0]
    checkpoint_data = save_args[0]
    saved_path = save_args[1]

    assert checkpoint_data["epoch"] == 5
    assert "model_state_dict" in checkpoint_data
    assert "optimizer_state_dict" in checkpoint_data
    assert saved_path == checkpoint_path

    # Check that success message was printed
    mock_echo.assert_called()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.trainer.trainer_logic.save_final_model")
@patch("click.echo")
def test_post_training_with_save_final_model(
    mock_echo, mock_save_final_model, simple_model
):
    """Test post_training function with save_final_model flag set to True."""
    trainer_logic.post_training(
        save_final_model_flag=True,
        model=simple_model,
        save_path="/path/to/save",
        save_name="model.pth",
        model_name="test_model",
        dataset_name="test_dataset",
        use_ddp=False,
        privacy_engine=None,
        delta=None,
    )

    # Check that save_final_model was called
    mock_save_final_model.assert_called_once_with(
        simple_model, "/path/to/save", "model.pth", "test_model", "test_dataset", False
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("click.echo")
def test_post_training_without_privacy(mock_echo, simple_model):
    """Test post_training function without privacy engine."""
    trainer_logic.post_training(
        save_final_model_flag=False,
        model=simple_model,
        save_path="",
        save_name="",
        model_name="",
        dataset_name="",
        use_ddp=False,
        privacy_engine=None,
        delta=None,
    )

    # Should not print anything for privacy when privacy_engine is None
    mock_echo.assert_not_called()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("click.echo")
def test_post_training_with_privacy(mock_echo, simple_model):
    """Test post_training function with privacy engine."""
    mock_privacy_engine = MagicMock()
    mock_privacy_engine.get_epsilon.return_value = 1.0
    delta = 1e-5

    trainer_logic.post_training(
        save_final_model_flag=False,
        model=simple_model,
        save_path="",
        save_name="",
        model_name="",
        dataset_name="",
        use_ddp=False,
        privacy_engine=mock_privacy_engine,
        delta=delta,
    )

    # Check that privacy budget was calculated and printed
    mock_privacy_engine.get_epsilon.assert_called_once_with(delta)
    mock_echo.assert_called()

    # Check that the printed message contains privacy information
    call_args = mock_echo.call_args[0][0]
    message = call_args.value if hasattr(call_args, "value") else str(call_args)
    assert "privacy budget" in message.lower()
    assert "ε = 1.00" in message


# SCHEDULER TESTS


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_scheduler_from_string_valid(optimizer):
    """Test creating scheduler from valid string name."""
    scheduler = trainer_logic.create_scheduler_from_string(
        "STEP_LR", optimizer, {"step_size": 10}
    )
    assert isinstance(scheduler, optim.lr_scheduler.StepLR)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_scheduler_from_string_invalid():
    """Test creating scheduler from invalid string name."""
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters())

    with pytest.raises(ValueError, match="Unsupported scheduler"):
        trainer_logic.create_scheduler_from_string("invalid_scheduler", optimizer)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_scheduler_none(optimizer):
    """Test get_scheduler with None input."""
    result = trainer_logic.get_scheduler(None, optimizer)
    assert result is None


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_scheduler_string(optimizer):
    """Test get_scheduler with string input."""
    scheduler = trainer_logic.get_scheduler("STEP_LR", optimizer, {"step_size": 10})
    assert isinstance(scheduler, optim.lr_scheduler.StepLR)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_scheduler_instance(optimizer):
    """Test get_scheduler with scheduler instance."""
    existing_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10)
    result = trainer_logic.get_scheduler(existing_scheduler, optimizer)
    assert result is existing_scheduler


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_scheduler_invalid_type(optimizer):
    """Test get_scheduler with invalid type."""
    with pytest.raises(ValueError, match="Scheduler must be a string or an instance"):
        trainer_logic.get_scheduler(123, optimizer)  # type: ignore


# OPTIMIZER TESTS


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_optimizer_instance():
    """Test get_optimizer with optimizer instance."""
    model = nn.Linear(10, 2)
    existing_optimizer = optim.Adam(model.parameters())
    result = trainer_logic.get_optimizer(existing_optimizer, model)
    assert result is existing_optimizer


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_optimizer_string_no_model():
    """Test get_optimizer with string but no model."""
    with pytest.raises(
        ValueError, match="Model must be provided if optimizer is a string"
    ):
        trainer_logic.get_optimizer("adam", None)  # type: ignore


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_optimizer_none_with_model():
    """Test get_optimizer with None optimizer but with model."""
    model = nn.Linear(10, 2)
    optimizer = trainer_logic.get_optimizer(None, model, learning_rate=0.01)  # type: ignore
    assert isinstance(optimizer, optim.Adam)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_optimizer_invalid_string():
    """Test get_optimizer with invalid optimizer string."""
    model = nn.Linear(10, 2)
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        trainer_logic.get_optimizer("invalid_optimizer", model)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_optimizer_valid_string():
    """Test get_optimizer with valid optimizer string."""
    model = nn.Linear(10, 2)
    optimizer = trainer_logic.get_optimizer("adam", model, learning_rate=0.01)
    assert isinstance(optimizer, optim.Adam)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_optimizer_with_kwargs():
    """Test get_optimizer with additional kwargs."""
    model = nn.Linear(10, 2)
    optimizer = trainer_logic.get_optimizer(
        "adam", model, learning_rate=0.01, weight_decay=0.001
    )
    assert isinstance(optimizer, optim.Adam)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_assign_device_to_optimizer_state():
    """Test assign_device_to_optimizer_state function."""
    model = nn.Linear(10, 2)
    optimizer = optim.Adam(model.parameters())

    # Initialize optimizer state by running one step
    data = torch.randn(5, 10)
    target = torch.randn(5, 2)
    loss = nn.MSELoss()(model(data), target)
    loss.backward()
    optimizer.step()

    device = torch.device("cpu")
    trainer_logic.assign_device_to_optimizer_state(optimizer, device)

    # Check that all tensor states are on the correct device
    for state in optimizer.state.values():
        for v in state.values():
            if isinstance(v, torch.Tensor):
                assert v.device == device


# CHECKPOINT UTILITIES TESTS


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_save_checkpoint_prefix_with_custom_name():
    """Test get_save_checkpoint_prefix with custom name."""
    result = trainer_logic.get_save_checkpoint_prefix(
        "custom_checkpoint", "model", "dataset"
    )
    assert result == "custom_checkpoint"


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_get_save_checkpoint_prefix_without_custom_name():
    """Test get_save_checkpoint_prefix without custom name."""
    result = trainer_logic.get_save_checkpoint_prefix(None, "resnet18", "cifar10")
    assert result == "resnet18_cifar10_checkpoint"


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("os.path.exists", return_value=False)
@patch("os.makedirs")
def test_define_save_checkpoint_path_create_dir(mock_makedirs, mock_exists):
    """Test define_save_checkpoint_path creates directory when it doesn't exist."""
    result = trainer_logic.define_save_checkpoint_path(
        "/custom/path", "custom_checkpoint", "subdir", "model", "dataset", 5
    )

    mock_makedirs.assert_called_once_with("/custom/path")
    assert result == "/custom/path/custom_checkpoint_epoch_5.pth"


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("os.path.exists", return_value=True)
def test_define_save_checkpoint_path_existing_dir(mock_exists):
    """Test define_save_checkpoint_path with existing directory."""
    result = trainer_logic.define_save_checkpoint_path(
        "/existing/path", None, None, "model", "dataset", 10
    )

    assert result == "/existing/path/model_dataset_checkpoint_epoch_10.pth"


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("os.getcwd", return_value="/current/dir")
@patch("os.path.exists", return_value=False)
@patch("os.makedirs")
def test_define_save_checkpoint_path_default_path(
    mock_makedirs, mock_exists, mock_getcwd
):
    """Test define_save_checkpoint_path with default path."""
    result = trainer_logic.define_save_checkpoint_path(
        None, None, "subdir", "model", "dataset", 5
    )

    expected_dir = "/current/dir/checkpoints/subdir"
    mock_makedirs.assert_called_once_with(expected_dir)
    assert result == f"{expected_dir}/model_dataset_checkpoint_epoch_5.pth"


# SAVE FINAL MODEL TESTS


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("os.path.exists", return_value=True)
@patch("torch.save")
@patch("click.echo")
@patch("os.path.isfile", return_value=False)
@patch("os.getcwd", return_value="/current/dir")
def test_save_final_model_default_path(
    mock_getcwd, mock_isfile, mock_echo, mock_torch_save, mock_exists
):
    """Test save_final_model with default path."""
    model = nn.Linear(10, 2)

    trainer_logic.save_final_model(model, None, None, "resnet", "cifar10", False)

    # Verify torch.save was called once and check the file path
    mock_torch_save.assert_called_once()
    call_args = mock_torch_save.call_args
    saved_path = call_args[0][1]
    assert saved_path == "/current/dir/resnet_cifar10_final.pth"
    mock_echo.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("os.path.exists", return_value=True)
@patch("torch.save")
@patch("click.echo")
@patch("os.path.isfile", return_value=False)
def test_save_final_model_custom_name(
    mock_isfile, mock_echo, mock_torch_save, mock_exists
):
    """Test save_final_model with custom save name."""
    # Create separate model instances to avoid circular reference
    actual_model = nn.Linear(10, 2)

    # Create a mock distributed model with .module attribute pointing to actual model
    class MockDistributedModel(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.module = model

    model = MockDistributedModel(actual_model)

    trainer_logic.save_final_model(
        model, "/custom/path", "my_model.pth", None, None, True
    )

    # Verify torch.save was called once and check the file path
    mock_torch_save.assert_called_once()
    call_args = mock_torch_save.call_args
    saved_path = call_args[0][1]
    assert saved_path == "/custom/path/my_model.pth"
    mock_echo.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("os.path.exists", return_value=True)
@patch("torch.save")
@patch("click.echo")
@patch(
    "os.path.isfile", side_effect=[True, True, False]
)  # File exists twice, then doesn't
def test_save_final_model_filename_collision(
    mock_isfile, mock_echo, mock_torch_save, mock_exists
):
    """Test save_final_model handles filename collisions."""
    model = nn.Linear(10, 2)

    trainer_logic.save_final_model(model, "/path", None, "model", None, False)

    # Should try "model_final.pth", then "model_final_1.pth", then "model_final_2.pth"
    mock_torch_save.assert_called_once()
    call_args = mock_torch_save.call_args
    saved_path = call_args[0][1]
    assert saved_path == "/path/model_final_2.pth"
    mock_echo.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("os.path.exists", return_value=True)
@patch("torch.save")
@patch("click.echo")
@patch("os.path.isfile", return_value=False)
def test_save_final_model_no_name_parts(
    mock_isfile, mock_echo, mock_torch_save, mock_exists
):
    """Test save_final_model with no name parts."""
    model = nn.Linear(10, 2)

    trainer_logic.save_final_model(model, "/path", None, None, None, False)

    mock_torch_save.assert_called_once()
    call_args = mock_torch_save.call_args
    saved_path = call_args[0][1]
    assert saved_path == "/path/model_final.pth"
    mock_echo.assert_called_once()


# LOG LOSS TESTS


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("builtins.open", create=True)
@patch("os.path.exists", return_value=True)
@patch("os.getcwd", return_value="/current/dir")
def test_log_loss_existing_file(mock_getcwd, mock_exists, mock_open):
    """Test log_loss with existing log file."""
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    trainer_logic.log_loss(5, 0.123, None, "loss.log")

    # Should open file in append mode and write loss data
    mock_open.assert_called_with("/current/dir/loss.log", "a", encoding="utf-8")
    mock_file.write.assert_called_once_with("5,0.123\n")


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("builtins.open", create=True)
@patch("os.path.exists", return_value=False)
def test_log_loss_new_file(mock_exists, mock_open):
    """Test log_loss creating new log file."""
    mock_file = MagicMock()
    mock_open.return_value.__enter__.return_value = mock_file

    trainer_logic.log_loss(1, 1.234, "/custom/dir", "custom.log")

    # Should create file with header first, then append data
    assert mock_open.call_count == 2
    calls = mock_open.call_args_list

    # First call: create file with header
    assert calls[0][0] == (
        "/custom/dir/custom.log",
        "w",
    )
    # Second call: append data
    assert calls[1][0] == (
        "/custom/dir/custom.log",
        "a",
    )


# CHECKPOINT LOADING TESTS


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_load_checkpoint_data_no_path():
    """Test load_checkpoint_data with no checkpoint path."""
    result = trainer_logic.load_checkpoint_data(None, torch.device("cpu"))
    assert result is None


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("os.path.isfile", return_value=False)
def test_load_checkpoint_data_file_not_found(mock_isfile):
    """Test load_checkpoint_data with non-existent file."""
    result = trainer_logic.load_checkpoint_data("/fake/path.pth", torch.device("cpu"))
    assert result is None


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("os.path.isfile", return_value=True)
@patch("torch.load")
def test_load_checkpoint_data_success(mock_torch_load, mock_isfile):
    """Test load_checkpoint_data with valid checkpoint."""
    mock_checkpoint = {
        "model_state_dict": {"weight": torch.randn(2, 2)},
        "optimizer_state_dict": {"state": {}},
        "epoch": 10,
    }
    mock_torch_load.return_value = mock_checkpoint

    result = trainer_logic.load_checkpoint_data("/valid/path.pth", torch.device("cpu"))

    assert result == mock_checkpoint
    mock_torch_load.assert_called_once_with(
        "/valid/path.pth", map_location=torch.device("cpu")
    )


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("os.path.isfile", return_value=True)
@patch("torch.load")
def test_load_checkpoint_data_malformed(mock_torch_load, mock_isfile):
    """Test load_checkpoint_data with malformed checkpoint."""
    # Missing required keys
    mock_checkpoint = {"epoch": 10}  # Missing model_state_dict and optimizer_state_dict
    mock_torch_load.return_value = mock_checkpoint

    result = trainer_logic.load_checkpoint_data(
        "/malformed/path.pth", torch.device("cpu")
    )

    assert result is None


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("os.path.isfile", return_value=True)
@patch("torch.load", side_effect=Exception("Load error"))
def test_load_checkpoint_data_load_exception(mock_torch_load, mock_isfile):
    """Test load_checkpoint_data with loading exception."""
    result = trainer_logic.load_checkpoint_data("/error/path.pth", torch.device("cpu"))

    assert result is None


# SCHEDULER NAME NORMALIZATION TESTS


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_scheduler_from_string_multisteplr(optimizer):
    """Test create_scheduler_from_string with multisteplr normalization."""
    scheduler = trainer_logic.create_scheduler_from_string(
        "multisteplr", optimizer, {"milestones": [10, 20]}
    )
    assert isinstance(scheduler, optim.lr_scheduler.MultiStepLR)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_scheduler_from_string_cosineannealinglr(optimizer):
    """Test create_scheduler_from_string with cosineannealinglr normalization."""
    scheduler = trainer_logic.create_scheduler_from_string(
        "cosineannealinglr", optimizer, {"T_max": 10}
    )
    assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_scheduler_from_string_cycliclr(optimizer):
    """Test create_scheduler_from_string with cycliclr normalization."""
    scheduler = trainer_logic.create_scheduler_from_string(
        "cycliclr", optimizer, {"base_lr": 0.001, "max_lr": 0.01}
    )
    assert isinstance(scheduler, optim.lr_scheduler.CyclicLR)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_scheduler_from_string_onecyclelr(optimizer):
    """Test create_scheduler_from_string with onecyclelr normalization."""
    scheduler = trainer_logic.create_scheduler_from_string(
        "onecyclelr", optimizer, {"max_lr": 0.01, "total_steps": 100}
    )
    assert isinstance(scheduler, optim.lr_scheduler.OneCycleLR)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_scheduler_from_string_cosineannealingwarmrestarts(optimizer):
    """Test create_scheduler_from_string with cosineannealingwarmrestarts normalization."""
    scheduler = trainer_logic.create_scheduler_from_string(
        "cosineannealingwarmrestarts", optimizer, {"T_0": 10}
    )
    assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_scheduler_from_string_lambdalr(optimizer):
    """Test create_scheduler_from_string with lambdalr normalization."""
    scheduler = trainer_logic.create_scheduler_from_string(
        "lambdalr", optimizer, {"lr_lambda": lambda epoch: 0.95**epoch}
    )
    assert isinstance(scheduler, optim.lr_scheduler.LambdaLR)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_scheduler_from_string_polylr(optimizer):
    """Test create_scheduler_from_string with polylr normalization."""
    scheduler = trainer_logic.create_scheduler_from_string(
        "polylr", optimizer, {"total_iters": 100}
    )
    assert isinstance(scheduler, optim.lr_scheduler.PolynomialLR)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_scheduler_from_string_linearlr(optimizer):
    """Test create_scheduler_from_string with linearlr normalization."""
    scheduler = trainer_logic.create_scheduler_from_string(
        "linearlr", optimizer, {"start_factor": 1.0, "total_iters": 100}
    )
    assert isinstance(scheduler, optim.lr_scheduler.LinearLR)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_scheduler_from_string_reducelronplateau(optimizer):
    """Test create_scheduler_from_string with reducelronplateau normalization."""
    scheduler = trainer_logic.create_scheduler_from_string(
        "reducelronplateau", optimizer, {"factor": 0.5}
    )
    assert isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_scheduler_from_string_steplr(optimizer):
    """Test create_scheduler_from_string with steplr normalization."""
    scheduler = trainer_logic.create_scheduler_from_string(
        "steplr", optimizer, {"step_size": 10}
    )
    assert isinstance(scheduler, optim.lr_scheduler.StepLR)
