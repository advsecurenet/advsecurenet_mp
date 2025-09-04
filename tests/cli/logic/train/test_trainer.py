from unittest.mock import MagicMock, patch

import pytest

from cli.logic.train.trainer import CLITrainer
from cli.shared.types.train import TrainingCliConfigType


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset")
@patch("cli.logic.train.trainer.create_model")
@patch("cli.logic.train.trainer.get_dataloader")
def test_cli_trainer_init(mock_prepare_dataset, mock_create_model, mock_get_dataloader):
    # Setup
    mock_config = MagicMock()
    mock_train_data = MagicMock()
    mock_prepare_dataset.return_value = mock_train_data

    # Initialize CLITrainer
    trainer = CLITrainer(mock_config)

    # Verify
    assert trainer.config == mock_config


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset", return_value=MagicMock())
@patch("cli.logic.train.trainer.set_visible_gpus")
@patch("cli.logic.train.trainer.DDPCoordinator")
@patch("cli.logic.train.trainer.DDPTrainer")
def test_cli_trainer_execute_ddp_training(
    mock_ddp_trainer,
    mock_ddp_training_coordinator,
    mock_set_visible_gpus,
    mock_prepare_dataset,
):
    # Setup
    mock_config = MagicMock()
    mock_config.device.use_ddp = True
    mock_config.device.gpu_ids = [0, 1]
    trainer = CLITrainer(mock_config)

    # Mock methods
    trainer._prepare_training_environment = MagicMock(return_value=MagicMock())

    # Execute DDP training
    trainer._execute_ddp_training()

    # Verify
    mock_set_visible_gpus.assert_called_once_with([0, 1])
    mock_ddp_training_coordinator.assert_called_once_with(trainer._ddp_training_fn, 2)
    mock_ddp_training_coordinator.return_value.run.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
@patch(
    "cli.logic.train.trainer.CLITrainer._prepare_training_environment",
    return_value=MagicMock(spec=TrainingCliConfigType),
)
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset", return_value=MagicMock())
@patch("cli.logic.train.trainer.Trainer")
def test_cli_trainer_execute_training(
    mock_trainer, mock_prepare_dataset, mock_prepare_training_environment
):
    # Setup
    mock_config = MagicMock()
    mock_config.device.use_ddp = False
    trainer = CLITrainer(mock_config)

    # Mock methods
    mock_train_config = MagicMock()
    trainer._prepare_training_environment = MagicMock(return_value=mock_train_config)

    # Execute training
    trainer._execute_training()

    # Verify
    mock_trainer.assert_called_once_with(mock_train_config)
    mock_trainer.return_value.train.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset", return_value=MagicMock())
@patch("cli.logic.train.trainer.create_model")
def test_cli_trainer_initialize_model(mock_create_model, mock_prepare_dataset):
    # Setup
    mock_config = MagicMock()
    trainer = CLITrainer(mock_config)

    # Execute
    trainer._initialize_model()

    # Verify
    mock_create_model.assert_called_once_with(mock_config.model)


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.train.trainer.get_dataloader")
@patch("cli.logic.train.trainer.get_datasets")
def test_cli_trainer_prepare_dataloader(mock_get_datasets, mock_get_dataloader):
    # Setup
    mock_config = MagicMock()
    mock_train_data = MagicMock()
    mock_get_datasets.return_value = (mock_train_data, None)
    trainer = CLITrainer(mock_config)

    # Execute
    dataloader = trainer._prepare_dataloader()

    # Verify
    mock_get_dataloader.assert_called_once_with(
        config=mock_config.dataloader,
        dataset=mock_train_data,
        dataset_type="train",
        use_ddp=mock_config.device.use_ddp,
    )
    assert dataloader == mock_get_dataloader.return_value


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset", return_value=MagicMock())
@patch("cli.logic.train.trainer.DDPTrainer")
def test_cli_trainer_ddp_training_fn(mock_ddp_trainer, mock_prepare_dataset):
    # Setup
    mock_config = MagicMock()
    trainer = CLITrainer(mock_config)

    # Mock methods
    mock_train_config = MagicMock()
    trainer._prepare_training_environment = MagicMock(return_value=mock_train_config)

    # Execute
    trainer._ddp_training_fn(0, 2)

    # Verify
    mock_ddp_trainer.assert_called_once_with(mock_train_config, 0, 2)
    mock_ddp_trainer.return_value.train.assert_called_once()


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset", return_value=MagicMock())
def test_cli_trainer_execute_with_exception(mock_prepare_dataset):
    # Setup
    mock_config = MagicMock()
    mock_config.device.use_ddp = False
    trainer = CLITrainer(mock_config)

    # Mock methods
    trainer._execute_training = MagicMock(side_effect=Exception("Training error"))

    # Execute and verify exception
    with pytest.raises(Exception, match="Training error"):
        trainer.train()

    trainer._execute_training.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset", return_value=MagicMock())
def test_cli_trainer_execute_ddp_training_no_gpu_ids(mock_prepare_dataset):
    """Test CLITrainer DDP training with no gpu_ids specified."""
    # Mock the config with no gpu_ids
    mock_config = MagicMock()
    mock_config.device.gpu_ids = None
    mock_config.device.use_ddp = True

    # Create CLITrainer instance
    trainer = CLITrainer(mock_config)

    with patch(
        "cli.logic.train.trainer.torch.cuda.device_count", return_value=4
    ), patch("cli.logic.train.trainer.set_visible_gpus") as mock_set_gpus, patch(
        "cli.logic.train.trainer.DDPCoordinator"
    ) as mock_ddp_coordinator:

        # Mock the DDPCoordinator instance
        mock_coordinator_instance = MagicMock()
        mock_ddp_coordinator.return_value = mock_coordinator_instance

        trainer._execute_ddp_training()

        # Verify gpu_ids were set to all available GPUs
        assert mock_config.device.gpu_ids == [0, 1, 2, 3]
        mock_set_gpus.assert_called_once_with([0, 1, 2, 3])
        mock_ddp_coordinator.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset", return_value=MagicMock())
def test_cli_trainer_execute_ddp_training_empty_gpu_ids(mock_prepare_dataset):
    """Test CLITrainer DDP training with empty gpu_ids list."""
    # Mock the config with empty gpu_ids
    mock_config = MagicMock()
    mock_config.device.gpu_ids = []
    mock_config.device.use_ddp = True

    # Create CLITrainer instance
    trainer = CLITrainer(mock_config)

    with patch(
        "cli.logic.train.trainer.torch.cuda.device_count", return_value=2
    ), patch("cli.logic.train.trainer.set_visible_gpus") as mock_set_gpus, patch(
        "cli.logic.train.trainer.DDPCoordinator"
    ) as mock_ddp_coordinator:

        # Mock the DDPCoordinator instance
        mock_coordinator_instance = MagicMock()
        mock_ddp_coordinator.return_value = mock_coordinator_instance

        trainer._execute_ddp_training()

        # Verify gpu_ids were set to all available GPUs
        assert mock_config.device.gpu_ids == [0, 1]
        mock_set_gpus.assert_called_once_with([0, 1])
        mock_ddp_coordinator.assert_called_once()


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset", return_value=MagicMock())
def test_cli_trainer_prepare_training_environment(mock_prepare_dataset):
    """Test CLITrainer prepare training environment method."""
    # Mock the config
    mock_config = MagicMock()

    # Create CLITrainer instance
    trainer = CLITrainer(mock_config)

    # Mock the helper methods
    mock_model = MagicMock()
    mock_dataloader = MagicMock()
    mock_train_config = MagicMock()

    with patch.object(
        trainer, "_initialize_model", return_value=mock_model
    ) as mock_init_model, patch.object(
        trainer, "_prepare_dataloader", return_value=mock_dataloader
    ) as mock_prep_dataloader, patch.object(
        trainer, "_prepare_train_config", return_value=mock_train_config
    ) as mock_prep_config:

        result = trainer._prepare_training_environment()

        # Verify all helper methods were called
        mock_init_model.assert_called_once()
        mock_prep_dataloader.assert_called_once()
        mock_prep_config.assert_called_once_with(mock_model, mock_dataloader)

        # Verify the result is the train config
        assert result == mock_train_config


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset", return_value=MagicMock())
def test_cli_trainer_prepare_train_config(mock_prepare_dataset):
    """Test CLITrainer prepare train config method."""
    # Mock the config with all required attributes
    mock_config = MagicMock()
    mock_config.training.training_hyperparameter = MagicMock()
    mock_config.device = MagicMock()
    mock_config.training.optimization = MagicMock()
    mock_config.training.checkpoint = MagicMock()
    mock_config.training.final_model = MagicMock()
    mock_config.training.differential_privacy = MagicMock()

    # Create CLITrainer instance
    trainer = CLITrainer(mock_config)

    # Mock inputs
    mock_model = MagicMock()
    mock_dataloader = MagicMock()

    # Mock the training process config and asdict
    with patch(
        "cli.logic.train.trainer.TrainingProcessConfig"
    ) as mock_training_process_config, patch(
        "cli.logic.train.trainer.ModelConfig"
    ) as mock_model_config, patch(
        "cli.logic.train.trainer.TrainConfig"
    ) as mock_train_config, patch(
        "cli.logic.train.trainer.asdict"
    ) as mock_asdict:

        # Mock asdict return value
        mock_asdict.return_value = {"epochs": 10, "lr": 0.001}

        # Mock instances
        mock_training_process_instance = MagicMock()
        mock_model_config_instance = MagicMock()
        mock_train_config_instance = MagicMock()

        mock_training_process_config.return_value = mock_training_process_instance
        mock_model_config.return_value = mock_model_config_instance
        mock_train_config.return_value = mock_train_config_instance

        result = trainer._prepare_train_config(mock_model, mock_dataloader)

        # Verify TrainingProcessConfig was created with correct parameters
        mock_training_process_config.assert_called_once_with(
            train_loader=mock_dataloader, epochs=10, lr=0.001
        )

        # Verify ModelConfig was created
        mock_model_config.assert_called_once_with(mock_model)

        # Verify TrainConfig was created with all required configs
        mock_train_config.assert_called_once_with(
            model_config=mock_model_config_instance,
            training_process_config=mock_training_process_instance,
            device_config=mock_config.device,
            optimization_config=mock_config.training.optimization,
            checkpoint_config=mock_config.training.checkpoint,
            final_model_config=mock_config.training.final_model,
            differential_privacy_config=mock_config.training.differential_privacy,
        )

        # Verify the result
        assert result == mock_train_config_instance


@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("cli.logic.train.trainer.CLITrainer._prepare_dataset", return_value=MagicMock())
def test_cli_trainer_train_with_ddp(mock_prepare_dataset):
    """Test CLITrainer train method with DDP enabled to cover line 55."""
    # Mock the config with DDP enabled
    mock_config = MagicMock()
    mock_config.device.use_ddp = True

    # Create CLITrainer instance
    trainer = CLITrainer(mock_config)

    # Mock the _execute_ddp_training method
    with patch.object(trainer, "_execute_ddp_training") as mock_execute_ddp:
        trainer.train()

        # Verify that _execute_ddp_training was called (covers line 55)
        mock_execute_ddp.assert_called_once()
