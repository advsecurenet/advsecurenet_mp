import logging
from dataclasses import asdict

import click
import torch

from advsecurenet.distributed.ddp_coordinator import DDPCoordinator
from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.train_config import TrainConfig, ModelConfig, TrainingProcessConfig, OptimizationConfig, DifferentialPrivacyConfig, CheckpointConfig, FinalModelConfig
from advsecurenet.trainer.ddp_trainer import DDPTrainer
from advsecurenet.trainer.trainer import Trainer
from advsecurenet.utils.ddp import set_visible_gpus
from cli.shared.types.train import TrainingCliConfigType
from cli.shared.utils.dataloader import get_dataloader
from cli.shared.utils.dataset import get_datasets
from cli.shared.utils.model import create_model

logger = logging.getLogger(__name__)


class CLITrainer:
    """
    Trainer class for the CLI. This module parses the CLI arguments and trains the model.

    Args:
        config (TrainingCliConfigType): The configuration for training.

    Attributes:
        config (TrainingCliConfigType): The configuration for training.

    """

    def __init__(self, config: TrainingCliConfigType):
        self.config: TrainingCliConfigType = config
        self.train_dataset = self._prepare_dataset()

    def train(self):
        """
        The main training function. This function parses the CLI arguments and executes the training.
        """

        try:
            if self.config.device.use_ddp:
                self._execute_ddp_training()
            else:
                self._execute_training()

        except Exception as e:
            logger.error(f"Error occurred during training: {e}")
            raise e

    def _execute_ddp_training(self) -> None:
        """
        DDP Training function. Initializes the DDPCoordinator and runs the training.
        """
        # if no gpu ids are provided, use all available gpus
        if self.config.device.gpu_ids is None or len(self.config.device.gpu_ids) == 0:
            self.config.device.gpu_ids = list(range(torch.cuda.device_count()))

        world_size = len(self.config.device.gpu_ids)

        set_visible_gpus(self.config.device.gpu_ids)

        ddp_trainer = DDPCoordinator(
            self._ddp_training_fn,
            world_size,
        )

        if self.config.training.verbose:
            click.echo(
                f"Running DDP training on {world_size} GPUs with the following IDs: {self.config.device.gpu_ids}"
            )

        ddp_trainer.run()

    def _ddp_training_fn(self, rank: int, world_size: int) -> None:
        train_config = self._prepare_training_environment()

        ddp_trainer = DDPTrainer(train_config, rank, world_size)
        ddp_trainer.train()

    def _execute_training(self) -> None:
        train_config = self._prepare_training_environment()

        trainer = Trainer(train_config)
        trainer.train()

    def _prepare_training_environment(self) -> TrainConfig:
        """
        Prepare the common training environment components like model and dataloader.

        Returns:
            TrainConfig: The training configuration.
        """
        model = self._initialize_model()

        train_loader = self._prepare_dataloader()
        train_config = self._prepare_train_config(model, train_loader)

        return train_config

    def _initialize_model(self) -> BaseModel:
        """
        Initialize the model.

        Returns:
            BaseModel: The initialized model.
        """

        return create_model(self.config.model)

    def _prepare_dataset(self):
        """
        Prepare the datasets.
        """
        train_data, _ = get_datasets(config=self.config.dataset)
        return train_data

    def _prepare_dataloader(self) -> torch.utils.data.DataLoader:
        """
        Initialize the dataloader for single process training.

        Returns:

            torch.utils.data.DataLoader: The training dataloader.
        """
        train_data_loader = get_dataloader(
            config=self.config.dataloader,
            dataset=self.train_dataset,
            dataset_type="train",
            use_ddp=self.config.device.use_ddp,
        )
        return train_data_loader

    def _prepare_train_config(
        self, model: BaseModel, train_data_loader: torch.utils.data.DataLoader
    ) -> TrainConfig:
        """
        Prepare the training config.
        """
        training_process_config = TrainingProcessConfig(
            train_loader=train_data_loader,
            criterion=self.config.training.training_hyperparameter.criterion,
            epochs=self.config.training.training_hyperparameter.epochs,
            learning_rate=self.config.training.training_hyperparameter.learning_rate,
            verbose=self.config.training.training_hyperparameter.verbose,
        )

        optimization_config = OptimizationConfig(
            optimizer=self.config.training.optimization.optimizer,
            optimizer_kwargs=self.config.training.optimization.optimizer_kwargs,
            scheduler=self.config.training.optimization.scheduler,
            scheduler_kwargs=self.config.training.optimization.scheduler_kwargs,
        )

        if not self.config.training.differential_privacy:
            differential_privacy_config = None
        else:
            differential_privacy_config = DifferentialPrivacyConfig(
                enable=self.config.training.differential_privacy.enable,
                noise_multiplier=self.config.training.differential_privacy.noise_multiplier,
                max_grad_norm=self.config.training.differential_privacy.max_grad_norm,
                delta=self.config.training.differential_privacy.delta,
                kwargs=self.config.training.differential_privacy.kwargs,
            )

        checkpoint_config = CheckpointConfig(
            save_checkpoint=self.config.training.checkpoint.save_checkpoint,
            save_checkpoint_path=self.config.training.checkpoint.save_checkpoint_path,
            save_checkpoint_name=self.config.training.checkpoint.save_checkpoint_name,
            checkpoint_interval=self.config.training.checkpoint.checkpoint_interval,
            load_checkpoint=self.config.training.checkpoint.load_checkpoint,
            load_checkpoint_path=self.config.training.checkpoint.load_checkpoint_path,
        )

        final_model_config = FinalModelConfig(
            save_final_model=self.config.training.final_model.save_final_model,
            save_model_path=self.config.training.final_model.save_model_path,
            save_model_name=self.config.training.final_model.save_model_name,
        )

        config = TrainConfig(
            model_config=ModelConfig(model),
            training_process_config=training_process_config,
            device_config=self.config.device,
            optimization_config=optimization_config,
            checkpoint_config=checkpoint_config,
            final_model_config=final_model_config,
            differential_privacy_config=differential_privacy_config,
        )
        return config
