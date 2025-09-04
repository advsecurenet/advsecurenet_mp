import logging

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from opacus.validators import ModuleValidator

from advsecurenet.trainer import trainer_logic
from advsecurenet.shared.types.configs.train_config import TrainConfig
from advsecurenet.utils.loss import get_loss_function
from advsecurenet.utils.model_utils import non_inplace_operations
from advsecurenet.utils.trainer_utils.differential_privacy_utils import (
    setup_privacy_engine,
)
from advsecurenet.utils.device_utils import setup_device

logger = logging.getLogger(__name__)


class Trainer:
    """
    Base trainer module for training a model.
    """

    def __init__(self, config: TrainConfig):
        """
        Initialize the trainer.

        Args:
            config (TrainConfig): The train config.
        """

        self._config = config
        self._processor = config.device_config.processor
        self._device = self._setup_device()
        self._loss_fn = get_loss_function(config.training_process_config.criterion)
        self._needs_global_patch = False

        model = config.model_config.model.to(self._device)
        train_loader = self._config.training_process_config.train_loader

        if (
            config.differential_privacy_config
            and config.differential_privacy_config.enable
        ):
            if not ModuleValidator.is_valid(model):
                model = ModuleValidator.fix(model)

            if hasattr(model, "inplace_false") and callable(model.inplace_false):
                model.inplace_false()
            else:
                self._needs_global_patch = True

        optimizer_kwargs = config.optimization_config.optimizer_kwargs or {}
        optimizer = trainer_logic.get_optimizer(
            config.optimization_config.optimizer,
            model,
            config.training_process_config.learning_rate,
            **optimizer_kwargs
        )

        start_epoch = 1

        if self._config.checkpoint_config.load_checkpoint:
            checkpoint = trainer_logic.load_checkpoint_data(
                checkpoint_path=self._config.checkpoint_config.load_checkpoint_path,
                device=self._device,
            )

            if checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                trainer_logic.assign_device_to_optimizer_state(optimizer, self._device)
                start_epoch = checkpoint["epoch"] + 1

        self.start_epoch = start_epoch

        if (
            config.differential_privacy_config
            and config.differential_privacy_config.enable
        ):
            (
                self.model,
                self.optimizer,
                self._train_loader,
                self._privacy_engine,
                private_loss_fn,
            ) = setup_privacy_engine(
                model, optimizer, train_loader, config.differential_privacy_config
            )

            if private_loss_fn:
                self._loss_fn = private_loss_fn
        else:
            self.model = model
            self.optimizer = optimizer
            self._train_loader = train_loader
            self._privacy_engine = None

        # Setup model (handles DDP wrapping if needed)
        self.model = self._setup_model(self.model)

        self._scheduler = trainer_logic.get_scheduler(
            scheduler=config.optimization_config.scheduler,
            optimizer=self.optimizer,
            scheduler_kwargs=config.optimization_config.scheduler_kwargs,
        )

    def _execute_training_loop(self) -> None:
        """
        Contains the actual training loop logic. This is called by the train method.
        """
        self._pre_training()

        # Use trange as a context manager to ensure proper cleanup
        with trange(
            self.start_epoch,
            self._config.training_process_config.epochs + 1,
            leave=True,
            position=0,
        ) as epoch_iterator:
            for epoch in epoch_iterator:
                self._run_epoch(epoch)

                if self._should_save_checkpoint(epoch):
                    checkpoint_path = self._get_checkpoint_path(epoch)
                    self._save_checkpoint(epoch, checkpoint_path)

        self._post_training()

    def _run_epoch(self, epoch: int) -> None:
        """
        Runs a single training epoch. Can be overridden for DDP.
        """
        trainer_logic.run_epoch(
            epoch,
            self._train_loader,
            self._device,
            self.model,
            self.optimizer,
            self._loss_fn,
            self._scheduler,
        )

    def _should_save_checkpoint(self, epoch: int) -> bool:
        """
        Determines if a checkpoint should be saved. Can be overridden for DDP.
        """
        return trainer_logic.should_save_checkpoint(
            epoch,
            self._config.checkpoint_config.save_checkpoint,
            self._config.checkpoint_config.checkpoint_interval,
        )

    def _get_checkpoint_path(self, epoch: int) -> str:
        """
        Gets the checkpoint path. Can be overridden for DDP.
        """
        return trainer_logic.define_save_checkpoint_path(
            save_checkpoint_path=self._config.checkpoint_config.save_checkpoint_path,
            save_checkpoint_name=self._config.checkpoint_config.save_checkpoint_name,
            checkpoint_sub_dir=None,  # Not available in config
            model_name="model",  # Default fallback
            dataset_name="dataset",  # Default fallback
            epoch=epoch,
        )

    def _save_checkpoint(self, epoch: int, checkpoint_path: str) -> None:
        """
        Saves a checkpoint. Can be overridden for DDP.
        """
        trainer_logic.save_checkpoint(
            epoch, self.optimizer, self.model, checkpoint_path
        )

    def _post_training(self) -> None:
        """
        Post-training logic. Can be overridden for DDP.
        """
        trainer_logic.post_training(
            save_final_model_flag=self._config.final_model_config.save_final_model,
            model=self.model,
            save_path=self._config.final_model_config.save_model_path,
            save_name=self._config.final_model_config.save_model_name,
            model_name=None,
            dataset_name=None,
            use_ddp=self._config.device_config.use_ddp or False,
            privacy_engine=self._privacy_engine,
            delta=(
                self._config.differential_privacy_config.delta
                if self._config.differential_privacy_config
                else None
            ),
        )

    def train(self) -> None:
        """
        Public method for training the model. It applies a global patch for DP
        compatibility if needed.
        """
        if self._needs_global_patch:
            # If the global patch is needed, run the loop inside the context manager.
            with non_inplace_operations():
                self._execute_training_loop()
        else:
            # Otherwise, run the loop normally.
            self._execute_training_loop()

    def _setup_model(self, model) -> torch.nn.Module:
        """
        Initializes the model and moves it to the device.
        """
        return model.to(self._device)

    def _setup_device(self):
        return setup_device(self._processor)

    def _pre_training(self) -> None:
        # Method to run before training starts.
        self.model.train()
