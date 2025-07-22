import logging
import os
from typing import Union, cast, Optional

import click
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from opacus.validators import ModuleValidator

from advsecurenet.trainer import trainer_logic

from advsecurenet.shared.optimizer import Optimizer
from advsecurenet.shared.scheduler import Scheduler
from advsecurenet.shared.types.configs.train_config import TrainConfig
from advsecurenet.utils.loss import get_loss_function
from advsecurenet.utils.model_utils import save_model, non_inplace_operations
from advsecurenet.utils.trainer_utils.differential_privacy_utils import setup_privacy_engine
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
        self._device = setup_device(config.device_config.processor)
        self._loss_fn = get_loss_function(config.training_process_config.criterion)
        self._needs_global_patch = False

        model = config.model_config.model.to(self._device)
        train_loader = self._config.training_process_config.train_loader

        if (config.differential_privacy_config):
            if not ModuleValidator.is_valid(model):
                model = ModuleValidator.fix(model)

            if hasattr(model, 'inplace_false') and callable(model.inplace_false):
                model.inplace_false()
            else:
                self._needs_global_patch = True

        optimizer_kwargs = config.optimization_config.optimizer_kwargs or {}
        optimizer = self._get_optimizer(config.optimization_config.optimizer, model, config.training_process_config.learning_rate, **optimizer_kwargs)

        start_epoch = 1

        if self._config.checkpoint_config.load_checkpoint:
            checkpoint = self._load_checkpoint_data(
                checkpoint_path=self._config.checkpoint_config.load_checkpoint_path,
                device=self._device,
            )
        
            if checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                self._assign_device_to_optimizer_state(optimizer, self._device)
                start_epoch = checkpoint["epoch"] + 1
            
        self.start_epoch = start_epoch

        if (
            not config.differential_privacy_config
            or not config.differential_privacy_config.enable
        ):
            self.model = model
            self.optimizer = optimizer
            self._train_loader = train_loader
            self._privacy_engine = None
        else:
            (
            self.model, 
            self.optimizer, 
            self._train_loader, 
            self._privacy_engine, 
            private_loss_fn
            ) = setup_privacy_engine(
                model, 
                optimizer,
                train_loader,
                config.differential_privacy_config
            )

            if private_loss_fn:
                self._loss_fn = private_loss_fn

        self._scheduler = self._get_scheduler(
            scheduler=config.optimization_config.scheduler,
            optimizer=self.optimizer,
            scheduler_kwargs=config.optimization_config.scheduler_kwargs,
        )

    def _execute_training_loop(self) -> None:
        """
        Contains the actual training loop logic. This is called by the train method.
        """
        self.model.train() # pre_training logic
        for epoch in trange(
            self.start_epoch, self._config.training_process_config.epochs + 1, leave=True, position=0
        ):
            trainer_logic.run_epoch(
                epoch, 
                self._train_loader, 
                self._device, 
                self.model, 
                self.optimizer, 
                self._loss_fn, 
                self._scheduler
            )
            if trainer_logic.should_save_checkpoint(epoch, self._config.checkpoint_config.save_checkpoint, self._config.checkpoint_config.checkpoint_interval):
                # You would need a helper to generate the path
                checkpoint_path = self._define_save_checkpoint_path(epoch)
                trainer_logic.save_checkpoint(epoch, self.optimizer, self.model, checkpoint_path)
        trainer_logic.post_training(self._config.final_model_config.save_final_model, self.model, self._config.final_model_config.save_model_path, self._config.final_model_config.save_model_name, model_name=None, dataset_name=None, use_ddp=self._config.device_config.use_ddp, privacy_engine=self._privacy_engine, delta=self._config.differential_privacy_config.delta)

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

    @staticmethod
    def _get_scheduler(
        scheduler: Optional[Union[str, torch.optim.lr_scheduler._LRScheduler]],
        optimizer: optim.Optimizer,
        scheduler_kwargs: Optional[dict] = None,
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        Returns the scheduler based on the given scheduler string or torch.optim.lr_scheduler._LRScheduler.

        Args:
            scheduler (str or torch.optim.lr_scheduler._LRScheduler, optional): The scheduler. Defaults to None.
            optimizer (optim.Optimizer): The optimizer for the scheduler.
            scheduler_kwargs (Optional[dict]): Additional keyword arguments for the scheduler.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The scheduler. I.e. ReduceLROnPlateau, etc.
        """
        if scheduler is None:
            return None
        if isinstance(scheduler, str):
            if scheduler.upper() not in Scheduler.__members__:
                raise ValueError(
                    "Unsupported scheduler! Choose from: "
                    + ", ".join([e.name for e in Scheduler])
                )
            scheduler_function_class = Scheduler[scheduler.upper()].value
            # Use the passed-in kwargs, with a fallback to an empty dict
            kwargs = scheduler_kwargs or {}
            scheduler = scheduler_function_class(optimizer, **kwargs)
        elif not isinstance(scheduler, torch.optim.lr_scheduler._LRScheduler):
            raise ValueError(
                "Scheduler must be a string or an instance of torch.optim.lr_scheduler._LRScheduler."
            )
        return cast(torch.optim.lr_scheduler._LRScheduler, scheduler)

    @staticmethod
    def _get_optimizer(
        optimizer: Union[str, optim.Optimizer],
        model: nn.Module,
        learning_rate: float = 0.001,
        **kwargs,
    ) -> optim.Optimizer:
        """
        Returns the optimizer based on the given optimizer string or optim.Optimizer.

        Args:
            optimizer (str or optim.Optimizer, optional): The optimizer. Defaults to Adam with learning rate 0.001.
            model (nn.Module, optional): The model to optimize. Required if optimizer is a string.
            learning_rate (float, optional): The learning rate. Defaults to 0.001.

        Returns:
            optim.Optimizer: The optimizer.

        Examples:

            >>> _get_optimizer("adam")
            >>> _get_optimizer(optim.Adam(model.parameters(), lr=0.001))

        """

        # if the optimizer is already an instance of optim.Optimizer, return it
        if isinstance(optimizer, optim.Optimizer):
            return optimizer

        if model is None and isinstance(optimizer, str):
            raise ValueError("Model must be provided if optimizer is a string.")

        # if the model is provided but the optimizer not, initialize the default optimizer
        if model is not None and optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        #  if the model is provided and the optimizer is a string, initialize the optimizer based on the string
        if model is not None and isinstance(optimizer, str):
            if optimizer.upper() not in Optimizer.__members__:
                raise ValueError(
                    "Unsupported optimizer! Choose from: "
                    + ", ".join([e.name for e in Optimizer])
                )

            optimizer_class = Optimizer[optimizer.upper()].value
            optimizer = optimizer_class(model.parameters(), lr=learning_rate, **kwargs)

        return cast(optim.Optimizer, optimizer)

    @staticmethod
    def _assign_device_to_optimizer_state(optimizer: optim.Optimizer, device: torch.device) -> None:
        # Pass optimizer as an argument since ptimizer might not be the one we want yet
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    @staticmethod
    def _get_save_checkpoint_prefix(save_checkpoint_name: Optional[str], model_name: str, dataset_name: str) -> str:
        """
        Returns the save checkpoint prefix.

        Returns:
            str: The save checkpoint prefix.

        Notes:
            If the save checkpoint name is provided, it will be used as the prefix. Otherwise, the model variant and the dataset name will be used as the prefix.
        """

        if save_checkpoint_name:
            return save_checkpoint_name
        else:
            return f"{model_name}_{dataset_name}_checkpoint"
        
    @staticmethod
    def _define_save_checkpoint_path(save_checkpoint_path: Optional[str], save_checkpoint_name: Optional[str], checkpoint_sub_dir: Optional[str], model_name: str, dataset_name: str, epoch) -> str:

        checkpoint_dir = save_checkpoint_path or os.path.join(
            os.getcwd(), f"checkpoints/{checkpoint_sub_dir}"
        )

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        save_checkpoint_prefix = Trainer._get_save_checkpoint_prefix(save_checkpoint_name, model_name, dataset_name)

        checkpoint_filename = f"{save_checkpoint_prefix}_epoch_{epoch}.pth"

        return os.path.join(checkpoint_dir, checkpoint_filename)

    @staticmethod
    def _save_checkpoint(epoch: int, optimizer: optim.Optimizer, model: nn.Module, save_checkpoint_path: str) -> None:
        """
        Saves the checkpoint.

        Args:
            epoch (int): The current epoch.
            optimizer (optim.Optimizer): The optimizer.
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            save_checkpoint_path,
        )
        click.echo(click.style(f"Saved checkpoint to {save_checkpoint_path}", fg="green"))

    @staticmethod
    def _should_save_checkpoint(epoch: int, save_checkpoint: bool, checkpoint_interval: int) -> bool:
        """
        Determines if a checkpoint should be saved based on the given epoch, the checkpoint interval and the current rank.
        Args:
            epoch (int): The current epoch.
        Returns:
            bool: True if a checkpoint should be saved, False otherwise.
        """
        return (
            save_checkpoint
            and checkpoint_interval > 0
            and epoch % checkpoint_interval == 0
        )
    

    @staticmethod
    def _save_final_model(
        model_to_save: nn.Module,
        save_path: Optional[str],
        save_name: Optional[str],
        model_name: Optional[str],
        dataset_name: Optional[str],
        use_ddp: bool,
    ) -> None:
        """
        Saves the final model to a specified path, handling filename collisions.
        """
        final_save_path = save_path or os.getcwd()

        if save_name:
            base_name = save_name
        else:
            name_parts = [part for part in [model_name, dataset_name] if part]
            if not name_parts:
                name_parts = ["model"]
            name_parts.append("final")
            base_name = "_".join(name_parts) + ".pth"

        final_save_name_with_path = os.path.join(final_save_path, base_name)
        index = 0
        name_root, name_ext = os.path.splitext(base_name)

        # This loop correctly generates and tests a new candidate path on each iteration.
        while os.path.isfile(final_save_name_with_path):
            index += 1
            new_filename = f"{name_root}_{index}{name_ext}"
            final_save_name_with_path = os.path.join(final_save_path, new_filename)

        save_model(
            model=model_to_save,
            filename=os.path.basename(final_save_name_with_path),
            filepath=final_save_path,
            distributed=use_ddp,
        )
    @staticmethod
    def _log_loss(epoch: int, loss: float, dir: str = None, filename: str = "loss.log"
    ) -> None:
        path = (
            os.path.join(dir, filename) if dir else os.path.join(os.getcwd(), filename)
        )
        # Save the loss to the log file. If the log file does not exist, create it in the current directory.
        if not os.path.exists(path):
            with open(path, "w", encoding="utf-8") as f:
                f.write("epoch,loss\n")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{loss}\n")

    
    @staticmethod
    def _load_checkpoint_data(
         checkpoint_path: Optional[str], device: torch.device
    ) -> Optional[dict]:
        """
        Loads checkpoint data from a file. This is a static utility method.

        Args:
            load_checkpoint (bool): Flag indicating if loading a checkpoint is enabled.
            checkpoint_path (Optional[str]): The path to the checkpoint file.
            device (torch.device): The device to map the loaded checkpoint to.

        Returns:
            Optional[dict]: The loaded checkpoint dictionary, or None if not loaded.
        """

        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            logger.warning("Checkpoint file not found at %s. Not loading.", checkpoint_path)
            return None
        
        try:
            logger.info("Loading checkpoint from %s", checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Basic validation to ensure essential keys exist
            if "model_state_dict" not in checkpoint or "optimizer_state_dict" not in checkpoint or "epoch" not in checkpoint:
                logger.error("Checkpoint is malformed or missing required keys. Not loading.")
                return None
            return checkpoint
        except Exception as e:
            logger.error("Failed to load checkpoint file from %s: %s", checkpoint_path, e)
            return None