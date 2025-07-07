import logging
import os
from typing import Union, cast, Optional

import click
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

from opacus import PrivacyEngine

from advsecurenet.shared.optimizer import Optimizer
from advsecurenet.shared.scheduler import Scheduler
from advsecurenet.shared.types.configs.train_config import TrainConfig
from advsecurenet.utils.loss import get_loss_function
from advsecurenet.utils.model_utils import save_model
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
        self._device = setup_device(config.processor)
        self._loss_fn = get_loss_function(config.criterion)

        model = config.model.to(self._device)
        train_loader = self._config.train_loader

        optimizer_kwargs = config.optimizer_kwargs or {}
        optimizer = self._get_optimizer(config.optimizer, model, config.learning_rate, **optimizer_kwargs)

        checkpoint = self._load_checkpoint_data(
            load_checkpoint=self._config.load_checkpoint,
            checkpoint_path=self._config.load_checkpoint_path,
            device=self._device,
        )
        
        start_epoch = 1
        if checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self._assign_device_to_optimizer_state(optimizer) # Pass optimizer explicitly
            start_epoch = checkpoint["epoch"] + 1
            
        self._start_epoch = start_epoch


        if (
            not config.differential_privacy
            or not config.differential_privacy.enable
        ):
            self._model = model
            self._optimizer = optimizer
            self._train_loader = train_loader
            self._privacy_engine = None
        else:
            (
            self._model, 
            self._optimizer, 
            self._train_loader, 
            self._privacy_engine, 
            private_loss_fn
            ) = setup_privacy_engine(
                model, 
                optimizer,
                train_loader,
                config.differential_privacy
            )

            if private_loss_fn:
                self._loss_fn = private_loss_fn

        self._scheduler = self._get_scheduler(
            scheduler=config.scheduler,
            optimizer=self._optimizer,
            scheduler_kwargs=config.scheduler_kwargs,
        )
        


    def train(self) -> None:
        """
        Public method for training the model.
        """
        self._pre_training()
        for epoch in trange(
            self._start_epoch, self._config.epochs + 1, leave=True, position=0
        ):
            self._run_epoch(epoch)
            if self._should_save_checkpoint(epoch):
                self._save_checkpoint(epoch, self._optimizer)<
        self._post_training()

    def _setup_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """
        Initializes the scheduler based on the given scheduler string or torch.optim.lr_scheduler._LRScheduler.

        Returns:
            torch.optim.lr_scheduler._LRScheduler: The scheduler. I.e. ReduceLROnPlateau, etc.
        """
        scheduler = self._get_scheduler(self._config.scheduler, self._optimizer)
        return scheduler

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

    def _get_optimizer(
        self,
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

    def _load_checkpoint_if_any(self) -> int:
        """
        Loads the checkpoint if any and returns the start epoch.

        Returns:
            int: The start epoch.
        """
        try:
            start_epoch = 1
            if self._config.load_checkpoint and self._config.load_checkpoint_path:
                if os.path.isfile(self._config.load_checkpoint_path):
                    logger.info(
                        "Loading checkpoint from %s", self._config.load_checkpoint_path
                    )
                    checkpoint = torch.load(self._config.load_checkpoint_path)
                    self._load_model_state_dict(checkpoint["model_state_dict"])
                    self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    self._assign_device_to_optimizer_state()
                    start_epoch = checkpoint["epoch"] + 1
                else:
                    logger.warning(
                        "Checkpoint file not found at %s",
                        self._config.load_checkpoint_path,
                    )
            return start_epoch
        except Exception as e:
            logger.error("Failed to load checkpoint: %s", e)
            return 1

    def _load_model_state_dict(self, state_dict):
        # Loads the given model state dict.
        self._model.load_state_dict(state_dict)

    def _get_model_state_dict(self) -> dict:
        # Returns the model state dict.
        return self._model.state_dict()

    def _assign_device_to_optimizer_state(self, optimizer: optim.Optimizer):
        # Pass optimizer as an argument since self._optimizer might not be the one we want yet
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(self._device)

    def _get_save_checkpoint_prefix(self) -> str:
        """
        Returns the save checkpoint prefix.

        Returns:
            str: The save checkpoint prefix.

        Notes:
            If the save checkpoint name is provided, it will be used as the prefix. Otherwise, the model variant and the dataset name will be used as the prefix.
        """

        if self._config.save_checkpoint_name:
            return self._config.save_checkpoint_name
        else:
            return f"{self._config.model._model_name}_{self._train_loader.dataset.__class__.__name__}_checkpoint"

    def _save_checkpoint(self, epoch: int, optimizer: optim.Optimizer) -> None:
        """
        Saves the checkpoint.

        Args:
            epoch (int): The current epoch.
            optimizer (optim.Optimizer): The optimizer.
        """
        checkpoint_sub_dir = "training"
        checkpoint_dir = self._config.save_checkpoint_path or os.path.join(
            os.getcwd(), f"checkpoints/{checkpoint_sub_dir}"
        )

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        save_checkpoint_prefix = self._get_save_checkpoint_prefix()
        checkpoint_filename = f"{save_checkpoint_prefix}_epoch_{epoch}.pth"
        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self._get_model_state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            checkpoint_path,
        )
        click.echo(click.style(f"Saved checkpoint to {checkpoint_path}", fg="green"))

    def _should_save_checkpoint(self, epoch: int) -> bool:
        """
        Determines if a checkpoint should be saved based on the given epoch, the checkpoint interval and the current rank.
        Args:
            epoch (int): The current epoch.
        Returns:
            bool: True if a checkpoint should be saved, False otherwise.
        """
        return (
            self._config.save_checkpoint
            and self._config.checkpoint_interval > 0
            and epoch % self._config.checkpoint_interval == 0
        )

    def _should_save_final_model(self) -> bool:
        """
        Determines if the final model should be saved based on the given save_final_model flag and the current rank.
        """
        return self._config.save_final_model

    def _save_final_model(self) -> None:
        """
        Saves the final model to the current directory with the name of the model variant and the dataset name.
        """
        if not self._config.save_model_path:
            self._config.save_model_path = os.getcwd()

        model_name = (
            self._config.model._model_name
            if hasattr(self._config.model, "_model_name")
            else "model"
        )
        dataset_name = (
            self._train_loader.dataset.name
            if hasattr(self._train_loader.dataset, "name")
            else "dataset"
        )

        if not self._config.save_model_name:
            self._config.save_model_name = f"{model_name}_{dataset_name}_final.pth"

        # if the same file exists, add a index to the file name
        index = 0
        while os.path.isfile(self._config.save_model_name):
            index += 1
            self._config.save_model_name = (
                f"{model_name}_{dataset_name}_final_{index}.pth"
            )

        save_model(
            model=self._model,
            filename=self._config.save_model_name,
            filepath=self._config.save_model_path,
            distributed=self._config.use_ddp,
        )

    def _run_batch(self, source: torch.Tensor, targets: torch.Tensor) -> float:
        """
        Runs the given batch.

        Args:
            source (torch.Tensor): The source.
            targets (torch.Tensor): The targets.

        Returns:
            float: The loss.
        """
        self._model.train()
        self._optimizer.zero_grad()
        output = self._model(source)

        if hasattr(output, "logits"):
            output = output.logits

        loss = self._loss_fn(output, targets)
        loss.backward()
        self._optimizer.step()
        if self._scheduler:
            self._scheduler.step()
        return loss.item()

    def _run_epoch(self, epoch: int) -> None:
        """
        Runs the given epoch.
        """
        total_loss = 0.0
        for _, (source, targets) in enumerate(
            tqdm(self._train_loader, leave=False)
        ):
            source, targets = source.to(self._device), targets.to(self._device)
            loss = self._run_batch(source, targets)
            total_loss += loss

        total_loss /= len(self._train_loader)
        click.echo(
            click.style(f"Epoch {epoch} - Average loss: {total_loss:.4f}", fg="blue")
        )
        self._log_loss(epoch, total_loss)

    def _pre_training(self) -> None:
        # Method to run before training starts.
        self._model.train()

    def _post_training(self) -> None:
        # Method to run after training ends.
        if self._should_save_final_model():
            self._save_final_model()

        if self._privacy_engine:
            delta = self._config.differential_privacy.delta
            epsilon = self._privacy_engine.get_epsilon(delta)
            click.echo(
                click.style(
                    f"\nTraining finished. Final privacy budget: (ε = {epsilon:.2f}, δ = {delta})",
                    fg="green",
                )
            )

    def _log_loss(
        self, epoch: int, loss: float, dir: str = None, filename: str = "loss.log"
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
        load_checkpoint: bool, checkpoint_path: Optional[str], device: torch.device
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
        if not load_checkpoint or not checkpoint_path:
            return None

        if not os.path.isfile(checkpoint_path):
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