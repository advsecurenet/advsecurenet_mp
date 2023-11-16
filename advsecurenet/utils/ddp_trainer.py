import torch
import torch.optim as optim
import os
import pkg_resources
import requests
from typing import Optional, cast, Any, Union, cast
from torch import nn
from tqdm import tqdm
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from advsecurenet.shared.types.configs.defense_configs.adversarial_training_config import AdversarialTrainingConfig
from advsecurenet.shared.types.configs.train_config import TrainConfig
from advsecurenet.shared.loss import Loss
from advsecurenet.shared.optimizer import Optimizer
from advsecurenet.utils.trainer import Trainer
from torch.utils.data.distributed import DistributedSampler


class DDPTrainer(Trainer):
    """
    DDPTrainer module is specialized module for training a model using DistributedDataParallel in a multi-GPU setting.

    Args:
        config (TrainConfig): The train config.
        rank (int): The rank of the current process.
        world_size (int): The total number of processes.

    Examples:

            >>> trainer = DDPTrainer(config, rank, world_size)
            >>> trainer.train()

    """

    def __init__(self, config: TrainConfig, rank: int, world_size: int) -> None:
        self.rank = rank
        self.world_size = world_size
        super().__init__(config)

    def _setup_device(self) -> torch.device:
        """
        Initializes the device based on the rank of the current process.

        Returns:
            torch.device: The device.
        """
        torch.cuda.set_device(self.rank)
        return torch.device(f"cuda:{self.rank}")

    def _setup_model(self) -> torch.nn.parallel.DistributedDataParallel:
        """
        Initializes the model based on the rank of the current process.

        Returns:
            DistributedDataParallel: The model.
        """
        model = self.config.model.to(self.device)
        return DDP(model, device_ids=[self.rank])

    def _load_model_state_dict(self, state_dict):
        """
        Loads the given model state dict.
        """
        self.model.module.load_state_dict(state_dict)

    def _get_model_state_dict(self) -> dict:
        # Returns the model state dict.
        return self.model.module.state_dict()

    def _assign_device_to_optimizer_state(self):
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(self.rank)

    def _get_save_checkpoint_prefix(self) -> str:
        """
        Returns the save checkpoint prefix.

        Returns:
            str: The save checkpoint prefix.

        Notes:
            If the save checkpoint name is provided, it will be used as the prefix. Otherwise, the model variant and the dataset name will be used as the prefix.
        """

        if self.config.save_checkpoint_name:
            return self.config.save_checkpoint_name
        else:
            return f"{self.config.model.model_variant}_{self.config.train_loader.dataset.__class__.__name__}_checkpoint"

    def _should_save_checkpoint(self, epoch: int) -> bool:
        """
        Determines if a checkpoint should be saved based on the given epoch, the checkpoint interval and the current rank.
        Args:
            epoch (int): The current epoch.
        Returns:
            bool: True if a checkpoint should be saved, False otherwise.  
        """
        return self.rank == 0 and self.config.save_checkpoint and self.config.checkpoint_interval > 0 and epoch % self.config.checkpoint_interval == 0

    def _should_save_final_model(self) -> bool:
        """
        Determines if the final model should be saved based on the given save_final_model flag and the current rank.
        """
        return self.rank == 0 and self.config.save_final_model

    def _run_epoch(self, epoch: int) -> None:
        """
        Runs the given epoch.
        """
        total_loss = 0.0
        sampler = self.config.train_loader.sampler
        assert isinstance(
            sampler, DistributedSampler), "Sampler must be of type DistributedSampler"
        sampler.set_epoch(epoch)

        # for source, targets in self.config.train_loader:
        for batch_idx, (source, targets) in enumerate(tqdm(self.config.train_loader)):
            source, targets = source.to(self.device), targets.to(self.device)
            loss = self._run_batch(source, targets)
            total_loss += loss

        total_loss /= len(self.config.train_loader)
        print(f"Epoch {epoch} loss: {total_loss}")
