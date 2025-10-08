import torch
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm, trange

from advsecurenet.distributed.ddp_base_task import DDPBaseTask
from advsecurenet.shared.types.configs.train_config import TrainConfig
from advsecurenet.trainer.trainer import Trainer
from advsecurenet.trainer import trainer_logic


class DDPTrainer(DDPBaseTask, Trainer):
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
        self._rank = rank
        self._world_size = world_size
        DDPBaseTask.__init__(
            self, model=config.model_config.model, rank=rank, world_size=world_size
        )
        Trainer.__init__(self, config)

    def _load_model_state_dict(self, state_dict):
        """
        Loads the given model state dict.
        """
        self.model.module.load_state_dict(state_dict)

    def _get_model_state_dict(self) -> dict:
        # Returns the model state dict.
        return self.model.module.state_dict()

    def _assign_device_to_optimizer_state(self):
        """
        Assigns the optimizer state tensors to the appropriate CUDA device based on rank.
        """
        if hasattr(self, "optimizer") and self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda(self._rank)

    def _get_save_checkpoint_prefix(self) -> str:
        """
        Returns the save checkpoint prefix.

        Returns:
            str: The save checkpoint prefix.

        Notes:
            If the save checkpoint name is provided, it will be used as the prefix. Otherwise, the model variant and the dataset name will be used as the prefix.
        """

        if self._config.checkpoint_config.save_checkpoint_name:
            return self._config.checkpoint_config.save_checkpoint_name
        else:
            return f"{self._config.model_config.model.model_name}_{self._config.training_process_config.train_loader.dataset.__class__.__name__}_checkpoint"

    def _should_save_checkpoint(self, epoch: int) -> bool:
        """
        Determines if a checkpoint should be saved based on the given epoch, the checkpoint interval and the current rank.
        Args:
            epoch (int): The current epoch.
        Returns:
            bool: True if a checkpoint should be saved, False otherwise.
        """
        return (
            self._rank == 0
            and self._config.checkpoint_config.save_checkpoint
            and self._config.checkpoint_config.checkpoint_interval > 0
            and epoch % self._config.checkpoint_config.checkpoint_interval == 0
        )

    def _should_save_final_model(self) -> bool:
        """
        Determines if the final model should be saved based on the given save_final_model flag and the current rank.
        """
        return self._rank == 0 and self._config.final_model_config.save_final_model

    def _run_epoch(self, epoch: int) -> None:
        """
        DDP-specific epoch running with distributed sampler handling.
        """
        # Set epoch for distributed sampler
        sampler = self._config.training_process_config.train_loader.sampler
        if isinstance(sampler, DistributedSampler):
            sampler.set_epoch(epoch)

        total_loss = 0.0

        # Only show tqdm on rank 0 to avoid cluttered output
        if self._rank == 0:
            data_iterator = tqdm(self._train_loader, leave=False, position=1)
        else:
            data_iterator = self._train_loader

        # Use trainer_logic for actual batch processing
        for source, targets in data_iterator:
            source, targets = source.to(self._device), targets.to(self._device)
            loss = trainer_logic.run_batch(
                source,
                targets,
                self.model,
                self.optimizer,
                self._loss_fn,
                self._scheduler,
            )
            total_loss += loss

        # DDP-specific: divide by world_size for proper averaging across processes
        total_loss /= len(self._train_loader) * self._world_size

        # Only log on rank 0
        if self._rank == 0:
            trainer_logic.log_loss(epoch, total_loss)

    def _post_training(self) -> None:
        """
        DDP-specific: Only save final model on rank 0.
        """
        if self._rank == 0:
            trainer_logic.post_training(
                save_final_model_flag=self._config.final_model_config.save_final_model,
                model=self.model,
                save_path=self._config.final_model_config.save_model_path,
                save_name=self._config.final_model_config.save_model_name,
                model_name=None,
                dataset_name=None,
                use_ddp=True,  # Set to True for DDP
                privacy_engine=self._privacy_engine,
                delta=(
                    self._config.differential_privacy_config.delta
                    if self._config.differential_privacy_config
                    else None
                ),
            )

    def _get_checkpoint_path(self, epoch: int) -> str:
        """
        DDP-specific: Generate checkpoint path (same as base but for clarity).
        """
        from advsecurenet.trainer import trainer_logic

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
        DDP-specific: Only save checkpoint on rank 0.
        """
        if self._rank == 0:
            from advsecurenet.trainer import trainer_logic

            trainer_logic.save_checkpoint(
                epoch, self.optimizer, self.model, checkpoint_path
            )
