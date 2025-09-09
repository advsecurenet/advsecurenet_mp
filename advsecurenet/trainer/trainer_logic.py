from typing import Optional, Union, cast
import logging
import os
import multiprocessing
import time
import gc

import multiprocessing as mp
import signal
from contextlib import contextmanager

import click
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from advsecurenet.shared.optimizer import Optimizer
from advsecurenet.shared.scheduler import Scheduler
from advsecurenet.utils.model_utils import save_model

logger = logging.getLogger(__name__)

# This file contains the pure, stateless logic functions for the training process.


def run_batch(
    source: torch.Tensor,
    targets: torch.Tensor,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    scheduler: Optional[lr_scheduler.LRScheduler],
) -> float:
    """
    Runs a single batch of training.

    Args:
        source (torch.Tensor): Input data batch.
        targets (torch.Tensor): Target labels batch.
        model (nn.Module): The neural network model.
        optimizer (optim.Optimizer): The optimizer for updating model parameters.
        loss_fn (nn.Module): The loss function to compute the training loss.
        scheduler (Optional[lr_scheduler.LRScheduler]): Optional learning rate scheduler.

    Returns:
        float: The loss value for this batch.
    """
    model.train()
    optimizer.zero_grad()
    output = model(source)

    if hasattr(output, "logits"):
        output = output.logits

    loss = loss_fn(output, targets)
    loss.backward()
    optimizer.step()
    if scheduler:
        scheduler.step()
    return loss.item()


def run_epoch(
    epoch: int,
    train_loader: DataLoader,
    device: torch.device,
    model: nn.Module,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    scheduler: Optional[lr_scheduler.LRScheduler],
) -> None:
    """
    Runs a single training epoch using an explicit iterator to ensure DataLoader worker cleanup.

    Args:
        epoch (int): The current epoch number.
        train_loader (DataLoader): The training data loader.
        device (torch.device): The device to run training on.
        model (nn.Module): The neural network model.
        optimizer (optim.Optimizer): The optimizer for updating model parameters.
        loss_fn (nn.Module): The loss function to compute the training loss.
        scheduler (Optional[lr_scheduler.LRScheduler]): Optional learning rate scheduler.

    Returns:
        None
    """
    total_loss = 0.0
    loader_length = len(train_loader)  # Save length before cleanup

    # Use explicit iterator for robust DataLoader worker cleanup
    data_iter = iter(train_loader)
    for _, (source, targets) in enumerate(tqdm(data_iter, leave=False)):
        source, targets = source.to(device), targets.to(device)
        loss = run_batch(source, targets, model, optimizer, loss_fn, scheduler)
        total_loss += loss
    del data_iter  # Explicitly delete iterator to trigger worker shutdown
    total_loss /= loader_length
    click.echo(
        click.style(f"Epoch {epoch} - Average loss: {total_loss:.4f}", fg="blue")
    )


def should_save_checkpoint(
    epoch: int, save_checkpoint: bool, checkpoint_interval: int
) -> bool:
    """
    Determines if a checkpoint should be saved.

    Args:
        epoch (int): The current epoch number.
        save_checkpoint (bool): Whether checkpoint saving is enabled.
        checkpoint_interval (int): The interval at which checkpoints should be saved.

    Returns:
        bool: True if a checkpoint should be saved, False otherwise.
    """
    return (
        save_checkpoint and checkpoint_interval > 0 and epoch % checkpoint_interval == 0
    )


def save_checkpoint(
    epoch: int, optimizer: optim.Optimizer, model: nn.Module, checkpoint_path: str
) -> None:
    """
    Saves the checkpoint.

    Args:
        epoch (int): The current epoch number.
        optimizer (optim.Optimizer): The optimizer state to save.
        model (nn.Module): The model state to save.
        checkpoint_path (str): The path where the checkpoint should be saved.

    Returns:
        None
    """
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    click.echo(click.style(f"Saved checkpoint to {checkpoint_path}", fg="green"))


def post_training(
    save_final_model_flag: bool,
    model: nn.Module,
    save_path: Optional[str],
    save_name: Optional[str],
    model_name: Optional[str],
    dataset_name: Optional[str],
    use_ddp: bool,
    privacy_engine,
    delta,
) -> None:
    """
    Logic to run after training ends.

    Args:
        save_final_model_flag (bool): Whether to save the final model.
        model (nn.Module): The trained model.
        save_path (Optional[str]): Path where to save the final model.
        save_name (Optional[str]): Custom name for the saved model.
        model_name (Optional[str]): Name of the model architecture.
        dataset_name (Optional[str]): Name of the dataset used for training.
        use_ddp (bool): Whether distributed data parallel was used.
        privacy_engine: The privacy engine used for differential privacy.
        delta: The delta parameter for differential privacy.

    Returns:
        None
    """
    if save_final_model_flag:
        save_final_model(model, save_path, save_name, model_name, dataset_name, use_ddp)

    if privacy_engine:
        epsilon = privacy_engine.get_epsilon(delta)
        click.echo(
            click.style(
                f"\nTraining finished. Final privacy budget: (ε = {epsilon:.2f}, δ = {delta})",
                fg="green",
            )
        )


# Optimizer and Scheduler utilities


def create_scheduler_from_string(
    scheduler_name: str,
    optimizer: optim.Optimizer,
    scheduler_kwargs: Optional[dict] = None,
) -> lr_scheduler.LRScheduler:
    """
    Creates a scheduler instance from a string name.

    Args:
        scheduler_name (str): The name of the scheduler.
        optimizer (optim.Optimizer): The optimizer for the scheduler.
        scheduler_kwargs (Optional[dict]): Additional keyword arguments for the scheduler.

    Returns:
        lr_scheduler.LRScheduler: The scheduler instance.

    Raises:
        ValueError: If the scheduler name is not supported.
    """
    # Mapping for common scheduler name variations to standardized enum names
    scheduler_name_mapping = {
        "STEPLR": "STEP_LR",
        "MULTISTEPLR": "MULTI_STEP_LR",
        "COSINEANNEALINGLR": "COSINE_ANNEALING_LR",
        "CYCLICLR": "CYCLIC_LR",
        "ONECYCLELR": "ONE_CYCLE_LR",
        "COSINEANNEALINGWARMRESTARTS": "COSINE_ANNEALING_WARM_RESTARTS",
        "LAMBDALR": "LAMBDA_LR",
        "POLYLR": "POLY_LR",
        "LINEARLR": "LINEAR_LR",
        "REDUCELRONPLATEAU": "REDUCE_LR_ON_PLATEAU",
    }

    # Normalize scheduler name and map to standard form
    normalized_name = scheduler_name.upper()
    normalized_name = scheduler_name_mapping.get(normalized_name, normalized_name)

    if normalized_name not in Scheduler.__members__:
        raise ValueError(
            "Unsupported scheduler! Choose from: "
            + ", ".join([e.name for e in Scheduler])
        )

    scheduler_function_class = Scheduler[normalized_name].value
    kwargs = scheduler_kwargs or {}
    return cast(lr_scheduler.LRScheduler, scheduler_function_class(optimizer, **kwargs))


def get_scheduler(
    scheduler: Optional[Union[str, lr_scheduler.LRScheduler]],
    optimizer: optim.Optimizer,
    scheduler_kwargs: Optional[dict] = None,
) -> Optional[lr_scheduler.LRScheduler]:
    """
    Returns the scheduler based on the given scheduler configuration.

    Args:
        scheduler (str or lr_scheduler.LRScheduler, optional): The scheduler configuration.
        optimizer (optim.Optimizer): The optimizer for the scheduler.
        scheduler_kwargs (Optional[dict]): Additional keyword arguments for the scheduler.

    Returns:
        lr_scheduler.LRScheduler: The scheduler instance, or None if no scheduler specified.
    """
    if scheduler is None:
        return None

    if isinstance(scheduler, str):
        return create_scheduler_from_string(scheduler, optimizer, scheduler_kwargs)
    elif isinstance(scheduler, lr_scheduler.LRScheduler):
        return scheduler
    else:
        raise ValueError(
            "Scheduler must be a string or an instance of lr_scheduler.LRScheduler."
        )


def get_optimizer(
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
        >>> get_optimizer("adam", model)
        >>> get_optimizer(optim.Adam(model.parameters(), lr=0.001), model)
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


def assign_device_to_optimizer_state(
    optimizer: optim.Optimizer, device: torch.device
) -> None:
    """
    Assigns the specified device to all tensors in the optimizer state.

    Args:
        optimizer (optim.Optimizer): The optimizer whose state should be moved to the device.
        device (torch.device): The target device.

    Returns:
        None
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


# Checkpoint utilities


def get_save_checkpoint_prefix(
    save_checkpoint_name: Optional[str], model_name: str, dataset_name: str
) -> str:
    """
    Returns the save checkpoint prefix.

    Args:
        save_checkpoint_name (Optional[str]): Custom checkpoint name.
        model_name (str): Name of the model.
        dataset_name (str): Name of the dataset.

    Returns:
        str: The save checkpoint prefix.

    Notes:
        If the save checkpoint name is provided, it will be used as the prefix.
        Otherwise, the model name and the dataset name will be used as the prefix.
    """
    if save_checkpoint_name:
        return save_checkpoint_name
    else:
        return f"{model_name}_{dataset_name}_checkpoint"


def define_save_checkpoint_path(
    save_checkpoint_path: Optional[str],
    save_checkpoint_name: Optional[str],
    checkpoint_sub_dir: Optional[str],
    model_name: str,
    dataset_name: str,
    epoch: int,
) -> str:
    """
    Defines the full path for saving a checkpoint.

    Args:
        save_checkpoint_path (Optional[str]): Base directory for checkpoints.
        save_checkpoint_name (Optional[str]): Custom checkpoint name.
        checkpoint_sub_dir (Optional[str]): Subdirectory for checkpoints.
        model_name (str): Name of the model.
        dataset_name (str): Name of the dataset.
        epoch (int): Current epoch number.

    Returns:
        str: The full path for saving the checkpoint.
    """
    checkpoint_dir = save_checkpoint_path or os.path.join(
        os.getcwd(), f"checkpoints/{checkpoint_sub_dir}"
    )

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    save_checkpoint_prefix = get_save_checkpoint_prefix(
        save_checkpoint_name, model_name, dataset_name
    )
    checkpoint_filename = f"{save_checkpoint_prefix}_epoch_{epoch}.pth"

    return os.path.join(checkpoint_dir, checkpoint_filename)


def save_final_model(
    model_to_save: nn.Module,
    save_path: Optional[str],
    save_name: Optional[str],
    model_name: Optional[str],
    dataset_name: Optional[str],
    use_ddp: bool,
) -> None:
    """
    Saves the final model to a specified path, handling filename collisions.

    Args:
        model_to_save (nn.Module): The model to save.
        save_path (Optional[str]): Directory to save the model.
        save_name (Optional[str]): Custom name for the saved model.
        model_name (Optional[str]): Name of the model.
        dataset_name (Optional[str]): Name of the dataset.
        use_ddp (bool): Whether DDP is being used.

    Returns:
        None
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


def log_loss(
    epoch: int, loss: float, dir: Optional[str] = None, filename: str = "loss.log"
) -> None:
    """
    Logs the loss for a given epoch to a file.

    Args:
        epoch (int): The epoch number.
        loss (float): The loss value.
        dir (Optional[str]): Directory to save the log file.
        filename (str): Name of the log file.

    Returns:
        None
    """
    path = os.path.join(dir, filename) if dir else os.path.join(os.getcwd(), filename)
    # Save the loss to the log file. If the log file does not exist, create it in the current directory.
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("epoch,loss\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{epoch},{loss}\n")


def load_checkpoint_data(
    checkpoint_path: Optional[str], device: torch.device
) -> Optional[dict]:
    """
    Loads checkpoint data from a file.

    Args:
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
        if (
            "model_state_dict" not in checkpoint
            or "optimizer_state_dict" not in checkpoint
            or "epoch" not in checkpoint
        ):
            logger.error(
                "Checkpoint is malformed or missing required keys. Not loading."
            )
            return None
        return checkpoint
    except Exception as e:
        logger.error("Failed to load checkpoint file from %s: %s", checkpoint_path, e)
        return None
