from typing import Optional

import click
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from advsecurenet.utils.model_utils import save_model

# This file contains the pure, stateless logic functions for the training process.

def run_batch(source: torch.Tensor, targets: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, loss_fn: nn.Module, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]) -> float:
    """Runs a single batch."""
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

def run_epoch(epoch: int, train_loader: DataLoader, device: torch.device, model: nn.Module, optimizer: optim.Optimizer, loss_fn: nn.Module, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler]) -> None:
    """Runs a single training epoch."""
    total_loss = 0.0
    for _, (source, targets) in enumerate(tqdm(train_loader, leave=False)):
        source, targets = source.to(device), targets.to(device)
        loss = run_batch(source, targets, model, optimizer, loss_fn, scheduler)
        total_loss += loss

    total_loss /= len(train_loader)
    click.echo(click.style(f"Epoch {epoch} - Average loss: {total_loss:.4f}", fg="blue"))
    # You might move _log_loss here as well

def should_save_checkpoint(epoch: int, save_checkpoint: bool, checkpoint_interval: int) -> bool:
    """Determines if a checkpoint should be saved."""
    return save_checkpoint and checkpoint_interval > 0 and epoch % checkpoint_interval == 0

def save_checkpoint(epoch: int, optimizer: optim.Optimizer, model: nn.Module, checkpoint_path: str) -> None:
    """Saves the checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    click.echo(click.style(f"Saved checkpoint to {checkpoint_path}", fg="green"))

def post_training(save_final_model_flag: bool, model: nn.Module, save_path: str, save_name: str, model_name: str, dataset_name: str, use_ddp: bool, privacy_engine, delta) -> None:
    """Logic to run after training ends."""
    if save_final_model_flag:
        # Assuming _save_final_model logic is also moved here
        pass # save_final_model(...)

    if privacy_engine:
        epsilon = privacy_engine.get_epsilon(delta)
        click.echo(
            click.style(
                f"\nTraining finished. Final privacy budget: (ε = {epsilon:.2f}, δ = {delta})",
                fg="green",
            )
        )