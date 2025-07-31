"""
Base configuration classes for training-related configurations.
These classes define the common structure shared between CLI and core modules.
"""

from dataclasses import dataclass
from typing import Optional, Union, Any

from torch import nn
from torch.optim import Optimizer, lr_scheduler


@dataclass
class TrainingHyperparametersBase:
    """Base class for training hyperparameters configuration."""
    criterion: Union[str, nn.Module] = "cross_entropy"
    epochs: int = 10
    learning_rate: float = 0.001
    verbose: bool = False


@dataclass  
class OptimizationBase:
    """Base class for optimization configuration."""
    optimizer: Union[str, Optimizer] = "adam"
    optimizer_kwargs: Optional[dict[str, Any]] = None
    scheduler: Optional[Union[str, lr_scheduler._LRScheduler]] = None
    scheduler_kwargs: Optional[dict] = None


@dataclass
class CheckpointBase:
    """Base class for checkpoint configuration."""
    save_checkpoint: bool = False
    save_checkpoint_path: Optional[str] = None
    save_checkpoint_name: Optional[str] = None
    checkpoint_interval: int = 1
    load_checkpoint: bool = False
    load_checkpoint_path: Optional[str] = None


@dataclass
class FinalModelBase:
    """Base class for final model saving configuration."""
    save_final_model: bool = False
    save_model_path: Optional[str] = None
    save_model_name: Optional[str] = None


@dataclass
class DifferentialPrivacyBase:
    """Base class for differential privacy configuration."""
    enable: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    kwargs: Optional[dict] = None
