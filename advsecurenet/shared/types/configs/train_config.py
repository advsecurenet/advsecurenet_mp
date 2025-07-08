from dataclasses import dataclass, field
from typing import Optional, Union, Any

from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

from advsecurenet.shared.types.configs.device_config import DeviceConfig


@dataclass
class ModelConfig:
    """
    Configuration class for the model.
    """

    model: nn.Module


@dataclass
class TrainingProcessConfig:
    """
    Configuration class for the training process.
    """

    train_loader: DataLoader
    criterion: Union[str, nn.Module] = "cross_entropy"
    epochs: int = 10
    learning_rate: float = 0.001
    verbose: bool = False


@dataclass
class OptimizationConfig:
    """
    Configuration class for the optimization process.
    """

    optimizer: Union[str, Optimizer] = "adam"
    optimizer_kwargs: Optional[dict[str, Any]] = None
    scheduler: Optional[Union[str, lr_scheduler._LRScheduler]] = None
    scheduler_kwargs: Optional[dict] = None

@dataclass
class DifferentialPrivacyConfig:
    """
    Configuration for Differential Privacy using Opacus.
    """
    enable: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    kwargs: Optional[dict] = None


@dataclass
class CheckpointConfig:
    """
    Configuration class for the checkpoint.
    """

    save_checkpoint: bool = False
    save_checkpoint_path: Optional[str] = None
    save_checkpoint_name: Optional[str] = None
    checkpoint_interval: int = 1
    load_checkpoint: bool = False
    load_checkpoint_path: Optional[str] = None


@dataclass
class FinalModelConfig:
    """
    Configuration class for the final model.
    """

    save_final_model: bool = False
    save_model_path: Optional[str] = None
    save_model_name: Optional[str] = None


@dataclass
class TrainConfig:
    """
    Dataclass to store the overall training configuration by aggregating other configurations.
    """
    model_config: ModelConfig
    training_process_config: TrainingProcessConfig
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    checkpoint_config: CheckpointConfig = field(default_factory=CheckpointConfig)
    final_model_config: FinalModelConfig = field(default_factory=FinalModelConfig)
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    differential_privacy: Optional[DifferentialPrivacyConfig] = None
