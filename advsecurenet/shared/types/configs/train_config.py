from dataclasses import dataclass, field
from typing import Optional, Union, Any

from torch import nn
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader

from advsecurenet.shared.types.configs.device_config import DeviceConfig
from shared.types.configs.base import (
    TrainingHyperparametersBase,
    OptimizationBase,
    CheckpointBase,
    FinalModelBase,
    DifferentialPrivacyBase,
)


@dataclass
class ModelConfig:
    """
    Configuration class for the model.
    """

    model: nn.Module


@dataclass
class TrainingProcessConfig(TrainingHyperparametersBase):
    """
    Configuration class for the training process.
    Inherits from shared base and adds train_loader.
    """

    train_loader: Optional[DataLoader] = None


@dataclass
class TrainConfig:
    """
    Dataclass to store the overall training configuration by aggregating other configurations.
    Uses base classes directly where no additional fields are needed.
    """

    model_config: ModelConfig
    training_process_config: TrainingProcessConfig
    optimization_config: OptimizationBase = field(default_factory=OptimizationBase)
    checkpoint_config: CheckpointBase = field(default_factory=CheckpointBase)
    final_model_config: FinalModelBase = field(default_factory=FinalModelBase)
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    differential_privacy_config: Optional[DifferentialPrivacyBase] = None
