from dataclasses import dataclass
from typing import Optional

from advsecurenet.shared.types.configs.device_config import DeviceConfig
from cli.shared.types.utils.dataloader import DataLoaderCliConfigType
from advsecurenet.shared.types.configs.dataset_config import CreateDatasetCliConfig
from cli.shared.types.utils.model import ModelCliConfigType
from advnet_common.types.configs.base import (
    TrainingHyperparametersBase,
    OptimizationBase,
    CheckpointBase,
    FinalModelBase,
    DifferentialPrivacyBase,
)


@dataclass
class Training:
    """
    This dataclass is used to store the configuration of the training.
    Uses base classes directly since no CLI-specific customization is needed.
    """

    training_hyperparameter: TrainingHyperparametersBase
    optimization: OptimizationBase
    checkpoint: CheckpointBase
    final_model: FinalModelBase
    differential_privacy: Optional[DifferentialPrivacyBase] = None
    verbose: bool = False


@dataclass
class TrainingCliConfigType:
    """
    This dataclass is used to store the configuration of the training CLI.
    """

    model: ModelCliConfigType
    dataset: CreateDatasetCliConfig
    dataloader: DataLoaderCliConfigType
    training: Training
    device: DeviceConfig
