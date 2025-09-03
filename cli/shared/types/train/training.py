from dataclasses import dataclass
from typing import Optional

from advsecurenet.shared.types.configs.device_config import DeviceConfig
from cli.shared.types.utils.dataloader import DataLoaderCliConfigType
from advsecurenet.shared.types.configs.dataset_config import CreateDatasetCliConfig
from cli.shared.types.utils.model import ModelCliConfigType


@dataclass
class TrainingHyperparameter:
    """
    CLI configuration for training hyperparameters.
    Uses the same structure as the shared base for consistency.
    """

    criterion: str = "cross_entropy"
    epochs: int = 10
    learning_rate: float = 0.001
    verbose: bool = False


@dataclass
class Optimization:
    """
    CLI configuration for optimization.
    Uses the same structure as the shared base for consistency.
    """

    optimizer: str = "adam"
    optimizer_kwargs: Optional[dict] = None
    scheduler: Optional[str] = None
    scheduler_kwargs: Optional[dict] = None


@dataclass
class Checkpoint:
    """
    CLI configuration for checkpoints.
    Uses the same structure as the shared base for consistency.
    """

    save_checkpoint: bool = False
    save_checkpoint_path: Optional[str] = None
    save_checkpoint_name: Optional[str] = None
    checkpoint_interval: int = 1
    load_checkpoint: bool = False
    load_checkpoint_path: Optional[str] = None


@dataclass
class FinalModel:
    """
    CLI configuration for final model saving.
    Uses the same structure as the shared base for consistency.
    """

    save_final_model: bool = False
    save_model_path: Optional[str] = None
    save_model_name: Optional[str] = None


@dataclass
class DifferentialPrivacy:
    """
    CLI configuration for differential privacy.
    Uses the same structure as the shared base for consistency.
    """

    enable: bool = False
    noise_multiplier: float = 1.0
    max_grad_norm: float = 1.0
    delta: float = 1e-5
    kwargs: Optional[dict] = None


@dataclass
class Training:
    """
    This dataclass is used to store the configuration of the training.
    """

    training_hyperparameter: TrainingHyperparameter
    optimization: Optimization
    checkpoint: Checkpoint
    final_model: FinalModel
    differential_privacy: Optional[DifferentialPrivacy] = None
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
