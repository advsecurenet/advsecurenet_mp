from typing import Optional, Tuple

from torch.utils.data import Dataset as TorchDataset

from advsecurenet.datasets import DatasetFactory
from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.shared.types.dataset import DatasetType


from cli.shared.types.utils.dataset import (
    CreateDatasetCliConfig,
)

from cli.shared.types.utils.dataset import CreateDatasetCliConfig, resolve_dataset_config


def get_datasets(
    config: CreateDatasetCliConfig, **kwargs
) -> Tuple[Optional[TorchDataset], Optional[TorchDataset]]:
    """
    High-level utility to load train and test datasets based on a CLI configuration object.

    This function acts as a simple bridge between the CLI configuration and the
    core dataset factory. It resolves the user-facing config and then delegates
    the loading process to the DatasetFactory.

    Args:
        config (CreateDatasetCliConfig): The user-facing configuration object.
        **kwargs: Runtime keyword arguments that will override any other settings.

    Returns:
        A tuple containing the loaded train and test datasets, respectively.
    """
    # 1. Resolve the user-facing config into a structured, internal config
    # Note: You will need a resolver function in your types file.
    resolved_config = resolve_dataset_config(config)

    # 2. Delegate the entire loading process to the factory
    loaded_datasets = DatasetFactory.load_dataset_from_config(
        resolved_config=resolved_config,
        **kwargs
    )

    # 3. Extract train and test data to return as a tuple for backward compatibility
    train_data = loaded_datasets.get("train")
    test_data = loaded_datasets.get("test")

    return train_data, test_data
