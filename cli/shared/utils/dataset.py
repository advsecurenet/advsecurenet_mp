from typing import Optional, Tuple, Dict, Any
import warnings

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
    loaded_datasets = DatasetFactory.load_from_config(
        resolved_config=resolved_config,
        **kwargs
    )

    # 3. Extract train and test data to return as a tuple for backward compatibility
    train_data = loaded_datasets.get("train")
    test_data = loaded_datasets.get("test")

    return train_data, test_data

def infer_dataset_type(dataset_identifier: str) -> DatasetType:
    return NotImplementedError()


def _merge_dicts_with_warning(
    base_dict: Dict[str, Any],
    override_dict: Dict[str, Any],
    warning_template: str
) -> Dict[str, Any]:
    """
    Merges an override dictionary into a copy of a base dictionary,
    issuing a formatted warning on any conflicts. Returns a new dictionary.
    
    Args:
        base_dict (Dict): The dictionary with default values.
        override_dict (Dict): The dictionary with new values to merge.
        warning_template (str): A format-string for the warning message.

    Returns:
        Dict[str, Any]: A new dictionary containing the merged result.
    """
    merged = base_dict.copy()
    for key, value in override_dict.items():
        if key in merged:
            warnings.warn(
                warning_template.format(
                    key=key,
                    old_value=merged.get(key),
                    new_value=value
                )
            )
        merged[key] = value
    return merged

def _prepare_load_kwargs(
    base_kwargs: Dict[str, Any],
    keys_to_override: Dict[str, Any],
    runtime_kwargs: Dict[str, Any],
    logical_name: str
) -> Dict[str, Any]:
    """
    Prepares the final keyword arguments for the load_dataset method by merging
    different sources of arguments and warning on conflicts.
    
    Args:
        base_kwargs (Dict): The initial kwargs from the configuration.
        keys_to_override (Dict): Dictionary of keys that must be set (e.g., {'dataset_name': 'cifar10'}).
        runtime_kwargs (Dict): Additional kwargs passed at runtime.
        logical_name (str): The logical name of the split (e.g., 'train') for warning messages.

    Returns:
        Dict[str, Any]: The final, processed dictionary of keyword arguments.
    """
    # 1. Start with the base kwargs and override with resolved config keys
    load_kwargs = _merge_dicts_with_warning(
        base_dict=base_kwargs,
        override_dict=keys_to_override,
        warning_template=(
            f"The 'dataset_kwargs' for split '{logical_name}' contains a '{{key}}' key. "
            f"It will be overridden by the resolved value '{{new_value}}'."
        )
    )

    # 2. Take the result and override with runtime kwargs from the CLI call
    final_kwargs = _merge_dicts_with_warning(
        base_dict=load_kwargs,
        override_dict=runtime_kwargs,
        warning_template=(
            f"Runtime argument '{{key}}' is overriding a configuration value for split '{logical_name}'. "
            f"Old: '{{old_value}}', New: '{{new_value}}'"
        )
    )
    
    return final_kwargs