from typing import Optional, Dict, Any
import warnings

from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.datasets.Cifar10 import CIFAR10Dataset, CIFAR100Dataset
from advsecurenet.datasets.Custom import CustomDataset
from advsecurenet.datasets.ImageNet import ImageNetDataset
from advsecurenet.datasets.MNIST import FashionMNISTDataset, MNISTDataset
from advsecurenet.datasets.svhn import SVHNDataset
from advsecurenet.datasets.HuggingFace import HuggingFaceDataset
from advsecurenet.shared.types import DatasetType
from cli.shared.types.utils.dataset import ResolvedDatasetConfig, ResolvedSplitConfig
from advsecurenet.utils.huggingface_utils import huggingface_dataset_utils
from advsecurenet.utils.huggingface_utils import huggingface_general_utils
from advsecurenet.utils.kwargs_utils import filter_kwargs_for_callable

DATASET_MAP = {
    DatasetType.CIFAR10: CIFAR10Dataset,
    DatasetType.IMAGENET: ImageNetDataset,
    DatasetType.MNIST: MNISTDataset,
    DatasetType.FASHION_MNIST: FashionMNISTDataset,
    DatasetType.CIFAR100: CIFAR100Dataset,
    DatasetType.SVHN: SVHNDataset,
    DatasetType.CUSTOM: CustomDataset,
    DatasetType.HUGGINGFACE: HuggingFaceDataset
}


class DatasetFactory:
    """
    A factory class to create and load datasets based on configuration.
    """

    @staticmethod
    def load_from_config(
        resolved_config: ResolvedDatasetConfig,
        **runtime_kwargs
    ) -> Dict[str, Optional[BaseDataset]]:
        """
        The main entry point for loading datasets from a resolved configuration.
        """
        loaded_datasets: Dict[str, Optional[BaseDataset]] = {}

        for logical_name, split_config in resolved_config.splits.items():
            try:
                # 1. Create the dataset provider instance with constructor args from the config
                dataset_type = _infner_dataset_type(split_config.identifier)
                
                if dataset_type == DatasetType.HUGGINGFACE:
                    identifier = huggingface_general_utils.process_hf_identifier(split_config.identifier)
                else:
                    identifier = split_config.identifier

                dataset_provider = DatasetFactory._create_provider(split_config, dataset_type)

                # 2. Prepare kwargs for the `load_dataset` method
                keys_to_override = {
                    'dataset_name': identifier,
                    'split': split_config.source_split_name,
                    'root': getattr(split_config, 'path', None),
                    'download': getattr(resolved_config, 'download', True)
                }

                load_kwargs = _prepare_load_kwargs(
                    base_kwargs=split_config.kwargs,
                    keys_to_override=keys_to_override,
                    runtime_kwargs=runtime_kwargs,
                    logical_name=logical_name
                )

                # 3. Process kwargs and load the dataset
                processed_load_kwargs = dataset_provider.process_kwargs_load_dataset(load_kwargs)
                dataset = dataset_provider.load_dataset(**processed_load_kwargs)
                loaded_datasets[logical_name] = dataset

            except Exception as e:
                print(f"Warning: Could not load dataset for split '{logical_name}'. Error: {e}")
                loaded_datasets[logical_name] = None
        
        return loaded_datasets
    
    @staticmethod
    def _create_provider(split_config: ResolvedSplitConfig, dataset_type: DatasetType) -> BaseDataset:
        """
        Creates an instance of a dataset provider, passing only the necessary
        arguments to its constructor.
        """
        # 1. Determine the dataset type and get the class
        dataset_cls = _infer_dataset_class_from_type(dataset_type)

        # 2. Prepare constructor arguments based on the dataset type
        constructor_args = {
            "preprocess_config": split_config.preprocessing,
        }

        constructor_args.update(split_config.constructor_args)

        constructor_args = filter_kwargs_for_callable(dataset_cls, constructor_args)

        return dataset_cls(**constructor_args)
    
def _infner_dataset_type(identifier: str) -> DatasetType:
    """
    Infers the dataset type from the identifier.

    Args:
        identifier: The dataset identifier.

    Returns:
        DatasetType: The inferred dataset type.

    Raises:
        ValueError: If the dataset identifier is not recognized.
    """
    if huggingface_dataset_utils.verify_hf_dataset_identifier_exists(identifier):
        dataset_type = DatasetType.HUGGINGFACE
    else:
        try:
            dataset_type = DatasetType(identifier.upper())
        except ValueError:
            raise ValueError(f"Unknown dataset identifier: {identifier}")
    
    return dataset_type
    
def _infer_dataset_class(identifier) -> type:
    """
    Infers the dataset class to use for a given split configuration.

    Args:
        identifier: The dataset identifier.

    Returns:
        type: The dataset class to instantiate.

    Raises:
        ValueError: If the dataset identifier is not recognized.
    """

    dataset_type = _infner_dataset_type(identifier)

    return _infer_dataset_class_from_type(dataset_type)
    

def _infer_dataset_class_from_type(dataset_type: DatasetType) -> type:
    """
    Infers the dataset class based on the dataset type.

    Args:
        dataset_type: The dataset type.

    Returns:
        type: The dataset class corresponding to the dataset type.

    Raises:
        ValueError: If the dataset type is not recognized.
    """
    try:
        return DATASET_MAP[dataset_type]
    except KeyError:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
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


@staticmethod
def available_datasets() -> list:
        """
        Returns a list of available datasets.

        Returns
        -------
        list
            A list of available datasets.
        """

        return list(DATASET_MAP.keys())
