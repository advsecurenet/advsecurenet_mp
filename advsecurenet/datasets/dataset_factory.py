from typing import Optional, Dict

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
                load_kwargs = split_config.kwargs.copy()
                load_kwargs['dataset_name'] = identifier
                load_kwargs['split'] = split_config.source_split_name
                load_kwargs['root'] = getattr(split_config, 'path', None)
                load_kwargs['download'] = getattr(resolved_config, 'download', True)
                
                # Merge runtime kwargs last
                load_kwargs.update(runtime_kwargs)

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
