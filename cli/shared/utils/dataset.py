from typing import Optional, Tuple, Dict

from torch.utils.data import Dataset as TorchDataset

from advsecurenet.datasets import DatasetFactory
from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.shared.types.dataset import DatasetType


from cli.shared.types.utils.dataset import (
    CreateDatasetCliConfigType,
)

from advsecurenet.utils.huggingface_utils import huggingface_dataset_utils
from advsecurenet.utils.huggingface_utils import huggingface_general_utils

from cli.shared.types.utils.dataset import determine_identifier_and_soruce, ResolvedSplitConfig, DatasetFinalType


def get_datasets(
    config: CreateDatasetCliConfigType, **kwargs
) -> Tuple[Optional[TorchDataset], Optional[TorchDataset]]:
    """
    Load the datasets conditionally based on provided paths.
    """
    dataset_type, config.dataset_name = _validate_dataset_name(config.dataset_name)


    identifier, _ = determine_identifier_and_soruce(config)

    if config.splits is None and config.split_config is None:
        splits = ["train", "test"]
    elif len(config.splits) > 2:
        splits = config.splits[:2]
        Warning(
            f"More than 2 splits provided: {config.splits}. Only the first two splits will be used: {splits}.")
    else:
        splits = config.splits

    if config.split_config is not None and len(config.split_config) > 2:
        Warning(
            f"More than 2 split configurations provided: {config.split_config}. Only the first two will be used: {config.split_config[:2]}.")

    internal_splits = ["train", "test"]

    final_config = DatasetFinalType(
        dataset_name=config.dataset_name,
        splits={},
    )
    if config.split_config is None:
        for split_name, internal_name in zip(splits, internal_splits):
            split = ResolvedSplitConfig(
                identifier=identifier,
                split=split_name,
                preprocessing=config.preprocessing,
                kwargs=config.dataset_arguments or {},
                num_classes=config.num_classes
            )
            final_config.splits[internal_name] = split
    else:
        # If a specific split config is provided, map its entries to our internal roles.
        user_split_configs = list(config.split_config.values())
        for i, internal_name in enumerate(internal_splits):
            if i >= len(user_split_configs):
                break  # Stop if the user provided fewer splits than we have internal roles for.

            user_config = user_split_configs[i]

            # Create a ResolvedSplitConfig by overriding global settings with split-specific ones.
            resolved_split = ResolvedSplitConfig(
                identifier=user_config.identifier or identifier,
                split=user_config.split_name,
                preprocessing=user_config.preprocessing or config.preprocessing,
                kwargs=user_config.dataset_arguments or config.dataset_arguments or {},
                num_classes=config.num_classes 
            )
            final_config.splits[internal_name] = resolved_split


    loaded_datasets: Dict[str, Optional[TorchDataset]] = {}
    for logical_name, resolved_config in final_config.splits.items():
        try:
            # 1. Determine the dataset type for this specific split
            dataset_type, processed_identifier = _validate_dataset_name(resolved_config.identifier)

            # 2. Create the dataset provider object
            dataset_provider = DatasetFactory.create_dataset(
                dataset_type=dataset_type,
                preprocess_config=resolved_config.preprocessing,
            )

            # 3. Prepare arguments for the load_dataset method
            load_kwargs = resolved_config.kwargs.copy()
            load_kwargs['dataset_name'] = processed_identifier
            load_kwargs['split'] = resolved_config.split
            load_kwargs.update(kwargs)

            # 4. Process kwargs and load the dataset
            processed_load_kwargs = dataset_provider.process_kwargs_load_dataset(load_kwargs)
            dataset = dataset_provider.load_dataset(**processed_load_kwargs)
            loaded_datasets[logical_name] = dataset

        except Exception as e:
            print(f"Warning: Could not load dataset for split '{logical_name}'. Error: {e}")
            loaded_datasets[logical_name] = None

    # Extract train and test data to return as a tuple for backward compatibility
    train_data = loaded_datasets.get("train")
    test_data = loaded_datasets.get("test")

    return train_data, test_data

def infer_dataset_type(dataset_identifier: str) -> DatasetType:
    return NotImplementedError()

def get_huggingface_datasets():
    #config, **kwargs 
    #-> Tuple[Optional[TorchDataset], Optional[TorchDataset]]:
    NotImplementedError()
    """
    Load datasets from Hugging Face Hub.

    Args:
        config (HuggingFaceDatasetCliConfigType): Configuration for Hugging Face datasets.
        **kwargs: Arbitrary keyword arguments for the dataset.

    Returns:
        Tuple[Optional[TorchDataset], Optional[TorchDataset]]: Tuple containing the training dataset (if requested)
        and the testing dataset (if requested).
    """
    # Check if dataset_id is a URL and extract the dataset ID if it is
    """dataset_id = config.dataset_id
    if HuggingFaceDataset.is_huggingface_url(dataset_id):
        extracted_id = HuggingFaceDataset.extract_dataset_id_from_url(dataset_id)
        if extracted_id:
            dataset_id = extracted_id

    # Create HuggingFaceDatasetConfig if not provided
    huggingface_config = config.huggingface_config
    if huggingface_config is None:
        huggingface_config = HuggingFaceDatasetConfig(
            dataset_id=dataset_id,
            subset=config.subset,
            split=config.split,
            revision=config.revision,
            cache_dir=config.cache_dir,
            trust_remote_code=config.trust_remote_code
        )
    
    # Create dataset object
    dataset_obj = DatasetFactory.create_dataset(
        dataset_type=DatasetType.HUGGINGFACE,
        preprocess_config=config.preprocessing,
        huggingface_config=huggingface_config,
        **kwargs
    )

    # Determine which splits to load
    train_split = config.split if config.split else "train"
    test_split = config.split if config.split else "test"
    
    # Load datasets
    train_data = None
    test_data = None
    
    try:
        # Override the split in huggingface_config temporarily for train data
        if not config.split:
            dataset_obj._split = train_split
        train_data = dataset_obj.load_dataset(train=True)
    except Exception as e:
        print(f"Warning: Could not load training dataset: {str(e)}")
    
    try:
        # Override the split in huggingface_config temporarily for test data
        if not config.split:
            dataset_obj._split = test_split
        test_data = dataset_obj.load_dataset(train=False)
    except Exception as e:
        print(f"Warning: Could not load test dataset: {str(e)}")
    
    return train_data, test_data"""


def _validate_dataset_name(dataset_name: str) -> str:
    """
    Validate the dataset name.

    Returns:
        str: The validated dataset name.

    Raises:
        ValueError: If the dataset name is not supported.
    """
    if huggingface_dataset_utils.verify_hf_dataset_identifier_exists(dataset_name):
        dataset_type = "HUGGINGFACE"
        dataset_name = huggingface_general_utils.process_hf_identifier(dataset_name)
    else:
        dataset_type = dataset_name.upper()

    try:
        DatasetType(dataset_type)
    except ValueError as e:
        raise ValueError(
            f"Unsupported dataset type! Entered dataset name: {dataset_name}. ")

    return dataset_type, dataset_name