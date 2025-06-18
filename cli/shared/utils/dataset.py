from typing import Optional, Tuple, cast

from torch.utils.data import Dataset as TorchDataset

from advsecurenet.datasets import DatasetFactory
from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.shared.types.dataset import DatasetType


from cli.shared.types.utils.dataset import (
    DatasetCliConfigType,
)

from advsecurenet.utils.huggingface_utils import huggingface_dataset_utils
from advsecurenet.utils.huggingface_utils import huggingface_general_utils


def get_datasets(
    config: DatasetCliConfigType, **kwargs
) -> Tuple[Optional[TorchDataset], Optional[TorchDataset]]:
    """
    Load the datasets conditionally based on provided paths.
    """
    dataset_type, config.dataset_name = _validate_dataset_name(config.dataset_name)

    # Only pass preprocessing to the factory
    train_dataset_obj = DatasetFactory.create_dataset(
        dataset_type=dataset_type,
        preprocess_config=config.preprocessing,
    )
    test_dataset_obj = DatasetFactory.create_dataset(
        dataset_type=dataset_type,
        preprocess_config=config.preprocessing,
    )

    def load_dataset_part(dataset_obj: BaseDataset, **kwargs) -> Optional[TorchDataset]:
        try:
            kwargs = dataset_obj.process_kwargs_load_dataset(kwargs)
            return dataset_obj.load_dataset(**kwargs)
        except FileNotFoundError:
            return None

    # Prepare kwargs for splits: exclude dataset_name, preprocessing, train_dataset_path, test_dataset_path
    config_dict = vars(config)
    base_exclude = {"preprocessing", "train_dataset_path", "test_dataset_path"}
    split_base_kwargs = {k: v for k, v in config_dict.items() if k not in base_exclude}

    splits = []
    if hasattr(config, "dataset_part"):
        if getattr(config, "dataset_part", None) in ["train", "all"]:
            splits.append(("train", train_dataset_obj, getattr(config, "train_dataset_path", None)))
        if getattr(config, "dataset_part", None) in ["test", "all"]:
            splits.append(("test", test_dataset_obj, getattr(config, "test_dataset_path", None)))
    else:
        splits.append(("train", train_dataset_obj, getattr(config, "train_dataset_path", None)))
        splits.append(("test", test_dataset_obj, getattr(config, "test_dataset_path", None)))


    train_data, test_data = None, None
    for split, dataset_obj, path in splits:
        split_kwargs = dict(split_base_kwargs)
        split_kwargs["root"] = path
        split_kwargs["split"] = split
        split_kwargs.update(kwargs)
        data = load_dataset_part(dataset_obj, **split_kwargs)
        if split == "train":
            train_data = data
        elif split == "test":
            test_data = data

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