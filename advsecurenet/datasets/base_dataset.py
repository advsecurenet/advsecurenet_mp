from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union
import inspect

import pkg_resources
import torch
from torch.utils.data import Dataset as TorchDataset
from torchvision import datasets
from torchvision.transforms import v2 as transforms

from advsecurenet.shared.types.configs.preprocess_config import (
    PreprocessConfig,
    PreprocessStep,
)
from advsecurenet.shared.types.dataset import DataType
from advsecurenet.utils.kwargs_utils import filter_kwargs_for_callable, pop_keys_from_dict, map_kwargs


class ImageFolderBaseDataset:
    """
    A mixin class for datasets that use the ImageFolder format.
    """

    def get_dataset_class(self):
        """
        Returns the dataset class.
        """
        return datasets.ImageFolder

    def _create_dataset(
        self,
        dataset_class: datasets,
        transform: transforms.Compose,
        root: Optional[str] = None,
        # (kwargs is needed for the consistency of the method signature)
        **kwargs,
    ):
        """
        Creates the dataset.
        """
        return dataset_class(
            root=root,
            transform=transform,
        )


class DatasetWrapper(TorchDataset):
    """
    A wrapper class for PyTorch datasets that allows for easy access to the underlying dataset and having customized parameters.
    """

    def __init__(self, dataset, name):
        self.dataset = dataset
        self.name = name

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class BaseDataset(TorchDataset, ABC):
    """
    A base class for PyTorch datasets.

    Args:
        preprocess_config (Optional[PreprocessConfig], optional): The preprocessing configuration for the dataset. Defaults to None.

    Attributes:
        _dataset (TorchDataset): The underlying PyTorch dataset.
        mean (List[float]): The mean values of the dataset.
        std (List[float]): The standard deviation values of the dataset.
        input_size (Tuple[int, int]): The input size of the dataset.
        name (str): The name of the dataset.
        num_classes (int): The number of classes in the dataset.
        num_input_channels (int): The number of input channels in the dataset.
        data_type (DataType): The type of the dataset (train or test).

    Note:
        This module uses v2 of the torchvision transforms. Please refer to the official PyTorch documentation for more information about the possible transforms.
    """

    def __init__(self, preprocess_config: Optional[PreprocessConfig] = None):
        self._dataset: DatasetWrapper
        self.mean: List[float] = []
        self.std: List[float] = []
        self.input_size: Tuple[int, int]
        self.crop_size: Tuple[int, int]
        self.name: str = ""
        self.num_classes: int
        self.num_input_channels: int
        self.data_type: DataType
        self._preprocess_config = preprocess_config

    @abstractmethod
    def get_dataset_class(self):
        """
        Returns the dataset class.
        """

    def load_dataset(self, **kwargs) -> DatasetWrapper:
        """
        Loads the dataset.

        Args:
            **kwargs: Arbitrary keyword arguments for the dataset.
                      Common ones: root, train, download, split, etc.

        Returns:
            DatasetWrapper: The dataset loaded into memory.
        """
        if "root" not in kwargs or kwargs["root"] is None:
            kwargs["root"] = pkg_resources.resource_filename("advsecurenet", "data")

        kwargs["transform"] = self.get_transforms()

        dataset_class = self.get_dataset_class()

        dataset = self._create_dataset(
            dataset_class=dataset_class,
            **kwargs)

        if "split" in kwargs:
            try:
                self.data_type = DataType(kwargs["split"].upper())
            except Exception:
                self.data_type = None

        self._dataset = DatasetWrapper(dataset=dataset, name=self.name)
        return self._dataset

    def get_transforms(self):
        """Returns the data transforms to be applied to the dataset."""
        if self._preprocess_config and self._preprocess_config. steps:
            preprocess_steps = self._preprocess_config.steps
            transform_steps = self._construct_transforms(preprocess_steps)
            return transforms.Compose(transform_steps)

        return transforms.Compose([transforms.ToTensor()])

    def _get_transform(self, name):
        """Retrieve a transform by name, raising an error if not found."""
        available_transforms = self._available_transforms()
        if name not in available_transforms:
            raise ValueError(
                f"Transform {name} is not available. Available transforms are: {list(available_transforms)}"
            )
        return getattr(transforms, name)

    def _convert_param(self, value):
        """Convert parameter strings to appropriate types, especially for torch types."""
        if isinstance(value, str) and "." in value and "torch" in value:
            try:
                return torch.__dict__[value.split(".")[-1]]
            except KeyError as exc:
                raise ValueError(f"Invalid torch parameter: {value}") from exc
        return value

    def _construct_transforms(
        self, preprocess_steps: List[Union[PreprocessStep, dict]]
    ) -> List[Union[transforms.Compose, transforms.Transform]]:
        """Construct a list of transforms based on configuration steps."""

        preprocess_steps: List[PreprocessStep] = self._to_preprocess_step(
            preprocess_steps
        )

        transform_steps = []
        for step in preprocess_steps:
            transform = self._get_transform(step.name)
            params = (
                {
                    k: self._convert_param(v)
                    for k, v in step.params.items()
                    if v is not None
                }
                if step.params
                else {}
            )
            transform_steps.append(transform(**params))
        return transform_steps

    def _to_preprocess_step(
        self, preprocess_steps: List[Union[PreprocessStep, dict]]
    ) -> List[PreprocessStep]:
        """
        Converts a list of steps to a PreprocessStep object.

        Args:
            preprocess_steps (List[Union[PreprocessStep, dict]]): The list of preprocess steps.

        Returns:
            List[PreprocessStep]: The list of preprocess steps as PreprocessStep objects.
        """
        if any(isinstance(step, dict) for step in preprocess_steps):
            preprocess_steps = [
                PreprocessStep(**step) if isinstance(step, dict) else step
                for step in preprocess_steps
            ]

        return preprocess_steps

    def _create_dataset(
        self,
        dataset_class: datasets,
        **kwargs,
    ):    
        filtered_kwargs = filter_kwargs_for_callable(
            dataset_class, kwargs)

        return dataset_class(**filtered_kwargs)

    def _available_transforms(self) -> List[str]:
        """
        Returns the available transforms.
        """
        available_transforms = transforms.__dict__.keys()
        available_transforms = [
            t for t in available_transforms if not t.startswith("_")
        ]
        return available_transforms

    def __len__(self) -> int:
        """
        Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        if self._dataset:
            return len(self._dataset)
        return 0

    def __getitem__(self, idx: int) -> Any:
        """
        Returns the item at the given index.

        Args:
            idx (int): The index of the item.

        Returns:
            Any: The item at the given index.

        Raises:
            NotImplementedError: If the dataset is not loaded or specified.
        """
        if self._dataset:
            return self._dataset[idx]
        raise NotImplementedError("Dataset not loaded or specified.")
    
    def _map_split_to_train(self, kwargs: dict) -> dict:
        """
        A transformation function that maps a 'split' key to a 'train' boolean key.
        This is designed to be used with the process_kwargs utility.
        """
        if "split" in kwargs:
            split_value = kwargs.pop("split").lower()
            if split_value in ["train", "test"]:
                kwargs["train"] = (split_value == "train")
        return kwargs

    def process_dataset_kwargs(self, kwargs: dict) -> dict:
        """
        Processes and maps generic dataset kwargs to dataset-specific arguments.
        """
        mapping = {
            'split': self._map_split_to_train  # Custom transformation
        }
        return map_kwargs(kwargs, mapping)

    def process_kwargs_load_dataset(self, kwargs: dict) -> dict:
        """
        Processes kwargs for loading the dataset, mapping generic keys to dataset-specific ones.
        """
        # Map generic keys to dataset-specific ones
        kwargs = self.process_dataset_kwargs(kwargs)

        kwargs = pop_keys_from_dict(
            kwargs, ["dataset_name", "num_classes"])
        
        return kwargs