from typing import Optional, Any, Dict, List, Tuple
import torch

from datasets import load_dataset as hf_hub_load_dataset

from advsecurenet.utils.kwargs_utils import (
    map_kwargs,
    filter_kwargs_for_callable,
    pop_keys_from_dict,
)
from advsecurenet.datasets.base_dataset import BaseDataset, DatasetWrapper
from advsecurenet.shared.types.configs.preprocess_config import PreprocessConfig


class HuggingFaceDataset(BaseDataset):
    """
    A dataset class for loading and processing datasets from Hugging Face Hub.
    
    This class extends BaseDataset to provide seamless integration with Hugging Face's
    datasets library, allowing users to load any dataset from the Hugging Face Hub
    and apply preprocessing transformations compatible with the advsecurenet framework.
    
    Args:
        preprocess_config (Optional[PreprocessConfig]): Configuration for preprocessing steps.
        num_classes (Optional[int]): Number of classes in the dataset. If None, will be
            inferred from the dataset.
        num_input_channels (int): Number of input channels (e.g., 3 for RGB images).
            Defaults to 3.
        mean (Optional[List[float]]): Mean values for normalization. If None, will use
            default values or infer from preprocessing config.
        std (Optional[List[float]]): Standard deviation values for normalization. If None,
            will use default values or infer from preprocessing config.
        input_key (str): Key name for input data in the Hugging Face dataset. 
            Defaults to "image".
        target_key (str): Key name for target/label data in the Hugging Face dataset.
            Defaults to "label".
        **kwargs: Additional keyword arguments passed to the Hugging Face load_dataset function.
    
    Example:
        >>> from advsecurenet.datasets.HuggingFace import HuggingFaceDataset
        >>> dataset = HuggingFaceDataset(
        ...     num_classes=10,
        ...     input_key="img",
        ...     target_key="label"
        ... )
        >>> data_wrapper = dataset.load_dataset(
        ...     path="cifar10",
        ...     split="train"
        ... )
    
    Note:
        This class requires the 'datasets' library from Hugging Face to be installed.
        The dataset must be available on the Hugging Face Hub or locally accessible.
    """
    def __init__(
        self,
        preprocess_config: Optional[PreprocessConfig] = None,
        num_classes: Optional[int] = None,
        num_input_channels: int = 3,
        mean: Optional[List[float]] = None,
        std: Optional[List[float]] = None,
        input_key: str = "image",
        target_key: str = "label",
        **kwargs,  # To catch other unused args
    ):
        super().__init__(preprocess_config)

        self.num_classes = num_classes
        self.num_input_channels = num_input_channels
        self.mean = mean
        self.std = std
        self._input_key = input_key
        self._target_key = target_key

    def get_dataset_class(self):
        """
        Not used by this HuggingFaceDataset's custom load_dataset flow.
        """
        return None

    def load_dataset(self, **kwargs) -> DatasetWrapper:
        """
        Load dataset from Hugging Face Hub.
        """
        filtered_kwargs = filter_kwargs_for_callable(hf_hub_load_dataset, kwargs)

        try:
            # Load raw Hugging Face dataset split
            self._raw_hf_data = hf_hub_load_dataset(**filtered_kwargs)

        except Exception as e:
            # Print the actual Hugging Face error if possible
            hf_error_msg = ""
            if hasattr(e, "args") and e.args:
                hf_error_msg = str(e.args[0])
            raise ValueError(
                f"Error loading Hugging Face dataset '{self.name}': {str(e)}. HF Message: {hf_error_msg}"
            ) from e

        # Get and store torchvision transforms
        self._transforms_to_apply = self.get_transforms()

        self._dataset = DatasetWrapper(dataset=self, name=self.name)
        return self._dataset

    def _create_dataset(
        self, dataset_class, transform, root, train, download, **kwargs
    ):
        """
        Not used for Hugging Face datasets.
        """
        raise NotImplementedError(
            "_create_dataset is not applicable to HuggingFaceDataset."
        )

    def __len__(self) -> int:
        """
        Return the number of samples in the loaded Hugging Face dataset split.
        """
        if self._raw_hf_data is None:
            raise RuntimeError(
                f"Hugging Face dataset '{self.name}' not loaded. Call load_dataset() first."
            )
        return len(self._raw_hf_data)

    def __getitem__(self, idx: int) -> Any:
        """
        Get a sample from the dataset at the given index and apply transforms.
        """
        if self._raw_hf_data is None:
            raise RuntimeError(
                f"Hugging Face dataset '{self.name}' not loaded. Call load_dataset() first."
            )

        item: Dict[str, Any] = self._raw_hf_data[idx]

        input_data = item.get(self._input_key)
        if input_data is None:
            raise KeyError(
                f"Image key '{self._input_key}' not found in dataset item at index {idx}. Available keys: {list(item.keys())}"
            )

        if self._transforms_to_apply:
            input_data = self._transforms_to_apply(input_data)

        target_data = item.get(self._target_key)
        if target_data is None:
            raise KeyError(
                f"Label key '{self._target_key}' not found in dataset item at index {idx}. Available keys: {list(item.keys())}"
            )

        # Convert label to tensor
        label_tensor = torch.tensor(target_data).long()

        return input_data, label_tensor

    def process_dataset_kwargs(self, kwargs: dict) -> dict:
        """
        Processes kwargs for Hugging Face datasets, mapping 'dataset_name' to 'path'.
        """
        mapping = {"dataset_name": "path"}
        processed_kwargs = map_kwargs(kwargs, mapping)

        popped_keys = ["download", "root"]
        processed_kwargs = pop_keys_from_dict(processed_kwargs, popped_keys)

        return processed_kwargs

    def process_kwargs_load_dataset(self, kwargs: dict) -> dict:
        """
        Processes kwargs for loading the Hugging Face dataset.
        """
        # Map generic keys to dataset-specific ones
        return self.process_dataset_kwargs(kwargs)
