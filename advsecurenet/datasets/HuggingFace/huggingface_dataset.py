from typing import Optional, Any, Dict # Added Any, Dict
import torch # Added torch
from datasets import load_dataset as hf_hub_load_dataset

from advsecurenet.datasets.base_dataset import BaseDataset, DatasetWrapper
from advsecurenet.shared.types.configs.preprocess_config import PreprocessConfig
from advsecurenet.shared.types.dataset import DataType


class HuggingFaceDataset(BaseDataset):
    """
    A dataset class for Hugging Face datasets.
    
    This class provides functionality to load datasets from the Hugging Face Hub.
    It supports various dataset types and can be used with any dataset available
    on the Hugging Face Hub.
    """
    
    def __init__(
        self,
        preprocess_config: Optional[PreprocessConfig] = None):
        """
        Initialize a HuggingFaceDataset.
        
        Args:
            preprocess_config (Optional[PreprocessConfig]): Configuration for preprocessing.
            num_classes (int, optional): Number of classes in the dataset.
            num_input_channels (int, optional): Number of input channels.
            input_size (Tuple[int, int], optional): Input size.
            mean (List[float], optional): Mean for normalization.
            std (List[float], optional): Standard deviation for normalization.
            
        Raises:
            ValueError: If huggingface_config is not provided.
        """
        
        super().__init__(preprocess_config)
        
        self.mean = None
        self.std = None
        #self.name = "uoft-cs/cifar10"
        self.input_size = (32, 32)
        self.crop_size = (32, 32)
        self.num_classes = 10
        self.num_input_channels = 3

        self._image_key = 'img'
        self._label_key = 'label'

    def get_dataset_class(self):
        """
        Not used by this HuggingFaceDataset's custom load_dataset flow.
        """
        return None
    
    def load_dataset(self, **kwargs) -> DatasetWrapper:
        """
        Load 'uoft-cs/cifar10' from Hugging Face Hub.
        'root' and 'download' args are ignored.
        """
        split_name = kwargs.get("split", "train")
        dataset_name = kwargs.get("dataset_name")

        try:
            self.data_type = DataType(split_name.upper())
        except Exception:
            self.data_type = None

        try:
            # Load raw Hugging Face dataset split
            self._raw_hf_data = hf_hub_load_dataset(
                path=dataset_name,
                split=split_name,
            )
            
            # Get and store torchvision transforms
            self._transforms_to_apply = self.get_transforms()
            
            self._dataset = DatasetWrapper(dataset=self, name=self.name)
            return self._dataset
        

        except Exception as e:
            # Print the actual Hugging Face error if possible
            hf_error_msg = ""
            if hasattr(e, 'args') and e.args:
                hf_error_msg = str(e.args[0])
            raise ValueError(f"Error loading Hugging Face dataset '{self.name}' (split: {split_name}): {str(e)}. HF Message: {hf_error_msg}") from e
        
    
    def _create_dataset(self, dataset_class, transform, root, train, download, **kwargs):
        """
        Not used for Hugging Face datasets.
        """
        raise NotImplementedError("_create_dataset is not applicable to HuggingFaceDataset.")
    

    def __len__(self) -> int:
        """
        Return the number of samples in the loaded Hugging Face dataset split.
        """
        if self._raw_hf_data is None:
            raise RuntimeError(f"Hugging Face dataset '{self.name}' not loaded. Call load_dataset() first.")
        return len(self._raw_hf_data)
    

    def __getitem__(self, idx: int) -> Any:
        """
        Get a sample from the dataset at the given index and apply transforms.
        """
        if self._raw_hf_data is None:
            raise RuntimeError(f"Hugging Face dataset '{self.name}' not loaded. Call load_dataset() first.")
        
        item: Dict[str, Any] = self._raw_hf_data[idx]
        
        image_data = item.get(self._image_key)
        if image_data is None:
            raise KeyError(f"Image key '{self._image_key}' not found in dataset item at index {idx}. Available keys: {list(item.keys())}")

        if self._transforms_to_apply:
            image_data = self._transforms_to_apply(image_data)

        label_data = item.get(self._label_key)
        if label_data is None:
            raise KeyError(f"Label key '{self._label_key}' not found in dataset item at index {idx}. Available keys: {list(item.keys())}")
        
        # Convert label to tensor
        label_tensor = torch.tensor(label_data).long()
            
        return image_data, label_tensor