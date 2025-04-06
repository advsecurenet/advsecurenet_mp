import re
import torch
from typing import List, Optional, Union, Dict, Any, Tuple
from torch.utils.data import Dataset as TorchDataset

from datasets import load_dataset, load_dataset_builder, Dataset, DatasetDict

from advsecurenet.datasets.base_dataset import BaseDataset
from advsecurenet.shared.types.configs.preprocess_config import PreprocessConfig, HuggingFaceDatasetConfig
from advsecurenet.shared.types.dataset import DataType


class HuggingFaceTorchDataset(TorchDataset):
    """
    A PyTorch Dataset wrapper for Hugging Face datasets.
    
    This class wraps a Hugging Face dataset to make it compatible with PyTorch's
    Dataset interface. It handles both image and non-image datasets.
    """
    
    def __init__(self, dataset, transform=None):
        """
        Initialize a HuggingFaceTorchDataset.
        
        Args:
            dataset: A Hugging Face dataset.
            transform: Optional transform to apply to the data.
        """
        self.dataset = dataset
        self.transform = transform
        
        # Check if this is an image dataset
        self.is_image_dataset = 'image' in dataset.features
        
        # Check if this is a classification dataset
        self.is_classification = 'label' in dataset.features
    
    def __len__(self):
        """
        Get the length of the dataset.
        
        Returns:
            int: The number of examples in the dataset.
        """
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset.
        
        Args:
            idx (int): The index of the item to get.
            
        Returns:
            tuple: A tuple containing the data and the label.
        """
        item = self.dataset[idx]
        
        if self.is_image_dataset:
            # Handle image datasets
            image = item['image']
            label = item['label'] if self.is_classification else 0
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        else:
            # Handle non-image datasets (text, tabular, etc.)
            features = {k: v for k, v in item.items() if k != 'label'}
            label = torch.tensor(item['label']) if self.is_classification else torch.tensor(0)
            
            return features, label


class HuggingFaceDataset(BaseDataset):
    """
    A dataset class for Hugging Face datasets.
    
    This class provides functionality to load datasets from the Hugging Face Hub.
    It supports various dataset types and can be used with any dataset available
    on the Hugging Face Hub.
    """
    
    def __init__(
        self,
        preprocess_config: Optional[PreprocessConfig] = None,
        huggingface_config: Optional[HuggingFaceDatasetConfig] = None,
        num_classes: int = None,
        num_input_channels: int = None,
        input_size: Tuple[int, int] = None,
        mean: List[float] = None,
        std: List[float] = None,
    ):
        """
        Initialize a HuggingFaceDataset.
        
        Args:
            preprocess_config (Optional[PreprocessConfig]): Configuration for preprocessing.
            huggingface_config (Optional[HuggingFaceDatasetConfig]): Configuration for the Hugging Face dataset.
            num_classes (int, optional): Number of classes in the dataset.
            num_input_channels (int, optional): Number of input channels.
            input_size (Tuple[int, int], optional): Input size.
            mean (List[float], optional): Mean for normalization.
            std (List[float], optional): Standard deviation for normalization.
            
        Raises:
            ValueError: If huggingface_config is not provided.
        """
        if huggingface_config is None:
            raise ValueError("HuggingFaceDatasetConfig must be provided")
        
        super().__init__(preprocess_config)
        
        self.huggingface_config = huggingface_config
        self._dataset = None
        self._split = None
        
        # Set dataset name
        self.name = f"huggingface-{huggingface_config.dataset_id}"
        
        # Try to get dataset info from Hugging Face
        try:
            builder = load_dataset_builder(
                huggingface_config.dataset_id,
                name=huggingface_config.subset,
                revision=huggingface_config.revision,
                cache_dir=huggingface_config.cache_dir,
                trust_remote_code=huggingface_config.trust_remote_code,
            )
            
            # Extract dataset information
            if 'label' in builder.info.features:
                self.num_classes = builder.info.features['label'].num_classes
            else:
                self.num_classes = num_classes or 2  # Default to binary classification
            
            if 'image' in builder.info.features:
                # Image dataset
                if hasattr(builder.info.features['image'], 'shape'):
                    shape = builder.info.features['image'].shape
                    if len(shape) == 3:
                        self.num_input_channels = shape[0]
                        self.input_size = (shape[1], shape[2])
                    else:
                        self.num_input_channels = 1
                        self.input_size = shape
                else:
                    self.num_input_channels = num_input_channels or 3
                    self.input_size = input_size or (224, 224)
            else:
                # Non-image dataset
                self.num_input_channels = num_input_channels or 1
                self.input_size = input_size or (1, 1)
            
            # Set normalization parameters
            self.mean = mean or [0.5] * self.num_input_channels
            self.std = std or [0.5] * self.num_input_channels
            
        except Exception:
            # Fallback to provided values
            self.num_classes = num_classes or 2
            self.num_input_channels = num_input_channels or 3
            self.input_size = input_size or (224, 224)
            self.mean = mean or [0.5] * self.num_input_channels
            self.std = std or [0.5] * self.num_input_channels
    
    def get_dataset_class(self):
        """
        Get the dataset class.
        
        Returns:
            type: The dataset class.
        """
        return Dataset
    
    def load_dataset(self, train: bool = True, root: str = None, download: bool = True, **kwargs) -> TorchDataset:
        """
        Load a dataset from the Hugging Face Hub.
        
        Args:
            train (bool, optional): Whether to load the training set. Defaults to True.
            root (str, optional): Root directory for the dataset. Not used for Hugging Face datasets.
            download (bool, optional): Whether to download the dataset. Not used for Hugging Face datasets.
            **kwargs: Additional arguments to pass to the dataset.
            
        Returns:
            TorchDataset: The loaded dataset.
            
        Raises:
            ValueError: If the dataset cannot be loaded.
        """
        try:
            # Set data type
            self.data_type = DataType.TRAIN if train else DataType.TEST
            
            # Determine split
            split = self._split or self.huggingface_config.split
            if not split:
                split = "train" if train else "test"
            
            # Load dataset
            hf_dataset = load_dataset(
                self.huggingface_config.dataset_id,
                name=self.huggingface_config.subset,
                split=split,
                revision=self.huggingface_config.revision,
                cache_dir=self.huggingface_config.cache_dir,
                trust_remote_code=self.huggingface_config.trust_remote_code,
            )
            
            # Get transforms
            transform = self.get_transforms()
            
            # Create PyTorch dataset
            self._dataset = HuggingFaceTorchDataset(hf_dataset, transform)
            
            return self._dataset
            
        except Exception as e:
            raise ValueError(f"Error loading Hugging Face dataset: {str(e)}")
    
    def _create_dataset(self, dataset_class, transform, root):
        """
        Create a dataset instance.
        
        This method is not used for Hugging Face datasets as we use the load_dataset method instead.
        
        Raises:
            NotImplementedError: This method is not implemented for Hugging Face datasets.
        """
        raise NotImplementedError("This method is not implemented for Hugging Face datasets")
    
    @staticmethod
    def is_huggingface_url(url: str) -> bool:
        """
        Check if a URL is a Hugging Face dataset URL.
        
        Args:
            url (str): The URL to check.
            
        Returns:
            bool: True if the URL is a Hugging Face dataset URL, False otherwise.
        """
        if not url:
            return False
        
        pattern = r'^(https?://) ?(www\.)?(huggingface\.co|hf\.co)/datasets/([^/]+/[^/]+|[^/]+).*$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def extract_dataset_id_from_url(url: str) -> Optional[str]:
        """
        Extract the dataset ID from a Hugging Face URL.
        
        Args:
            url (str): The URL to extract the dataset ID from.
            
        Returns:
            Optional[str]: The dataset ID if the URL is a valid Hugging Face dataset URL, None otherwise.
        """
        if not HuggingFaceDataset.is_huggingface_url(url):
            return None
        
        pattern = r'^(https?://) ?(www\.)?(huggingface\.co|hf\.co)/datasets/([^/]+/[^/]+|[^/]+).*$'
        match = re.match(pattern, url)
        if match:
            return match.group(4)
        
        return None
