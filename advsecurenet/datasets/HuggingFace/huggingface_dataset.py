import re
import torch
from typing import Optional

from datasets import load_dataset, load_dataset_builder, Dataset, DatasetDict

from advsecurenet.datasets.base_dataset import BaseDataset
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
        self.input_size = (32, 32)
        self.crop_size = (32, 32)
        self.name = "cifar10"
        self.num_classes = 10
        self.num_input_channels = 3
    
    def get_dataset_class(self):
        """
        Get the dataset class.
        
        Returns:
            type: The dataset class.
        """
        return Dataset
    
    def load_dataset(self, train: bool = True, root: str = None, download: bool = True, **kwargs):
        """
        Load a dataset from the Hugging Face Hub.
        
        Args:
            train (bool, optional): Whether to load the training set. Defaults to True.
            root (str, optional): Root directory for the dataset. Not used for Hugging Face datasets.
            download (bool, optional): Whether to download the dataset. Not used for Hugging Face datasets.
            **kwargs: Additional arguments to pass to the dataset.
            
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