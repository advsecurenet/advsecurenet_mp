import re
from typing import List, Optional, Any
import warnings

from transformers import AutoModel, AutoConfig

from advsecurenet.models.base_model import BaseModel
from advsecurenet.shared.types.configs.model_config import HuggingFaceModelConfig


class HuggingFaceModel(BaseModel):
    """
    A model class for Hugging Face models.
    
    This class provides functionality to load models from the Hugging Face Hub.
    It supports both pretrained and non-pretrained models, and can be used with
    any model available on the Hugging Face Hub.
    """

    def __init__(self, config: HuggingFaceModelConfig):
        """
        Initialize a HuggingFaceModel.
        
        Args:
            config (HuggingFaceModelConfig): Configuration for the Hugging Face model.
        """
        super().__init__()
        self.config = config
        self.load_model()
        
    def load_model(self):
        """
        Load a model from the Hugging Face Hub.
        
        This method loads a model from the Hugging Face Hub based on the configuration.
        If pretrained is True, it loads the pretrained weights, otherwise it initializes
        the model with random weights.
        
        Raises:
            ValueError: If the model cannot be loaded or if there's a mismatch in the number of classes.
        """
        try:
            model_id = self.config.model_id
            
            # Load model configuration
            model_config = AutoConfig.from_pretrained(
                model_id,
                revision=self.config.revision,
                cache_dir=self.config.cache_dir,
                trust_remote_code=self.config.trust_remote_code,
            )
            
            # Load model
            if self.config.pretrained:
                self.model = AutoModel.from_pretrained(
                    model_id,
                    revision=self.config.revision,
                    cache_dir=self.config.cache_dir,
                    trust_remote_code=self.config.trust_remote_code,
                )
            else:
                self.model = AutoModel.from_config(model_config)
            
            # Check for classifier mismatch and raise error or warning
            if hasattr(self.model, 'classifier') and hasattr(self.model.classifier, 'out_features'):
                if self.model.classifier.out_features != self.config.num_classes:
                    error_msg = (f"Class mismatch: Model has {self.model.classifier.out_features} output classes "
                                f"but config specifies {self.config.num_classes} classes. "
                                f"Please ensure the number of classes matches the model architecture.")
                    raise ValueError(error_msg)
            
        except Exception as e:
            raise ValueError(f"Error loading Hugging Face model: {str(e)}")
    
    @classmethod
    def models(cls) -> List[str]:
        """
        Get a list of available models.
        
        Returns:
            List[str]: A list of available model names.
        """
        # Hugging Face Hub has too many models to list, so we return an empty list
        return []
    
    @staticmethod
    def is_huggingface_url(url: str) -> bool:
        """
        Check if a URL is a Hugging Face URL.
        
        Args:
            url (str): The URL to check.
            
        Returns:
            bool: True if the URL is a Hugging Face URL, False otherwise.
        """
        if not url:
            return False
        
        pattern = r'^(https?://) ?(www\.)?(huggingface\.co|hf\.co)/([^/]+/[^/]+).*$'
        return bool(re.match(pattern, url))
    
    @staticmethod
    def extract_model_id_from_url(url: str) -> Optional[str]:
        """
        Extract the model ID from a Hugging Face URL.
        
        Args:
            url (str): The URL to extract the model ID from.
            
        Returns:
            Optional[str]: The model ID if the URL is a valid Hugging Face URL, None otherwise.
        """
        if not HuggingFaceModel.is_huggingface_url(url):
            return None
        
        pattern = r'^(https?://) ?(www\.)?(huggingface\.co|hf\.co)/([^/]+/[^/]+).*$'
        match = re.match(pattern, url)
        if match:
            return match.group(4)
        
        return None