import re
from typing import List, Optional, Any
import warnings
import torch

from transformers import AutoModel, AutoConfig, AutoModelForImageClassification

from advsecurenet.models.base_model import BaseModel, check_model_loaded
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
        self._model_url = config.model_url
        self._pretrained = config.pretrained
        self._revision = config.revision
        self._cache_dir = config.cache_dir
        self._trust_remote_code = config.trust_remote_code
        super().__init__()
        
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
            # Load model
            if self._pretrained:
                self.model = AutoModelForImageClassification.from_pretrained(
                    HuggingFaceModel.extract_model_id_from_url(self._model_url),
                    revision=self._revision,
                    cache_dir=self._cache_dir,
                    trust_remote_code=self._trust_remote_code,
                )
            else:
                # Load model configuration
                model_config = AutoConfig.from_pretrained(
                    HuggingFaceModel.extract_model_id_from_url(self._model_url),
                    revision=self._revision,
                    cache_dir=self._cache_dir,
                    trust_remote_code=self._trust_remote_code,
                )

                self.model = AutoModelForImageClassification.from_config(model_config)
            
        except Exception as e:
            raise ValueError(f"Error loading Hugging Face model: {str(e)}")
    
    @classmethod
    def models(cls) -> List[str]:
        """
        Get a list of available models.
        
        Returns:
            List[str]: A list of available model names.
        """
        raise NotImplementedError("This method is not applicable for huggingface models.")
    
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
    
    @check_model_loaded # Use the decorator from BaseModel
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        Forward pass specific to Hugging Face models.

        This method overrides the base forward pass to handle the specific output
        format of Hugging Face models, which typically return an object containing
        various outputs (like logits, hidden states, etc.). This method extracts
        and returns only the logits tensor.

        Args:
            x (torch.Tensor): The primary input tensor to the model (e.g., pixel values).
            *args: Additional positional arguments to pass to the underlying Hugging Face model's forward method.
            **kwargs: Additional keyword arguments to pass to the underlying Hugging Face model's forward method
                      (e.g., attention_mask).

        Returns:
            torch.Tensor: The output logits tensor from the Hugging Face model.

        Raises:
            TypeError: If the underlying Hugging Face model's output is neither a Tensor nor an object
                       with a 'logits' attribute.
            ValueError: If the model is not loaded (handled by the decorator).
        """
        # Call the underlying Hugging Face model
        output = self.model(x, *args, **kwargs)

        # Extract logits
        if hasattr(output, 'logits') and isinstance(output.logits, torch.Tensor):
            return output.logits
        elif isinstance(output, torch.Tensor):
            # Fallback if the HF model unexpectedly returned a raw tensor
            warnings.warn("HuggingFace model returned a raw Tensor instead of an output object. Returning the tensor directly.")
            return output
        else:
            # If the output is something else unexpected
            raise TypeError(
                f"HuggingFaceModel expected output with 'logits' attribute or a Tensor, but got {type(output)}."
            )