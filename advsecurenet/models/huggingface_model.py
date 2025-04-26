import re
from typing import List, Optional, Any
import warnings
import torch

import transformers
from transformers import AutoModel, AutoConfig

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
        self._architecture_overrides = config.architecture if config.architecture is not None else {}
        self._model_class_name_override = config.model_class_name
        super().__init__()
        
    def load_model(self):
        """
        Load a model from the Hugging Face Hub.

        Prioritizes manual 'model_class_name' from config if provided.
        Otherwise, attempts to infer the class from the model's config on the Hub.
        Falls back to AutoModel if inference fails or is not possible.

        Raises:
            ValueError: If the model ID cannot be extracted, the specified manual class
                        is invalid, or loading fails.
        """
        try:
            model_id = HuggingFaceModel.extract_model_id_from_url(self._model_url)
            if not model_id:
                raise ValueError(f"Could not extract model ID from URL: {self._model_url}")

            ModelClass = None
            determined_class_name = "Undetermined"

            # --- Step 1: Determine Model Class ---

            # Priority 1: Manual Override
            if self._model_class_name_override:
                try:
                    ManualModelClass = getattr(transformers, self._model_class_name_override, None)
                    if ManualModelClass is not None and issubclass(ManualModelClass, torch.nn.Module):
                        ModelClass = ManualModelClass
                        determined_class_name = f"{self._model_class_name_override} (Manual)"
                    else:
                        # Raise error if manually specified class is invalid
                        raise ValueError(f"Manually specified model_class_name '{self._model_class_name_override}' not found or invalid in transformers.")
                except Exception as e:
                    raise ValueError(f"Error loading manually specified class '{self._model_class_name_override}': {e}") from e

            # Priority 2: Inference from Hub Config (if manual override not used)
            if ModelClass is None:
                config = AutoConfig.from_pretrained(
                    model_id,
                    revision=self._revision,
                    cache_dir=self._cache_dir,
                    trust_remote_code=self._trust_remote_code,
                )

                # Default to AutoModel if inference fails
                ModelClass = AutoModel
                determined_class_name = "AutoModel (Base - Fallback)"

                if config.architectures and isinstance(config.architectures, (list, tuple)) and len(config.architectures) > 0:
                    arch_name = config.architectures[0]
                    try:
                        InferredModelClass = getattr(transformers, arch_name, None)
                        if InferredModelClass is not None and issubclass(InferredModelClass, torch.nn.Module):
                            ModelClass = InferredModelClass
                            determined_class_name = f"{arch_name} (Inferred)"
                        else:
                            warnings.warn(f"Architecture '{arch_name}' specified in config not found/invalid. Falling back to AutoModel.")
                    except Exception as e:
                        warnings.warn(f"Error trying to load inferred class '{arch_name}': {e}. Falling back to AutoModel.")

            print(f"AdvSecureNet: Determined model class: {determined_class_name}")

            # --- Step 2: Load Model using Determined Class ---
            common_args = {
                "revision": self._revision,
                "cache_dir": self._cache_dir,
                "trust_remote_code": self._trust_remote_code,
            }

            if self._pretrained:
                load_args = common_args.copy()
                if self._architecture_overrides:
                    warnings.warn("Architecture arguments are applied via config for non-pretrained models. Ignoring for pretrained loading.")
                self.model = ModelClass.from_pretrained(model_id, **load_args)
            else:
                # Load config again if not already loaded (only needed if manual override was used)
                if 'config' not in locals():
                     config = AutoConfig.from_pretrained(
                         model_id,
                         revision=self._revision,
                         cache_dir=self._cache_dir,
                         trust_remote_code=self._trust_remote_code,
                     )

                # Apply architecture overrides to the config object
                if self._architecture_overrides:
                    for key, value in self._architecture_overrides.items():
                        if hasattr(config, key):
                            setattr(config, key, value)
                        else:
                            warnings.warn(f"Architecture override arg '{key}' not found in model config, ignoring.")

                self.model = ModelClass.from_config(config)

        except Exception as e:
            # Add more context to the final error message
            raise ValueError(f"Error loading Hugging Face model '{model_id}' using class '{determined_class_name}': {str(e)}") from e
        
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