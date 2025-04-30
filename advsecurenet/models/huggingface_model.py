import re
from typing import List, Optional, Tuple
import warnings
import torch

import transformers
from transformers import AutoModel, AutoConfig

from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError 

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
        self._model_id = config.model_id
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
                    self._model_id,
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
                self.model = ModelClass.from_pretrained(self._model_id, **load_args)
            else:
                # Load config again if not already loaded (only needed if manual override was used)
                if 'config' not in locals():
                     config = AutoConfig.from_pretrained(
                         self._model_id,
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
            raise ValueError(f"Error loading Hugging Face model '{self._model_id}' using class '{determined_class_name}': {str(e)}") from e
        
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
    def is_huggingface_id(identifier: str) -> bool:
        """
        Checks if a string matches the typical Hugging Face model ID format (e.g., 'user/repo').
        Uses regex for basic format validation, does not check Hub existence.
        """
        if not identifier:
            return False
        # Regex: Starts with allowed chars, has '/', ends with allowed chars.
        # Allowed chars: letters, numbers, dot, underscore, hyphen.
        pattern = r"^[a-zA-Z0-9._-]+/[a-zA-Z0-9._-]+$"
        return bool(re.match(pattern, identifier))
    
    @staticmethod
    def _check_hub_for_id(model_id: str) -> bool:
        """
        Internal helper: Checks if a specific model ID exists on the Hub.
        Assumes model_id is already validated for the correct format (e.g., "user/repo").

        Returns:
            bool: True if the model exists, False if specifically not found.
        Raises:
            Exception: Propagates exceptions (network errors, etc.) from model_info.
        """
        # REMOVED: Initial format check (if not model_id or '/' not in model_id:)
        try:
            model_info(model_id)
            return True
        except RepositoryNotFoundError:
            # Model ID specifically not found on the Hub
            return False
    
    @staticmethod
    def verify_hf_identifier_exists(identifier: str) -> bool:
        """
        Verifies if a Hugging Face identifier (URL or model ID) corresponds
        to an existing model on the Hub. Validates format before checking.

        Args:
            identifier (str): The model identifier (e.g., "user/repo" or "https://huggingface.co/user/repo").

        Returns:
            bool: True if the identifier points to an existing model on the Hub, False otherwise.
                  Returns False also if network errors occur during the check.
        """
        if not identifier:
            return False

        model_id_to_check: Optional[str] = None

        if HuggingFaceModel.is_huggingface_url(identifier):
            extracted_id = HuggingFaceModel.extract_model_id_from_url(identifier)
            if extracted_id:
                # Extracted ID should already be in the correct format
                model_id_to_check = extracted_id
            else:
                # Invalid URL format or failed extraction
                return False # Cannot proceed
        elif HuggingFaceModel.is_huggingface_id(identifier):
            # Identifier matches the ID format
            model_id_to_check = identifier
        else:
            # Identifier is neither a valid URL nor a valid ID format
            return False

        # If we have a valid ID format, perform the actual check on the Hub
        try:
            return HuggingFaceModel._check_hub_for_id(model_id_to_check)
        except Exception as e:
            # Treat Hub check errors (network, etc.) as "doesn't exist" for inference purposes
            warnings.warn(f"Could not verify Hugging Face identifier '{identifier}' due to Hub check error: {e}")
            return False
    
    @staticmethod
    def resolve_hf_identifiers(config: HuggingFaceModelConfig) -> Tuple[str, str]:
        """
        Determines the canonical Hugging Face model ID and model name from the config.

        Handles cases where model_id is provided vs. not provided, and whether
        the identifiers are URLs or plain IDs.

        Args:
            config (CreateModelConfig): The resolved configuration object potentially
                                        containing model_name and model_id.

        Returns:
            Tuple[str, str]: A tuple containing (canonical_model_id, canonical_model_name).

        Raises:
            ValueError: If a URL is provided but the ID cannot be extracted.
        """
        provided_model_identifier = getattr(config, "model_identifier", None)
        provided_model_name = config.model_name # Assumed to be always present

        final_model_id: str
        final_model_name: str

        if not provided_model_identifier:
            # Case 1: model_id field was NOT provided in config
            identifier = provided_model_name
            if HuggingFaceModel.is_huggingface_url(identifier):
                extracted_id = HuggingFaceModel.extract_model_id_from_url(identifier)
                if not extracted_id:
                    raise ValueError(f"Could not extract model ID from URL in model_name: {identifier}")
                # If only model_name (as URL) was given, use extracted ID for both
                final_model_id = extracted_id
                final_model_name = extracted_id 
            else:
                # Assume model_name is the ID
                final_model_id = identifier
                final_model_name = identifier
        else:
            # Case 2: model_id field WAS provided in config
            final_model_name = provided_model_name # Use the required model_name directly

            identifier_for_id = provided_model_identifier
            if HuggingFaceModel.is_huggingface_url(identifier_for_id):
                extracted_id = HuggingFaceModel.extract_model_id_from_url(identifier_for_id)
                if not extracted_id:
                    raise ValueError(f"Could not extract model ID from URL in model_id field: {identifier_for_id}")
                final_model_id = extracted_id
            else:
                # Assume provided model_id is the ID
                final_model_id = identifier_for_id

        return final_model_id, final_model_name

    
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