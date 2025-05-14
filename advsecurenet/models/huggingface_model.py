import re
from typing import List, Optional, Tuple
import warnings
import torch

import transformers
from transformers import AutoModel, AutoConfig

from huggingface_hub import model_info
from huggingface_hub.utils import RepositoryNotFoundError 

from advsecurenet.models.base_model import BaseModel, check_model_loaded
from advsecurenet.shared.types.configs.model_config import HuggingFaceResolvedConfig, CreateModelConfig, determine_identifier_and_soruce


class HuggingFaceModel(BaseModel):
    """
    A model class for Hugging Face models.
    
    This class provides functionality to load models from the Hugging Face Hub.
    It supports both pretrained and non-pretrained models, and can be used with
    any model available on the Hugging Face Hub.
    """

    def __init__(self, config: HuggingFaceResolvedConfig):
        """
        Initialize a HuggingFaceModel.
        
        Args:
            config (HuggingFaceResolvedConfig): Configuration for the Hugging Face model.
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
        final_determined_class_name = "Undetermined" # Default for the final error message wrapper
        try:
            ModelClass, final_determined_class_name, config_object = self._determine_model_class_and_config()
            self.model = self._instantiate_model(ModelClass, config_object)

        except Exception as e:
            # Re-raise specific ValueErrors from manual override if they match the pattern
            if isinstance(e, ValueError) and (
                (self._model_class_name_override and f"Manually specified model_class_name '{self._model_class_name_override}'" in str(e)) or
                (self._model_class_name_override and f"Error loading manually specified class '{self._model_class_name_override}'" in str(e))
            ):
                raise e # Re-raise the more specific error from manual override handling
            
            # For all other errors, wrap with the generic message using the determined class name
            raise ValueError(f"Error loading Hugging Face model '{self._model_id}' using class '{final_determined_class_name}': {str(e)}") from e
        
    
    @staticmethod
    def _resolve_manual_class_override(model_class_name_override: Optional[str]) -> Tuple[Optional[type], str]:
        """
        Attempts to resolve the model class using manual override.
        Returns (ModelClass, determined_class_name) or (None, "Undetermined") if no override.
        Raises ValueError if manual override is specified but invalid/not found.
        """
        if not model_class_name_override:
            return None, "Undetermined"

        try:
            ManualModelClass = getattr(transformers, model_class_name_override, None)
            if ManualModelClass is not None and issubclass(ManualModelClass, torch.nn.Module):
                return ManualModelClass, f"{model_class_name_override} (Manual)"
            else:
                raise ValueError(f"Manually specified model_class_name '{model_class_name_override}' not found or invalid in transformers.")
        except Exception as e:
            # Catch broader exceptions during getattr/issubclass for manual override
            if isinstance(e, ValueError) and f"Manually specified model_class_name '{model_class_name_override}'" in str(e):
                raise # Re-raise the specific ValueError
            raise ValueError(f"Error loading manually specified class '{model_class_name_override}': {e}") from e
        
    def _resolve_inferred_class_from_hub(self) -> Tuple[type, str, AutoConfig]:
        """
        Infers model class from Hub configuration. Loads AutoConfig.
        Returns (ModelClass, determined_class_name, config_object).
        """
        config = AutoConfig.from_pretrained(
            self._model_id,
            revision=self._revision,
            cache_dir=self._cache_dir,
            trust_remote_code=self._trust_remote_code,
        )

        ModelClass = AutoModel  # Default
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
        
        return ModelClass, determined_class_name, config
    
    def _determine_model_class_and_config(self) -> Tuple[type, str, Optional[AutoConfig]]:
        """
        Determines the ModelClass, its descriptive name, and an optional AutoConfig object.
        Handles manual override first, then inference.
        """
        ModelClass, determined_class_name = self._resolve_manual_class_override(self._model_class_name_override)

        if ModelClass:  # Manual override successful
            return ModelClass, determined_class_name, None # No config loaded yet

        # No successful manual override, proceed to inference
        return self._resolve_inferred_class_from_hub()
    
    def _instantiate_model(self, ModelClass: type, config_from_resolution: Optional[AutoConfig]) -> torch.nn.Module:
        """
        Instantiates the model using the determined ModelClass and config.
        """
        common_args = {
            "revision": self._revision,
            "cache_dir": self._cache_dir,
            "trust_remote_code": self._trust_remote_code,
        }

        if self._pretrained:
            load_args = common_args.copy()
            if self._architecture_overrides:
                warnings.warn("Architecture arguments are applied via config for non-pretrained models. Ignoring for pretrained loading.")
            return ModelClass.from_pretrained(self._model_id, **load_args)
        else:
            config_to_use = config_from_resolution
            if config_to_use is None: # Manual override was used, config not loaded in resolution step
                 config_to_use = AutoConfig.from_pretrained(
                     self._model_id,
                     **common_args # revision, cache_dir, trust_remote_code
                 )

            if self._architecture_overrides:
                for key, value in self._architecture_overrides.items():
                    if hasattr(config_to_use, key):
                        setattr(config_to_use, key, value)
                    else:
                        warnings.warn(f"Architecture override arg '{key}' not found in model config, ignoring.")
            return ModelClass.from_config(config_to_use)
        
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

        model_id_to_check = HuggingFaceModel.process_hf_identifier(identifier)

        if not HuggingFaceModel.is_huggingface_id(model_id_to_check):
            return False

        try:
            return HuggingFaceModel._check_hub_for_id(model_id_to_check)
        except Exception as e:
            # Treat Hub check errors (network, etc.) as "doesn't exist" for inference purposes
            warnings.warn(f"Could not verify Hugging Face identifier '{identifier}' due to Hub check error: {e}")
            return False
        
    @staticmethod
    def process_hf_identifier(identifier):
        # Process the chosen identifier
        if HuggingFaceModel.is_huggingface_url(identifier):
            extracted_id = HuggingFaceModel.extract_model_id_from_url(identifier)
            # If extraction fails, extracted_id will be None or empty, so return it directl
            return extracted_id
        else:
            # Assume it's a direct ID
            return identifier
        
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
    
    @staticmethod
    def resolve_hf_identifiers(config: CreateModelConfig) -> Tuple[str, str]:
        """
        Determines the canonical Hugging Face model ID and model name from the config.

        Prioritizes `config.model_identifier`. If not present, uses `config.model_name`.
        Handles cases where identifiers are URLs or plain IDs.

        Args:
            config (CreateModelConfig): The configuration object.

        Returns:
            Str: A string containing the resolved identifier.

        Raises:
            ValueError: If an identifier is required but not found, or if a URL is provided but the ID cannot be extracted.
        """
        identifier_to_process, identifier_source_field_name = determine_identifier_and_soruce(config)

        final_model_id = HuggingFaceModel.process_hf_identifier(identifier_to_process)

        if final_model_id is None:
            # This means process_hf_identifier failed (likely due to URL extraction)
            raise ValueError(
                f"Could not extract model ID from URL in '{identifier_source_field_name.name.lower()}': {identifier_to_process}"
            )

        return final_model_id

    
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