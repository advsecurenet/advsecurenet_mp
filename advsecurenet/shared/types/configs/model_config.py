from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum, auto


@dataclass
class BaseModelConfig:
    """
    Base configuration class for different model configurations.
    """

    model_name: str
    pretrained: Optional[bool] = False
    architecture: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StandardModelConfig(BaseModelConfig):
    """
    Configuration for a standard model.
    """

    weights: Optional[str] = "IMAGENET1K_V1"


@dataclass
class CustomModelConfig(BaseModelConfig):
    """
    Configuration for a custom model.
    """

    custom_models_path: Optional[str] = "CustomModels"


@dataclass
class ExternalModelConfig(BaseModelConfig):
    """
    Configuration for an external model.
    """

    model_arch_path: Optional[str] = None
    model_weights_path: Optional[str] = None

@dataclass
class HuggingFaceInputConfig(BaseModelConfig):
    """
    User-provided configuration parameters for Hugging Face models,
    used prior to the resolution of the specific model_id.
    This config is typically part of the broader CreateModelConfig.
    """
    pretrained: Optional[bool] = True
    revision: Optional[str] = None
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False
    model_class_name: str = None

@dataclass
class HuggingFaceResolvedConfig(HuggingFaceInputConfig):
    """
    Fully resolved configuration for Hugging Face models, including the model_id.
    This is the config type expected by HuggingFaceModel.__init__.
    """
    model_id: str = None


@dataclass
class CreateModelConfig(StandardModelConfig, CustomModelConfig, ExternalModelConfig, HuggingFaceInputConfig):
    """
    Config parameters for creating a model in the model factory.
    """

    is_external: bool = False
    random_seed: Optional[int] = None
    model_identifier: str = None

class IdentifierSource(Enum):
    MODEL_IDENTIFIER = auto()
    MODEL_NAME = auto()

@staticmethod
def determine_identifier_and_soruce(config: CreateModelConfig):
    if config.model_identifier:
        identifier = config.model_identifier
        source = IdentifierSource.MODEL_IDENTIFIER
    else:
        identifier = config.model_name
        source = IdentifierSource.MODEL_NAME
    return identifier, source