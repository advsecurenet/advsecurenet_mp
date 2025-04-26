from dataclasses import dataclass, field
from typing import Optional, Dict, Any


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

    def __post_init__(self):
        self.architecture.setdefault("num_input_channels", 3)


@dataclass
class ExternalModelConfig(BaseModelConfig):
    """
    Configuration for an external model.
    """

    model_arch_path: Optional[str] = None
    model_weights_path: Optional[str] = None

    def __post_init__(self):
        self.architecture.setdefault("num_input_channels", 3)

@dataclass
class HuggingFaceModelConfig(BaseModelConfig):
    """
    Configuration for Hugging Face models.
    """
    pretrained: Optional[bool] = True

    model_url: str = None
    revision: Optional[str] = None
    cache_dir: Optional[str] = None
    trust_remote_code: bool = False


@dataclass
class CreateModelConfig(StandardModelConfig, CustomModelConfig, ExternalModelConfig, HuggingFaceModelConfig):
    """
    Config parameters for creating a model in the model factory.
    """

    is_external: bool = False
    random_seed: Optional[int] = None