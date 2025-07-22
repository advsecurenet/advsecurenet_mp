from dataclasses import dataclass
from typing import List, Optional, Any, Dict


@dataclass
class ModelNormConfig:
    """
    This dataclass is used to store the configuration of the normalization layer of a model.
    """

    add_norm_layer: bool
    norm_mean: Optional[List[float]]
    norm_std: Optional[List[float]]


@dataclass
class ModelPathConfig:
    """
    This dataclass is used to store the configuration of the paths of a model.
    """

    model_arch_path: Optional[str]
    model_weights_path: Optional[str]


@dataclass
class ModelCliConfigType:
    """
    This dataclass is used to store the configuration of the model CLI.
    """

    model_name: str
    pretrained: bool
    is_external: Optional[bool] = False
    weights: Optional[str] = None
    random_seed: Optional[int] = None
    path_configs: Optional[ModelPathConfig] = None
    norm_config: Optional[ModelNormConfig] = None
    architecture: Optional[Dict[str, Any]] = None
