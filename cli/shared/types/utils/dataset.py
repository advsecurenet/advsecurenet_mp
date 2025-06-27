from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum, auto

from advsecurenet.shared.types.configs.preprocess_config import PreprocessConfig

# ----------------------------------------------------------------
# 1. User-Facing Configuration Classes (for YAML files)
# ----------------------------------------------------------------

@dataclass
class BaseDatasetCliConfig:
    """
    Base configuration containing the most essential, non-optional dataset properties.
    """
    dataset_name: str
    num_classes: int = 10
    preprocessing: Optional[PreprocessConfig] = None

@dataclass
class UserSplitConfig:
    """
    User-provided overrides for a single logical split (e.g., 'train' or 'test').
    Any field set here will override the global setting for that specific split.
    """
    # The name of the dataset to use for this split, if different from the global one.
    identifier: Optional[str] = None
    # The actual split name required by the library (e.g., 'validation' for a logical 'test' split).
    split_name: Optional[str] = None
    # Custom preprocessing configuration for this split.
    preprocessing: Optional[PreprocessConfig] = None
    # Library-specific keyword arguments for this split (e.g., for HuggingFace).
    dataset_kwargs: Optional[Dict[str, Any]] = None
    # Additional constructor arguments for the dataset class, if needed.
    constructor_args: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class CreateDatasetCliConfig(BaseDatasetCliConfig):
    """
    The comprehensive, user-facing configuration for creating datasets, typically
    loaded from a YAML file. It supports both simple and advanced setups.
    """
    # An optional, alternative identifier for the dataset (e.g., a HuggingFace repo ID).
    # If provided, this will be used instead of `dataset_name` for loading.
    identifier: Optional[str] = None
    # Global library-specific keyword arguments that apply to all splits.
    dataset_kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    # Additional constructor arguments for the dataset class, if needed.
    constructor_args: Optional[Dict[str, Any]] = field(default_factory=dict)
    # A dictionary of split-specific overrides. The keys are the logical split names
    # (e.g., "train") and the values are UserSplitConfig objects.
    split_config: Optional[Dict[str, UserSplitConfig]] = None
    load_splits: List[str] = field(default_factory=list)
    # Attack-specific: The size of the random sample to take from the dataset.
    random_sample_size: Optional[int] = None

@dataclass
class AttacksDatasetCliConfig(CreateDatasetCliConfig):
    """
    A specialized configuration for attacks, adding attack-specific parameters.
    """
    random_sample_size: Optional[int] = None

# ----------------------------------------------------------------
# 2. Internal, Resolved Configuration Classes (for application use)
# ----------------------------------------------------------------

@dataclass
class ResolvedSplitConfig:
    """
    Internal, fully-resolved configuration for a single dataset split.
    This object contains all the final, unambiguous settings needed to load one split.
    """
    identifier: str
    num_classes: int
    source_split_name: Optional[str] = None
    preprocessing: Optional[PreprocessConfig] = None
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    constructor_args: Optional[Dict[str, Any]] = field(default_factory=dict)

@dataclass
class ResolvedDatasetConfig:
    """
    The final, resolved configuration object used by the application's loading logic.
    It is the result of processing a CreateDatasetCliConfig.
    """
    dataset_name: str
    splits: Dict[str, ResolvedSplitConfig]
    random_sample_size: Optional[int] = None

# ----------------------------------------------------------------
# 3. Helper Enums and Functions
# ----------------------------------------------------------------

class IdentifierSource(Enum):
    """Enumeration to track the source of the dataset identifier."""
    IDENTIFIER = auto()
    NAME = auto()

def determine_identifier_and_source(config: CreateDatasetCliConfig) -> tuple[str, IdentifierSource]:
    """Determines the primary dataset identifier to use from the configuration."""
    if config.identifier:
        return config.identifier, IdentifierSource.IDENTIFIER
    return config.dataset_name, IdentifierSource.NAME

def resolve_dataset_config(config: CreateDatasetCliConfig) -> ResolvedDatasetConfig:
    """
    Resolves a user-facing CreateDatasetCliConfig into a structured,
    internal ResolvedDatasetConfig. It applies global defaults and handles
    split-specific overrides.
    """
    identifier, _ = determine_identifier_and_source(config)

    if not config.load_splits and config.split_config is None:
        splits = ["train", "test"]
    elif len(config.load_splits) > 2:
        splits = config.load_splits[:2]
        Warning(
            f"More than 2 splits provided: {config.load_splits}. Only the first two splits will be used: {splits}.")
    else:
        splits = config.load_splits

    if config.split_config is not None and len(config.split_config) > 2:
        Warning(
            f"More than 2 split configurations provided: {config.split_config}. Only the first two will be used: {config.split_config[:2]}.")

    internal_splits = ["train", "test"]

    final_splits = {}
    if config.split_config is None:
        for split_name, internal_name in zip(splits, internal_splits):
            split = ResolvedSplitConfig(
                identifier=identifier,
                source_split_name=split_name,
                preprocessing=config.preprocessing,
                kwargs=config.dataset_kwargs or {},
                num_classes=config.num_classes,
                constructor_args=config.constructor_args or {}
            )
            final_splits[internal_name] = split
    else:
        # If a specific split config is provided, map its entries to our internal roles.
        user_split_configs = list(config.split_config.values())
        for i, internal_name in enumerate(internal_splits):
            if i >= len(user_split_configs):
                break  # Stop if the user provided fewer splits than we have internal roles for.

            user_config = user_split_configs[i]

            # Create a ResolvedSplitConfig by overriding global settings with split-specific ones.
            resolved_split = ResolvedSplitConfig(
                identifier=user_config.identifier or identifier,
                source_split_name=user_config.split_name,
                preprocessing=user_config.preprocessing or config.preprocessing,
                kwargs=user_config.dataset_kwargs or config.dataset_kwargs or {},
                num_classes=config.num_classes,
                constructor_args=user_config.constructor_args or config.constructor_args or {}
            )
            final_splits[internal_name] = resolved_split

    return ResolvedDatasetConfig(
        dataset_name=config.dataset_name,
        splits=final_splits,
        random_sample_size=config.random_sample_size
    )
