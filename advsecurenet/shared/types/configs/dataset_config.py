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
    path: Optional[str] = None



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
    num_classes: Optional[int] = 10
    source_split_name: Optional[str] = None
    preprocessing: Optional[PreprocessConfig] = None
    kwargs: Optional[Dict[str, Any]] = field(default_factory=dict)
    constructor_args: Optional[Dict[str, Any]] = field(default_factory=dict)
    path: Optional[str] = None



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
# 3. Helper and Functions
# ----------------------------------------------------------------



def _get_identifier(config: CreateDatasetCliConfig) -> str:
    """Determines the primary dataset identifier to use from the global configuration."""
    if config.identifier:
        return config.identifier
    return config.dataset_name



def _get_user_splits(config: CreateDatasetCliConfig) -> List[str]:
    """
    Determines the splits to load based on the user configuration.

    This function extracts the logical split names that the user wants to load,
    following a priority order to determine which splits configuration to use.
    The split names can be any custom names defined by the user.

    Priority Order:
        1. Keys from `split_config` dictionary (if provided)
        2. Values from `load_splits` list (if explicitly provided and not empty)
        3. Default: ['train', 'test']

    Args:
        config (CreateDatasetCliConfig): The user-facing dataset configuration.

    Returns:
        List[str]: A list of logical split names to load. These names will be used
                  as keys in the final ResolvedDatasetConfig.splits dictionary.

    Note:
        Split Name Flexibility:
        - The split names can be any custom names (e.g., 'validation', 'holdout', 'custom_split')
        - However, CLI utilities in `cli/shared/utils/dataset.py` currently only retrieve
          'train' and 'test' splits by default
        - When using the API directly (as in `examples/advsecurenet/benign_training/benign_training.ipynb`),
          any split name works as long as it matches the dictionary key name
        - For custom split names to work with CLI utilities, the codebase would need updates
          to handle non-standard split names

    Examples:
        >>> # Using split_config (highest priority) - typically created by factory
        >>> # This would normally be populated by load_dataset() factory function
        >>> config = CreateDatasetCliConfig(
        ...     dataset_name="cifar10",
        ...     split_config={"training": UserSplitConfig(), "validation": UserSplitConfig()}
        ... )
        >>> _get_user_splits(config)
        ['training', 'validation']

        >>> # Using load_splits (second priority) - direct API usage
        >>> config = CreateDatasetCliConfig(
        ...     dataset_name="cifar10",
        ...     load_splits=["train", "test", "holdout"]
        ... )
        >>> _get_user_splits(config)
        ['train', 'test', 'holdout']

        >>> # Default case (no configuration provided)
        >>> config = CreateDatasetCliConfig(dataset_name="cifar10")
        >>> _get_user_splits(config)
        ['train', 'test']
    """
    # First priority: use the keys from split_config if it's provided and has keys.
    if config.split_config:
        return list(config.split_config.keys())


    # Second priority: use load_splits if it's explicitly provided and not empty.
    if config.load_splits:
        return config.load_splits

    # Default case: if neither of the above is provided, default to train and test.
    return ["train", "test"]



def _create_resolved_split(
    global_config: CreateDatasetCliConfig,
    split_name: str,
    user_split_config: Optional[UserSplitConfig] = None,
) -> ResolvedSplitConfig:
    """
    Creates a single ResolvedSplitConfig, overriding global settings with
    split-specific settings if provided.
    """
    global_identifier = _get_identifier(global_config)
    if user_split_config:
        # Use split-specific values, falling back to global ones.
        identifier = user_split_config.identifier or global_identifier
        source_split_name = user_split_config.split_name or split_name
        preprocessing = user_split_config.preprocessing or global_config.preprocessing
        kwargs = user_split_config.dataset_kwargs or global_config.dataset_kwargs or {}
        constructor_args = (
            user_split_config.constructor_args or global_config.constructor_args or {}
        )
        constructor_args = (
            user_split_config.constructor_args or global_config.constructor_args or {}
        )
        path = user_split_config.path
    else:
        # Use global settings only.
        identifier = global_identifier
        source_split_name = split_name
        preprocessing = global_config.preprocessing
        kwargs = global_config.dataset_kwargs or {}
        constructor_args = global_config.constructor_args or {}
        path = None

    return ResolvedSplitConfig(
        identifier=identifier,
        source_split_name=source_split_name,
        preprocessing=preprocessing,
        kwargs=kwargs,
        num_classes=global_config.num_classes,
        constructor_args=constructor_args,
        path=path,
    )



def resolve_dataset_config(config: CreateDatasetCliConfig) -> ResolvedDatasetConfig:
    """
    Resolves a user-facing CreateDatasetCliConfig into a structured,
    internal ResolvedDatasetConfig. It applies global defaults and handles
    split-specific overrides.
    """
    user_splits_to_process = _get_user_splits(config)
    
    if not user_splits_to_process:
        raise ValueError(
            f"No splits defined for dataset '{config.dataset_name}'. "
            "Please specify splits in 'split_config' or 'load_splits'."
        )

    final_splits = {}

    for split_name in user_splits_to_process:
        # Get the specific configuration for this split, if it exists.
        user_split_config = (
            config.split_config.get(split_name) if config.split_config else None
        )

        # Create the final, resolved configuration for this split.
        resolved_split = _create_resolved_split(
            global_config=config,
            split_name=split_name,
            user_split_config=user_split_config,
        )
        # The key in the final dictionary is the split name itself (e.g., 'train', 'test').
        final_splits[split_name] = resolved_split

    return ResolvedDatasetConfig(
        dataset_name=config.dataset_name,
        splits=final_splits,
        random_sample_size=config.random_sample_size,
    )

