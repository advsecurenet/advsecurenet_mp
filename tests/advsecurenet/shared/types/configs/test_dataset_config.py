"""
Tests for advsecurenet.shared.types.configs.dataset_config module.
"""

import sys
import os
from unittest.mock import patch

# Add the module path directly to avoid import issues with __init__.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../../../"))

import pytest

# Import dataset_config directly to avoid torch dependency from __init__.py
from advsecurenet.shared.types.configs.dataset_config import (
    BaseDatasetCliConfig,
    UserSplitConfig,
    CreateDatasetCliConfig,
    AttacksDatasetCliConfig,
    ResolvedSplitConfig,
    ResolvedDatasetConfig,
    _get_identifier,
    _get_user_splits,
    _create_resolved_split,
    resolve_dataset_config,
)

# Import PreprocessConfig directly to avoid torch dependency
from advsecurenet.shared.types.configs.preprocess_config import PreprocessConfig


@pytest.mark.advsecurenet
@pytest.mark.essential
class TestBaseDatasetCliConfig:
    """Test class for BaseDatasetCliConfig."""

    def test_base_dataset_cli_config_creation(self):
        """Test creating BaseDatasetCliConfig with defaults."""
        config = BaseDatasetCliConfig(dataset_name="test_dataset")

        assert config.dataset_name == "test_dataset"
        assert config.num_classes == 10  # default value
        assert config.preprocessing is None

    def test_base_dataset_cli_config_custom_values(self):
        """Test creating BaseDatasetCliConfig with custom values."""
        preprocessing = PreprocessConfig()
        config = BaseDatasetCliConfig(
            dataset_name="custom_dataset", num_classes=5, preprocessing=preprocessing
        )

        assert config.dataset_name == "custom_dataset"
        assert config.num_classes == 5
        assert config.preprocessing == preprocessing


@pytest.mark.advsecurenet
@pytest.mark.essential
class TestUserSplitConfig:
    """Test class for UserSplitConfig."""

    def test_user_split_config_defaults(self):
        """Test UserSplitConfig with default values."""
        config = UserSplitConfig()

        assert config.identifier is None
        assert config.split_name is None
        assert config.preprocessing is None
        assert config.dataset_kwargs is None
        assert config.constructor_args == {}  # default_factory
        assert config.path is None

    def test_user_split_config_custom_values(self):
        """Test UserSplitConfig with custom values."""
        preprocessing = PreprocessConfig()
        dataset_kwargs = {"param": "value"}
        constructor_args = {"arg": "val"}

        config = UserSplitConfig(
            identifier="custom_id",
            split_name="validation",
            preprocessing=preprocessing,
            dataset_kwargs=dataset_kwargs,
            constructor_args=constructor_args,
            path="/custom/path",
        )

        assert config.identifier == "custom_id"
        assert config.split_name == "validation"
        assert config.preprocessing == preprocessing
        assert config.dataset_kwargs == dataset_kwargs
        assert config.constructor_args == constructor_args
        assert config.path == "/custom/path"


@pytest.mark.advsecurenet
@pytest.mark.essential
class TestCreateDatasetCliConfig:
    """Test class for CreateDatasetCliConfig."""

    def test_create_dataset_cli_config_defaults(self):
        """Test CreateDatasetCliConfig with default values."""
        config = CreateDatasetCliConfig(dataset_name="test_dataset")

        assert config.dataset_name == "test_dataset"
        assert config.num_classes == 10
        assert config.preprocessing is None
        assert config.identifier is None
        assert config.dataset_kwargs == {}  # default_factory
        assert config.constructor_args == {}  # default_factory
        assert config.split_config is None
        assert config.load_splits == []  # default_factory
        assert config.random_sample_size is None

    def test_create_dataset_cli_config_custom_values(self):
        """Test CreateDatasetCliConfig with custom values."""
        preprocessing = PreprocessConfig()
        dataset_kwargs = {"key": "value"}
        constructor_args = {"arg": "val"}
        split_config = {"train": UserSplitConfig()}
        load_splits = ["train", "test"]

        config = CreateDatasetCliConfig(
            dataset_name="custom_dataset",
            num_classes=20,
            preprocessing=preprocessing,
            identifier="custom_id",
            dataset_kwargs=dataset_kwargs,
            constructor_args=constructor_args,
            split_config=split_config,
            load_splits=load_splits,
            random_sample_size=1000,
        )

        assert config.dataset_name == "custom_dataset"
        assert config.num_classes == 20
        assert config.preprocessing == preprocessing
        assert config.identifier == "custom_id"
        assert config.dataset_kwargs == dataset_kwargs
        assert config.constructor_args == constructor_args
        assert config.split_config == split_config
        assert config.load_splits == load_splits
        assert config.random_sample_size == 1000


@pytest.mark.advsecurenet
@pytest.mark.essential
class TestAttacksDatasetCliConfig:
    """Test class for AttacksDatasetCliConfig."""

    def test_attacks_dataset_cli_config_creation(self):
        """Test creating AttacksDatasetCliConfig."""
        config = AttacksDatasetCliConfig(dataset_name="attack_dataset")

        assert config.dataset_name == "attack_dataset"
        assert config.num_classes == 10
        assert config.random_sample_size is None

    def test_attacks_dataset_cli_config_with_sample_size(self):
        """Test AttacksDatasetCliConfig with random_sample_size."""
        config = AttacksDatasetCliConfig(
            dataset_name="attack_dataset", random_sample_size=500
        )

        assert config.dataset_name == "attack_dataset"
        assert config.random_sample_size == 500


@pytest.mark.advsecurenet
@pytest.mark.essential
class TestResolvedSplitConfig:
    """Test class for ResolvedSplitConfig."""

    def test_resolved_split_config_defaults(self):
        """Test ResolvedSplitConfig with default values."""
        config = ResolvedSplitConfig(identifier="test_id")

        assert config.identifier == "test_id"
        assert config.num_classes == 10
        assert config.source_split_name is None
        assert config.preprocessing is None
        assert config.kwargs == {}
        assert config.constructor_args == {}
        assert config.path is None

    def test_resolved_split_config_custom_values(self):
        """Test ResolvedSplitConfig with custom values."""
        preprocessing = PreprocessConfig()
        kwargs = {"param": "value"}
        constructor_args = {"arg": "val"}

        config = ResolvedSplitConfig(
            identifier="custom_id",
            num_classes=5,
            source_split_name="validation",
            preprocessing=preprocessing,
            kwargs=kwargs,
            constructor_args=constructor_args,
            path="/custom/path",
        )

        assert config.identifier == "custom_id"
        assert config.num_classes == 5
        assert config.source_split_name == "validation"
        assert config.preprocessing == preprocessing
        assert config.kwargs == kwargs
        assert config.constructor_args == constructor_args
        assert config.path == "/custom/path"


@pytest.mark.advsecurenet
@pytest.mark.essential
class TestResolvedDatasetConfig:
    """Test class for ResolvedDatasetConfig."""

    def test_resolved_dataset_config_creation(self):
        """Test creating ResolvedDatasetConfig."""
        splits = {
            "train": ResolvedSplitConfig(identifier="test_id"),
            "test": ResolvedSplitConfig(identifier="test_id"),
        }

        config = ResolvedDatasetConfig(dataset_name="test_dataset", splits=splits)

        assert config.dataset_name == "test_dataset"
        assert config.splits == splits
        assert config.random_sample_size is None

    def test_resolved_dataset_config_with_sample_size(self):
        """Test ResolvedDatasetConfig with random_sample_size."""
        splits = {"train": ResolvedSplitConfig(identifier="test_id")}

        config = ResolvedDatasetConfig(
            dataset_name="test_dataset", splits=splits, random_sample_size=1000
        )

        assert config.random_sample_size == 1000


@pytest.mark.advsecurenet
@pytest.mark.essential
class TestHelperFunctions:
    """Test class for helper functions."""

    def test_get_identifier_with_identifier(self):
        """Test _get_identifier when identifier is provided."""
        config = CreateDatasetCliConfig(dataset_name="dataset", identifier="custom_id")

        result = _get_identifier(config)
        assert result == "custom_id"

    def test_get_identifier_without_identifier(self):
        """Test _get_identifier when identifier is None."""
        config = CreateDatasetCliConfig(dataset_name="dataset")

        result = _get_identifier(config)
        assert result == "dataset"

    def test_get_user_splits_from_split_config(self):
        """Test _get_user_splits with split_config (highest priority)."""
        split_config = {"training": UserSplitConfig(), "validation": UserSplitConfig()}
        config = CreateDatasetCliConfig(
            dataset_name="dataset",
            split_config=split_config,
            load_splits=["ignored", "splits"],  # should be ignored
        )

        result = _get_user_splits(config)
        assert result == ["training", "validation"]

    def test_get_user_splits_from_load_splits(self):
        """Test _get_user_splits with load_splits (second priority)."""
        config = CreateDatasetCliConfig(
            dataset_name="dataset", load_splits=["train", "test", "holdout"]
        )

        result = _get_user_splits(config)
        assert result == ["train", "test", "holdout"]

    def test_get_user_splits_default(self):
        """Test _get_user_splits with default behavior."""
        config = CreateDatasetCliConfig(dataset_name="dataset")

        result = _get_user_splits(config)
        assert result == ["train", "test"]

    def test_get_user_splits_empty_split_config(self):
        """Test _get_user_splits with empty split_config."""
        config = CreateDatasetCliConfig(
            dataset_name="dataset",
            split_config={},  # empty dict
            load_splits=["custom", "splits"],
        )

        result = _get_user_splits(config)
        # Empty split_config should fall back to load_splits
        assert result == ["custom", "splits"]

    def test_get_user_splits_empty_load_splits(self):
        """Test _get_user_splits with empty load_splits."""
        config = CreateDatasetCliConfig(
            dataset_name="dataset", load_splits=[]  # empty list
        )

        result = _get_user_splits(config)
        # Empty load_splits should fall back to default
        assert result == ["train", "test"]


@pytest.mark.advsecurenet
@pytest.mark.essential
class TestCreateResolvedSplit:
    """Test class for _create_resolved_split function."""

    def test_create_resolved_split_without_user_config(self):
        """Test _create_resolved_split without user split config."""
        global_config = CreateDatasetCliConfig(
            dataset_name="global_dataset",
            num_classes=5,
            preprocessing=PreprocessConfig(),
            dataset_kwargs={"global": "param"},
            constructor_args={"global": "arg"},
        )

        result = _create_resolved_split(global_config, "train")

        assert result.identifier == "global_dataset"
        assert result.source_split_name == "train"
        assert result.num_classes == 5
        assert result.preprocessing == global_config.preprocessing
        assert result.kwargs == {"global": "param"}
        assert result.constructor_args == {"global": "arg"}
        assert result.path is None

    def test_create_resolved_split_with_user_config_overrides(self):
        """Test _create_resolved_split with user split config overrides."""
        global_preprocessing = PreprocessConfig()
        split_preprocessing = PreprocessConfig()

        global_config = CreateDatasetCliConfig(
            dataset_name="global_dataset",
            identifier="global_id",
            num_classes=5,
            preprocessing=global_preprocessing,
            dataset_kwargs={"global": "param"},
            constructor_args={"global": "arg"},
        )

        user_split_config = UserSplitConfig(
            identifier="split_id",
            split_name="validation",
            preprocessing=split_preprocessing,
            dataset_kwargs={"split": "param"},
            constructor_args={"split": "arg"},
            path="/split/path",
        )

        result = _create_resolved_split(global_config, "test", user_split_config)

        # Should use split-specific values
        assert result.identifier == "split_id"
        assert result.source_split_name == "validation"
        assert result.num_classes == 5  # from global
        assert result.preprocessing == split_preprocessing
        assert result.kwargs == {"split": "param"}
        assert result.constructor_args == {"split": "arg"}
        assert result.path == "/split/path"

    def test_create_resolved_split_with_partial_user_config(self):
        """Test _create_resolved_split with partial user split config."""
        global_preprocessing = PreprocessConfig()

        global_config = CreateDatasetCliConfig(
            dataset_name="global_dataset",
            identifier="global_id",
            preprocessing=global_preprocessing,
            dataset_kwargs={"global": "param"},
            constructor_args={"global": "arg"},
        )

        user_split_config = UserSplitConfig(
            identifier="split_id",
            # split_name is None, should fall back to original split_name
            # preprocessing is None, should fall back to global
            dataset_kwargs={"split": "param"},
            # constructor_args is None, should fall back to global
        )

        result = _create_resolved_split(global_config, "train", user_split_config)

        assert result.identifier == "split_id"  # from split
        assert result.source_split_name == "train"  # fallback to original
        assert result.preprocessing == global_preprocessing  # fallback to global
        assert result.kwargs == {"split": "param"}  # from split
        assert result.constructor_args == {"global": "arg"}  # fallback to global

    def test_create_resolved_split_with_none_values(self):
        """Test _create_resolved_split with None values in global config."""
        global_config = CreateDatasetCliConfig(
            dataset_name="dataset", dataset_kwargs=None, constructor_args=None
        )

        result = _create_resolved_split(global_config, "train")

        assert result.identifier == "dataset"
        assert result.kwargs == {}  # None becomes empty dict
        assert result.constructor_args == {}  # None becomes empty dict


@pytest.mark.advsecurenet
@pytest.mark.essential
class TestResolveDatasetConfig:
    """Test class for resolve_dataset_config function."""

    def test_resolve_dataset_config_simple(self):
        """Test resolve_dataset_config with simple configuration."""
        config = CreateDatasetCliConfig(
            dataset_name="test_dataset", num_classes=10, load_splits=["train", "test"]
        )

        result = resolve_dataset_config(config)

        assert result.dataset_name == "test_dataset"
        assert result.random_sample_size is None
        assert len(result.splits) == 2
        assert "train" in result.splits
        assert "test" in result.splits

        # Check train split
        train_split = result.splits["train"]
        assert train_split.identifier == "test_dataset"
        assert train_split.source_split_name == "train"
        assert train_split.num_classes == 10

    def test_resolve_dataset_config_with_split_config(self):
        """Test resolve_dataset_config with split_config."""
        user_split = UserSplitConfig(identifier="custom_id", split_name="validation")
        config = CreateDatasetCliConfig(
            dataset_name="test_dataset", split_config={"custom_split": user_split}
        )

        result = resolve_dataset_config(config)

        assert len(result.splits) == 1
        assert "custom_split" in result.splits

        custom_split = result.splits["custom_split"]
        assert custom_split.identifier == "custom_id"
        assert custom_split.source_split_name == "validation"

    def test_resolve_dataset_config_with_random_sample_size(self):
        """Test resolve_dataset_config with random_sample_size."""
        config = CreateDatasetCliConfig(
            dataset_name="test_dataset", random_sample_size=500, load_splits=["train"]
        )

        result = resolve_dataset_config(config)

        assert result.random_sample_size == 500

    def test_resolve_dataset_config_no_splits_raises_error(self):
        """Test resolve_dataset_config raises error when no splits are defined."""
        config = CreateDatasetCliConfig(dataset_name="test_dataset")

        # Mock _get_user_splits to return empty list to trigger the error condition
        with patch(
            "advsecurenet.shared.types.configs.dataset_config._get_user_splits",
            return_value=[],
        ):
            with pytest.raises(ValueError, match="No splits defined for dataset"):
                resolve_dataset_config(config)

    def test_resolve_dataset_config_complex_scenario(self):
        """Test resolve_dataset_config with complex configuration."""
        global_preprocessing = PreprocessConfig()
        split_preprocessing = PreprocessConfig()

        train_split = UserSplitConfig(
            identifier="train_custom_id",
            preprocessing=split_preprocessing,
            dataset_kwargs={"train_param": "value"},
        )

        test_split = UserSplitConfig(split_name="validation", path="/test/path")

        config = CreateDatasetCliConfig(
            dataset_name="complex_dataset",
            identifier="global_id",
            num_classes=20,
            preprocessing=global_preprocessing,
            dataset_kwargs={"global_param": "value"},
            constructor_args={"global_arg": "value"},
            split_config={"train": train_split, "test": test_split},
            random_sample_size=1000,
        )

        result = resolve_dataset_config(config)

        assert result.dataset_name == "complex_dataset"
        assert result.random_sample_size == 1000
        assert len(result.splits) == 2

        # Check train split (has overrides)
        train_resolved = result.splits["train"]
        assert train_resolved.identifier == "train_custom_id"
        assert train_resolved.source_split_name == "train"  # fallback
        assert train_resolved.preprocessing == split_preprocessing
        assert train_resolved.kwargs == {"train_param": "value"}
        assert train_resolved.num_classes == 20

        # Check test split (partial overrides)
        test_resolved = result.splits["test"]
        assert test_resolved.identifier == "global_id"  # fallback
        assert test_resolved.source_split_name == "validation"  # override
        assert test_resolved.preprocessing == global_preprocessing  # fallback
        assert test_resolved.path == "/test/path"
        assert test_resolved.num_classes == 20
