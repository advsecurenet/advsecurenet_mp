from unittest.mock import MagicMock, patch

import pytest
from torch.utils.data import Dataset as TorchDataset

from advsecurenet.shared.types.dataset import DatasetType
from advsecurenet.shared.types.configs.dataset_config import (
    AttacksDatasetCliConfig,
    CreateDatasetCliConfig,
    UserSplitConfig,
    ResolvedDatasetConfig,
    ResolvedSplitConfig,
)
from cli.shared.utils.dataset import get_datasets
from advsecurenet.shared.types.configs.dataset_config import (
    _get_identifier,
    _create_resolved_split,
    _get_user_splits,
)


@pytest.mark.cli
@pytest.mark.essential
@patch("advsecurenet.datasets.DatasetFactory.load_dataset_from_config")
def test_get_datasets_standard(mock_create_dataset):
    mock_dataset = {
        "train": MagicMock(spec=TorchDataset),
        "test": MagicMock(spec=TorchDataset),
    }

    preprocessing_mock = MagicMock()

    mock_create_dataset.return_value = mock_dataset
    mock_config = CreateDatasetCliConfig(
        dataset_name="CIFAR10",
        num_classes=10,
        preprocessing=preprocessing_mock,
        split_config={
            "train": UserSplitConfig(
                split_name="train", dataset_kwargs={"root": "path/to/train"}
            ),
            "test": UserSplitConfig(
                split_name="test", dataset_kwargs={"root": "path/to/test"}
            ),
        },
    )

    train_data, test_data = get_datasets(mock_config)

    resolved_config = ResolvedDatasetConfig(
        dataset_name="CIFAR10",
        splits={
            "train": ResolvedSplitConfig(
                identifier="CIFAR10",
                source_split_name="train",
                kwargs={"root": "path/to/train"},
                num_classes=10,
                preprocessing=preprocessing_mock,
            ),
            "test": ResolvedSplitConfig(
                identifier="CIFAR10",
                source_split_name="test",
                kwargs={"root": "path/to/test"},
                num_classes=10,
                preprocessing=preprocessing_mock,
            ),
        },
    )

    mock_create_dataset.assert_called_once_with(resolved_config=resolved_config)
    assert isinstance(train_data, TorchDataset)
    assert isinstance(test_data, TorchDataset)


@pytest.mark.cli
@pytest.mark.essential
@patch("advsecurenet.datasets.DatasetFactory.load_dataset_from_config")
def test_get_datasets_attacks(mock_create_dataset):
    mock_dataset = {
        "train": MagicMock(spec=TorchDataset),
        "test": MagicMock(spec=TorchDataset),
    }

    mock_create_dataset.return_value = mock_dataset
    preprocessing_mock = MagicMock()
    mock_config = AttacksDatasetCliConfig(
        dataset_name="CIFAR10",
        num_classes=10,
        preprocessing=preprocessing_mock,
        random_sample_size=100,
        split_config={
            "train": UserSplitConfig(
                split_name="train", dataset_kwargs={"root": "path/to/train"}
            ),
            "test": UserSplitConfig(
                split_name="test", dataset_kwargs={"root": "path/to/test"}
            ),
        },
    )

    train_data, test_data = get_datasets(mock_config)

    resolved_config = ResolvedDatasetConfig(
        dataset_name="CIFAR10",
        random_sample_size=100,
        splits={
            "train": ResolvedSplitConfig(
                identifier="CIFAR10",
                source_split_name="train",
                kwargs={"root": "path/to/train"},
                num_classes=10,
                preprocessing=preprocessing_mock,
            ),
            "test": ResolvedSplitConfig(
                identifier="CIFAR10",
                source_split_name="test",
                kwargs={"root": "path/to/test"},
                num_classes=10,
                preprocessing=preprocessing_mock,
            ),
        },
    )

    mock_create_dataset.assert_called_once_with(resolved_config=resolved_config)

    assert isinstance(train_data, TorchDataset)
    assert isinstance(test_data, TorchDataset)


@pytest.mark.cli
@pytest.mark.essential
def test_get_identifier_with_identifier():
    """Test _get_identifier returns config.identifier when provided"""
    config = CreateDatasetCliConfig(
        dataset_name="CIFAR10", identifier="custom-identifier"
    )
    result = _get_identifier(config)
    assert result == "custom-identifier"


@pytest.mark.cli
@pytest.mark.essential
def test_get_identifier_without_identifier():
    """Test _get_identifier returns dataset_name when no identifier"""
    config = CreateDatasetCliConfig(dataset_name="CIFAR10")
    result = _get_identifier(config)
    assert result == "CIFAR10"


@pytest.mark.cli
@pytest.mark.essential
def test_get_user_splits_default():
    """Test _get_user_splits returns default ['train', 'test'] when no split config or load_splits"""
    config = CreateDatasetCliConfig(dataset_name="CIFAR10")
    result = _get_user_splits(config)
    assert result == ["train", "test"]


@pytest.mark.cli
@pytest.mark.essential
def test_get_user_splits_with_load_splits():
    """Test _get_user_splits returns load_splits when provided"""
    config = CreateDatasetCliConfig(
        dataset_name="CIFAR10", load_splits=["train", "validation", "test"]
    )
    result = _get_user_splits(config)
    assert result == ["train", "validation", "test"]


@pytest.mark.cli
@pytest.mark.essential
def test_create_resolved_split_global_only():
    """Test _create_resolved_split with global settings only (no user_split_config)"""

    preprocessing_mock = MagicMock()
    global_config = CreateDatasetCliConfig(
        dataset_name="CIFAR10",
        num_classes=10,
        preprocessing=preprocessing_mock,
        dataset_kwargs={"download": True},
        constructor_args={"arg1": "value1"},
    )

    result = _create_resolved_split(global_config, "train", None)

    assert result.identifier == "CIFAR10"
    assert result.source_split_name == "train"
    assert result.preprocessing == preprocessing_mock
    assert result.kwargs == {"download": True}
    assert result.constructor_args == {"arg1": "value1"}
    assert result.num_classes == 10
    assert result.path is None
