from unittest.mock import MagicMock, patch

import pytest
from torch.utils.data import Dataset as TorchDataset

from advsecurenet.shared.types.dataset import DatasetType
from cli.shared.types.utils.dataset import (
    AttacksDatasetCliConfig,
    CreateDatasetCliConfig,
    UserSplitConfig,
    ResolvedDatasetConfig,
    ResolvedSplitConfig
)
from cli.shared.utils.dataset import get_datasets


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.dataset.DatasetFactory.load_dataset")
def test_get_datasets_standard(mock_create_dataset):
    mock_dataset = MagicMock()
    mock_create_dataset.return_value = mock_dataset
    mock_config = CreateDatasetCliConfig(
        dataset_name="CIFAR10",
        num_classes=10,
        preprocessing=MagicMock(),
        split_config={
            "train": UserSplitConfig(split_name = "train", dataset_kwargs = {"root": "path/to/train"}),
            "test": UserSplitConfig(split_name = "test", dataset_kwargs = {"root": "path/to/test"})
        }
    )

    mock_dataset.load_dataset.side_effect = [
        MagicMock(spec=TorchDataset),
        MagicMock(spec=TorchDataset),
    ]

    train_data, test_data = get_datasets(mock_config)

    mock_create_dataset.assert_called_once_with(
        dataset_type=DatasetType.CIFAR10, preprocess_config=mock_config.preprocessing
    )
    assert isinstance(train_data, TorchDataset)
    assert isinstance(test_data, TorchDataset)


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.dataset.DatasetFactory.load_dataset")
def test_get_datasets_attacks(mock_create_dataset):
    mock_dataset = MagicMock()
    mock_create_dataset.return_value = mock_dataset
    mock_config = AttacksDatasetCliConfig(
        dataset_name="CIFAR10",
        num_classes=10,
        preprocessing=MagicMock(),
        split_config={
            "train": UserSplitConfig(split_name = "train", dataset_kwargs = {"root": "path/to/train"}),
            "test": UserSplitConfig(split_name = "test", dataset_kwargs = {"root": "path/to/test"})
        }
    )

    mock_dataset.load_dataset.side_effect = [
        MagicMock(spec=TorchDataset),
        MagicMock(spec=TorchDataset),
    ]

    train_data, test_data = get_datasets(mock_config)

    mock_create_dataset.assert_called_once_with(
        dataset_type=DatasetType.CIFAR10, preprocess_config=mock_config.preprocessing
    )
    mock_dataset.load_dataset.assert_any_call(
        train=True, root="path/to/train", download=True
    )
    mock_dataset.load_dataset.assert_any_call(
        train=False, root="path/to/test", download=True
    )
    assert isinstance(train_data, TorchDataset)
    assert isinstance(test_data, TorchDataset)


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.dataset.DatasetFactory.load_dataset_from_config")
def test_get_datasets_file_not_found(mock_create_dataset):
    mock_dataset = MagicMock()
    preprocessing_mock = MagicMock()
    mock_create_dataset.return_value = mock_dataset
    mock_config = CreateDatasetCliConfig(
        dataset_name="CIFAR10",
        num_classes=10,
        preprocessing=preprocessing_mock,
        split_config={
            "train": UserSplitConfig(split_name = "train", dataset_kwargs = {"root": "path/to/train"}),
            "test": UserSplitConfig(split_name = "test", dataset_kwargs = {"root": "path/to/test"})
        }
    )

    mock_dataset.load_dataset_from_config.side_effect = FileNotFoundError

    train_data, test_data = get_datasets(mock_config)

    resolved_config = ResolvedDatasetConfig(
        dataset_name="CIFAR10",
        splits={
            "train": ResolvedSplitConfig(identifier="CIFAR10", source_split_name="train", kwargs={"root": "path/to/train"}, num_classes=10, preprocessing=preprocessing_mock),
            "test": ResolvedSplitConfig(identifier="CIFAR10", source_split_name="test", kwargs = {"root": "path/to/test"}, num_classes=10, preprocessing=preprocessing_mock)
        }
    )

    mock_create_dataset.assert_called_once_with(resolved_config=resolved_config)
    assert train_data is None
    assert test_data is None
