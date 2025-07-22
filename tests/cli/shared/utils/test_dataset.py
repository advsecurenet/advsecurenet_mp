from unittest.mock import MagicMock, patch

import pytest
from torch.utils.data import Dataset as TorchDataset

from advsecurenet.shared.types.dataset import DatasetType
from cli.shared.types.utils.dataset import (
    AttacksDatasetCliConfig,
    CreateDatasetCliConfig,
    UserSplitConfig,
    ResolvedDatasetConfig,
    ResolvedSplitConfig,
)
from cli.shared.utils.dataset import get_datasets


@pytest.mark.cli
@pytest.mark.essential
@patch("cli.shared.utils.dataset.DatasetFactory.load_dataset_from_config")
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
@patch("cli.shared.utils.dataset.DatasetFactory.load_dataset_from_config")
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
