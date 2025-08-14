from unittest.mock import Mock

import pytest
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data.sampler import RandomSampler

from advsecurenet.dataloader.data_loader_factory import DataLoaderFactory, od_collate_fn
from advsecurenet.shared.types.configs.dataloader_config import DataLoaderConfig


class MockDataset(TorchDataset):
    def __len__(self):
        return 10

    def __getitem__(self, item):
        return torch.zeros(1)


@pytest.fixture
def mock_dataset():
    return MockDataset()


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_dataloader_with_config(mock_dataset):
    config = DataLoaderConfig(
        dataset=mock_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        sampler=None,
    )
    dataloader = DataLoaderFactory.create_dataloader(config)
    assert isinstance(dataloader, TorchDataLoader)
    assert dataloader.dataset == mock_dataset
    assert dataloader.batch_size == 32
    assert dataloader.num_workers == 4
    assert dataloader.drop_last is True
    assert dataloader.pin_memory is True
    assert isinstance(dataloader.sampler, RandomSampler)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_dataloader_without_config(mock_dataset):
    dataloader = DataLoaderFactory.create_dataloader(
        dataset=mock_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        sampler=None,
    )
    assert isinstance(dataloader, TorchDataLoader)
    assert dataloader.dataset == mock_dataset
    assert dataloader.batch_size == 32
    assert dataloader.num_workers == 4
    assert dataloader.drop_last is True
    assert dataloader.pin_memory is True
    assert isinstance(dataloader.sampler, RandomSampler)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_dataloader_invalid_dataset_type():
    with pytest.raises(ValueError):
        DataLoaderFactory.create_dataloader(dataset="invalid_dataset_type")


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_dataloader_with_sampler_and_shuffle(mock_dataset):
    config = DataLoaderConfig(
        dataset=mock_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        sampler=Mock(),
    )
    dataloader = DataLoaderFactory.create_dataloader(config)
    assert isinstance(dataloader, TorchDataLoader)
    assert dataloader.dataset == mock_dataset
    assert dataloader.batch_size == 32
    assert dataloader.num_workers == 4
    # shuffle should be False when sampler is provided
    assert config.shuffle is False
    assert dataloader.drop_last is True
    assert dataloader.pin_memory is True
    assert isinstance(dataloader.sampler, Mock)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_create_dataloader_with_sampler_and_no_shuffle(mock_dataset):
    config = DataLoaderConfig(
        dataset=mock_dataset,
        batch_size=32,
        num_workers=4,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
        sampler=Mock(),
    )
    dataloader = DataLoaderFactory.create_dataloader(config)
    assert isinstance(dataloader, TorchDataLoader)
    assert dataloader.dataset == mock_dataset
    assert dataloader.batch_size == 32
    assert dataloader.num_workers == 4
    # shuffle should be False when sampler is provided
    assert config.shuffle is False
    assert dataloader.drop_last is True
    assert dataloader.pin_memory is True
    assert isinstance(dataloader.sampler, Mock)


def test_od_collate_fn_basic():
    # Simulate a batch of two images with two objects each
    img1 = torch.zeros((3, 32, 32))
    img2 = torch.ones((3, 32, 32))
    annots1 = [
        {"bbox": [0, 0, 10, 10], "category_id": 1},
        {"bbox": [5, 5, 10, 10], "category_id": 2},
    ]
    annots2 = [{"bbox": [1, 1, 5, 5], "category_id": 3}]
    batch = [(img1, annots1), (img2, annots2)]
    images, targets = od_collate_fn(batch)
    assert images.shape == (2, 3, 32, 32)
    assert "boxes" in targets and "labels" in targets and "scores" in targets
    assert len(targets["boxes"]) == 2
    assert len(targets["labels"]) == 2
    assert len(targets["scores"]) == 2
    # Check that boxes and labels are tensors
    assert all(isinstance(b, torch.Tensor) for b in targets["boxes"])
    assert all(isinstance(l, torch.Tensor) for l in targets["labels"])
    assert all(isinstance(s, torch.Tensor) for s in targets["scores"])


def test_od_collate_fn_empty_annots():
    img = torch.zeros((3, 32, 32))
    batch = [(img, [])]
    images, targets = od_collate_fn(batch)
    assert images.shape == (1, 3, 32, 32)
    assert targets["boxes"][0].shape == (0, 4)
    assert targets["labels"][0].shape == (0,)
    assert targets["scores"][0].shape == (0,)


def test_od_collate_fn_raises_on_missing_bbox_or_category_id():
    img = torch.zeros((3, 32, 32))
    # Missing 'bbox'
    batch_missing_bbox = [(img, [{"category_id": 1}])]
    with pytest.raises(ValueError, match="Malformed annotation object"):
        od_collate_fn(batch_missing_bbox)
    # Missing 'category_id'
    batch_missing_category = [(img, [{"bbox": [0, 0, 10, 10]}])]
    with pytest.raises(ValueError, match="Malformed annotation object"):
        od_collate_fn(batch_missing_category)


def test_create_od_dataloader(mock_dataset):
    config = DataLoaderConfig(
        dataset=mock_dataset,
        batch_size=2,
        num_workers=0,
        shuffle=True,
        drop_last=False,
        pin_memory=False,
        sampler=None,
    )
    dataloader = DataLoaderFactory.create_od_dataloader(config)
    assert hasattr(dataloader, "collate_fn") or hasattr(dataloader, "_collate_fn")
    # The collate_fn should be od_collate_fn or a wrapper
    # We can't check for exact function equality due to DataLoader internals, but we can check it runs
    batch = [(torch.zeros((3, 32, 32)), [])]
    images, targets = dataloader.collate_fn(batch)
    assert images.shape == (1, 3, 32, 32)
    assert "boxes" in targets
