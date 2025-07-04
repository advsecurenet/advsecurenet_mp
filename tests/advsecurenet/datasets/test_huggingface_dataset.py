import pytest
from unittest.mock import MagicMock, patch

import torch

from advsecurenet.datasets.HuggingFace.huggingface_dataset import HuggingFaceDataset
from advsecurenet.shared.types.configs.preprocess_config import PreprocessConfig

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_huggingface_dataset_init_defaults():
    ds = HuggingFaceDataset(num_classes=10)
    assert ds.num_classes == 10
    assert ds.num_input_channels == 3
    assert ds.mean == None
    assert ds.std == None
    assert ds._input_key == "image"
    assert ds._target_key == "label"

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_huggingface_dataset_init_custom():
    ds = HuggingFaceDataset(
        num_classes=5,
        num_input_channels=1,
        mean=[0.1],
        std=[0.2],
        input_key="img",
        target_key="lbl"
    )
    assert ds.num_classes == 5
    assert ds.num_input_channels == 1
    assert ds.mean == [0.1]
    assert ds.std == [0.2]
    assert ds._input_key == "img"
    assert ds._target_key == "lbl"

@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.datasets.HuggingFace.huggingface_dataset.filter_kwargs_for_callable", lambda f, k: k)
@patch("advsecurenet.datasets.HuggingFace.huggingface_dataset.hf_hub_load_dataset")
def test_huggingface_dataset_load_dataset_success(mock_hf_load):
    mock_data = MagicMock()
    ds = HuggingFaceDataset(num_classes=10)
    mock_hf_load.return_value = mock_data
    ds.get_transforms = MagicMock(return_value=None)
    result = ds.load_dataset(path="foo", split="train")
    assert ds._raw_hf_data == mock_data
    assert result.name == ds.name
    assert isinstance(result, type(ds._dataset))

@pytest.mark.advsecurenet
@pytest.mark.essential
@patch("advsecurenet.datasets.HuggingFace.huggingface_dataset.filter_kwargs_for_callable", lambda f, k: k)
@patch("advsecurenet.datasets.HuggingFace.huggingface_dataset.hf_hub_load_dataset", side_effect=Exception("fail"))
def test_huggingface_dataset_load_dataset_error(mock_hf_load):
    ds = HuggingFaceDataset(num_classes=10)
    with pytest.raises(ValueError) as excinfo:
        ds.load_dataset(path="foo", split="train")
    assert "Error loading Hugging Face dataset" in str(excinfo.value)

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_huggingface_dataset_len_and_getitem_success():
    ds = HuggingFaceDataset(num_classes=10)
    ds._raw_hf_data = [
        {"image": torch.zeros(3, 32, 32), "label": 1},
        {"image": torch.ones(3, 32, 32), "label": 2},
    ]
    ds._transforms_to_apply = None
    assert len(ds) == 2
    x, y = ds[0]
    assert torch.equal(x, torch.zeros(3, 32, 32))
    assert y.item() == 1

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_huggingface_dataset_len_not_loaded():
    ds = HuggingFaceDataset(num_classes=10)
    ds._raw_hf_data = None
    with pytest.raises(RuntimeError):
        len(ds)

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_huggingface_dataset_getitem_not_loaded():
    ds = HuggingFaceDataset(num_classes=10)
    ds._raw_hf_data = None
    with pytest.raises(RuntimeError):
        ds[0]

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_huggingface_dataset_getitem_missing_input_key():
    ds = HuggingFaceDataset(num_classes=10)
    ds._raw_hf_data = [{"label": 1}]
    ds._transforms_to_apply = None
    with pytest.raises(KeyError):
        ds[0]

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_huggingface_dataset_getitem_missing_target_key():
    ds = HuggingFaceDataset(num_classes=10)
    ds._raw_hf_data = [{"image": torch.zeros(3, 32, 32)}]
    ds._transforms_to_apply = None
    with pytest.raises(KeyError):
        ds[0]

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_huggingface_dataset_getitem_with_transform():
    ds = HuggingFaceDataset(num_classes=10)
    ds._raw_hf_data = [{"image": torch.zeros(3, 32, 32), "label": 1}]
    mock_transform = MagicMock(return_value="transformed")
    ds._transforms_to_apply = mock_transform
    x, y = ds[0]
    assert x == "transformed"
    assert y.item() == 1

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_huggingface_dataset_process_dataset_kwargs():
    ds = HuggingFaceDataset(num_classes=10)
    kwargs = {"dataset_name": "foo", "download": True, "root": "/tmp", "split": "train"}
    processed = ds.process_dataset_kwargs(kwargs)
    assert "path" in processed
    assert "dataset_name" not in processed
    assert "download" not in processed
    assert "root" not in processed

@pytest.mark.advsecurenet
@pytest.mark.essential
def test_huggingface_dataset_process_kwargs_load_dataset():
    ds = HuggingFaceDataset(num_classes=10)
    kwargs = {"dataset_name": "foo", "download": True, "root": "/tmp", "split": "train"}
    processed = ds.process_kwargs_load_dataset(kwargs)
    assert "path" in processed
    assert "dataset_name" not in processed
    assert "download" not in processed
    assert "root" not in processed

@pytest.mark.advsecurenet
@pytest.mark.essential
@pytest.mark.integration
def test_huggingface_dataset_with_real_data():
    # This will download a small split from HuggingFace (requires internet)
    kwargs = {'split': 'train', 'path': 'uoft-cs/cifar10'}
    ds = HuggingFaceDataset(num_classes=10, input_key="img")
    dataset_wrapper = ds.load_dataset(**kwargs)
    # Check that the wrapper and dataset are not empty
    assert len(ds) > 0
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.long
    # Check that the wrapper's name matches
    assert dataset_wrapper.name == ds.name