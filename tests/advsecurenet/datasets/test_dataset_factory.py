import pytest
from advsecurenet.datasets.dataset_factory import DatasetFactory
from advsecurenet.datasets.HuggingFace.huggingface_dataset import HuggingFaceDataset
from advsecurenet.datasets.base_dataset import DatasetWrapper
from cli.shared.types.utils.dataset import ResolvedDatasetConfig, ResolvedSplitConfig
from torchvision.datasets.cifar import CIFAR10
import warnings
from unittest.mock import patch, MagicMock

from datasets import load_dataset as hf_hub_load_dataset

from advsecurenet.datasets.dataset_factory import (
    _create_provider,
    _infner_dataset_type,
    _infer_dataset_class_from_type,
    _merge_dicts_with_warning,
    _prepare_load_kwargs,
    DATASET_MAP,
    available_datasets
)

from advsecurenet.shared.types.dataset import DatasetType
from advsecurenet.datasets.Cifar10.cifar10_dataset import CIFAR10Dataset

# Fixture 1: Downloads the TRAIN split once per session.
@pytest.fixture(scope="session")
def real_cifar10_hf_train_split():
    """
    Downloads the real CIFAR-10 TRAIN split from Hugging Face once and
    caches it for the entire test session.
    """
    print("\n(Downloading real Hugging Face TRAIN split for testing...)")
    return hf_hub_load_dataset("uoft-cs/cifar10", split="train")

# Fixture 2: Downloads the TEST split once per session.
@pytest.fixture(scope="session")
def real_cifar10_hf_test_split():
    """
    Downloads the real CIFAR-10 TEST split from Hugging Face once and
    caches it for the entire test session.
    """
    print("\n(Downloading real Hugging Face TEST split for testing...)")
    return hf_hub_load_dataset("uoft-cs/cifar10", split="test")

@pytest.fixture
def mock_hf_hub_load(real_cifar10_hf_train_split, real_cifar10_hf_test_split):
    """
    A fixture that patches hf_hub_load_dataset and yields a mock that
    returns pre-loaded train/test splits.
    """
    def mock_loader(*args, **kwargs):
        path = kwargs.get("path")
        split_name = kwargs.get("split")
        if path == "uoft-cs/cifar10":
            if split_name == "train": return real_cifar10_hf_train_split
            if split_name == "test": return real_cifar10_hf_test_split
        raise ValueError(f"Mock received unexpected call with path='{path}' and split='{split_name}'")

    with patch("advsecurenet.datasets.HuggingFace.huggingface_dataset.hf_hub_load_dataset", side_effect=mock_loader, autospec=True) as mock:
        yield mock


# It requests both fixtures to ensure they both download correctly.
@pytest.mark.advsecurenet
@pytest.mark.integration
@pytest.mark.essential
def test_huggingface_real_download_and_load(real_cifar10_hf_train_split, real_cifar10_hf_test_split):
    """
    Verifies that the real train/test splits can be downloaded and are not empty.
    This is the primary integration test.
    """
    assert len(real_cifar10_hf_train_split) > 0
    assert len(real_cifar10_hf_test_split) > 0

    assert real_cifar10_hf_train_split is not None
    assert real_cifar10_hf_test_split is not None
    assert len(real_cifar10_hf_train_split) > 0
    assert len(real_cifar10_hf_test_split) > 0

# This is the expected, normalized config object that the factory will receive.
# We will create it manually for the first test.
RESOLVED_CONFIG_FOR_TEST = ResolvedDatasetConfig(
    dataset_name='Test_cifar_10_huggingface',
    splits={
        'train': ResolvedSplitConfig(
            identifier='https://huggingface.co/datasets/uoft-cs/cifar10',
            num_classes=10,
            source_split_name='train',
            constructor_args={'input_key': 'img', 'target_key': 'label'}
        ),
        'test': ResolvedSplitConfig(
            identifier='https://huggingface.co/datasets/uoft-cs/cifar10',
            num_classes=10,
            source_split_name='test',
            constructor_args={'input_key': 'img', 'target_key': 'label'}
        )
    }
)

# These are the raw config dictionaries, simulating user input for the second test.
config_variant_1 = {
    "dataset_name": "Test_cifar_10_huggingface",
    "num_classes": 10,
    "split_config": {
        "train": {
            "identifier": "https://huggingface.co/datasets/uoft-cs/cifar10",
            "split_name": "train",
            "constructor_args": {"input_key": "img", "target_key": "label"},
        },
        "test": {
            "identifier": "https://huggingface.co/datasets/uoft-cs/cifar10",
            "split_name": "test",
            "constructor_args": {"input_key": "img", "target_key": "label"},
        },
    },
}

config_variant_2 = {
    "dataset_name": "https://huggingface.co/datasets/uoft-cs/cifar10",
    "num_classes": 10,
    "constructor_args": {"input_key": "img", "target_key": "label"},
}

config_variant_3 = {
    "dataset_name": "https://huggingface.co/datasets/uoft-cs/cifar10",
    "num_classes": 10,
    "constructor_args": {"input_key": "img", "target_key": "label"},
    "load_splits": ["train", "test"],
}

cifar10_config = {
    "dataset_name": "cifar10",
    "num_classes": 10,
}


@pytest.mark.advsecurenet
@pytest.mark.integration
@pytest.mark.essential
def test_load_dataset_from_resolved_config(mock_hf_hub_load):
    """
    Tests the factory's `load_dataset_from_config` method directly with a
    manually created, pre-resolved configuration object.
    """

    loaded_datasets = DatasetFactory.load_dataset_from_config(RESOLVED_CONFIG_FOR_TEST)

    # Assert: Check that the datasets were loaded and configured correctly
    assert "train" in loaded_datasets
    assert "test" in loaded_datasets
    train_wrapper = loaded_datasets["train"]
    assert isinstance(train_wrapper, DatasetWrapper)
    assert isinstance(train_wrapper.dataset, HuggingFaceDataset)
    assert train_wrapper.dataset._input_key == "img"
    assert train_wrapper.dataset._target_key == "label"
    assert len(train_wrapper.dataset) > 0

    assert mock_hf_hub_load.call_count == 2
    mock_hf_hub_load.assert_any_call(path='uoft-cs/cifar10', split='train')
    mock_hf_hub_load.assert_any_call(path='uoft-cs/cifar10', split='test')


@pytest.mark.advsecurenet
@pytest.mark.integration
@pytest.mark.essential
@pytest.mark.parametrize(
    "config_dict",
    [config_variant_1, config_variant_2, config_variant_3],
    ids=["detailed_split_config", "global_url", "global_url_explicit_splits"],
)
def test_load_dataset_with_kwargs(config_dict, mock_hf_hub_load):
    """
    Tests the factory's `load_dataset` method, passing the raw user config
    as keyword arguments. This tests the internal config resolution and loading pipeline.
    """
    # Act: Call the factory with the raw config dictionary as kwargs
    loaded_datasets = DatasetFactory.load_dataset(**config_dict)

    # Assert: Check that the datasets were loaded and configured correctly
    assert "train" in loaded_datasets
    assert "test" in loaded_datasets
    train_wrapper = loaded_datasets["train"]
    assert isinstance(train_wrapper, DatasetWrapper)
    assert isinstance(train_wrapper.dataset, HuggingFaceDataset)
    assert train_wrapper.dataset._input_key == "img"
    assert train_wrapper.dataset._target_key == "label"
    assert len(train_wrapper.dataset) > 0

    assert mock_hf_hub_load.call_count == 2
    mock_hf_hub_load.assert_any_call(path='uoft-cs/cifar10', split='train')
    mock_hf_hub_load.assert_any_call(path='uoft-cs/cifar10', split='test')

@pytest.mark.advsecurenet
@pytest.mark.integration
@pytest.mark.essential
def test_load_dataset_cifar10_with_kwargs():
    """
    Tests the factory's `load_dataset` method for a standard torchvision
    dataset (CIFAR10) using keyword arguments.
    """
    # Setup: Use a temporary directory for the dataset download
    config = cifar10_config.copy()

    # Act: Call the factory with the raw config dictionary as kwargs
    loaded_datasets = DatasetFactory.load_dataset(**config)

    # Assert: Check that the datasets were loaded and configured correctly
    assert "train" in loaded_datasets
    assert "test" in loaded_datasets
    train_wrapper = loaded_datasets["train"]
    assert isinstance(train_wrapper, DatasetWrapper)

    # Assert that the correct dataset type was instantiated
    assert isinstance(train_wrapper.dataset, CIFAR10)

    # Verify that the dataset is not empty and we can get an item
    assert len(train_wrapper.dataset) > 0
    input_data, target_data = train_wrapper.dataset[0]
    assert input_data is not None
    assert target_data is not None

@pytest.mark.advsecurenet
@pytest.mark.unit
def test_available_datasets():
    """
    Tests that available_datasets returns a non-empty list of DatasetType members.
    """
    available = available_datasets()
    assert isinstance(available, list)
    assert len(available) > 0
    assert all(isinstance(item, DatasetType) for item in available)
    assert DatasetType.CIFAR10 in available

@pytest.mark.advsecurenet
@pytest.mark.unit
def test_merge_dicts_with_warning():
    """
    Tests merging dictionaries, including the warning mechanism for overrides.
    """
    base = {'a': 1, 'b': 2}
    override = {'b': 3, 'c': 4}
    
    # Test merging with an override, expecting a warning
    with pytest.warns(UserWarning, match="Key 'b' was overridden. Old: 2, New: 3"):
        merged = _merge_dicts_with_warning(
            base, override, "Key '{key}' was overridden. Old: {old_value}, New: {new_value}"
        )
    
    assert merged == {'a': 1, 'b': 3, 'c': 4}

    # Test merging with no overlap, expecting no warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        merged_no_conflict = _merge_dicts_with_warning(base, {'c': 4}, "No conflict")
        assert len(w) == 0
        assert merged_no_conflict == {'a': 1, 'b': 2, 'c': 4}

@pytest.mark.advsecurenet
@pytest.mark.unit
def test_prepare_load_kwargs():
    """
    Tests the preparation of keyword arguments for the load_dataset method.
    """
    base_kwargs = {'trust_remote_code': False}
    keys_to_override = {'path': 'some/path', 'split': 'train'}
    runtime_kwargs = {'trust_remote_code': True, 'use_auth_token': 'my_token'}

    with pytest.warns(UserWarning, match="Runtime argument 'trust_remote_code' is overriding"):
        final_kwargs = _prepare_load_kwargs(
            base_kwargs=base_kwargs,
            keys_to_override=keys_to_override,
            runtime_kwargs=runtime_kwargs,
            logical_name='train'
        )
    
    expected = {
        'path': 'some/path',
        'split': 'train',
        'trust_remote_code': True,
        'use_auth_token': 'my_token'
    }
    assert final_kwargs == expected

@pytest.mark.advsecurenet
@pytest.mark.unit
def test_infer_dataset_class_from_type():
    """
    Tests that the correct dataset class is returned for a given DatasetType.
    """
    assert _infer_dataset_class_from_type(DatasetType.CIFAR10) is CIFAR10Dataset
    assert _infer_dataset_class_from_type(DatasetType.HUGGINGFACE) is HuggingFaceDataset

    # Negative case: Test with an invalid type
    with pytest.raises(ValueError, match="Unknown dataset type"):
        _infer_dataset_class_from_type("INVALID_TYPE")


@pytest.mark.advsecurenet
@pytest.mark.unit
@patch('advsecurenet.datasets.dataset_factory.huggingface_dataset_utils.verify_hf_dataset_identifier_exists')
def test_infner_dataset_type(mock_verify_hf):
    """
    Tests the inference of dataset type from an identifier string.
    """
    # Positive case: Hugging Face dataset
    mock_verify_hf.return_value = True
    assert _infner_dataset_type('user/repo') == DatasetType.HUGGINGFACE

    # Positive case: Standard dataset
    mock_verify_hf.return_value = False
    assert _infner_dataset_type('cifar10') == DatasetType.CIFAR10

    # Negative case: Unknown identifier
    mock_verify_hf.return_value = False
    with pytest.raises(ValueError, match="Unknown dataset identifier: unknown_dataset"):
        _infner_dataset_type('unknown_dataset')

@pytest.mark.advsecurenet
@pytest.mark.unit
def test_create_provider():
    """
    Tests the creation of a dataset provider instance.
    """
    # Realistic example for a Hugging Face dataset
    hf_split_config = ResolvedSplitConfig(
        identifier='user/repo',
        num_classes=10,
        constructor_args={
            'input_key': 'pixel_values',
            'target_key': 'label',
            'num_classes': 10 # This should be passed to the constructor
        }
    )
    
    provider = _create_provider(hf_split_config, DatasetType.HUGGINGFACE)
    assert isinstance(provider, HuggingFaceDataset)
    assert provider.num_classes == 10
    assert provider._input_key == 'pixel_values'

    # Example for a standard dataset
    cifar_split_config = ResolvedSplitConfig(
        identifier='cifar10',
        num_classes=10,
        constructor_args={'num_classes': 10}
    )
    provider = _create_provider(cifar_split_config, DatasetType.CIFAR10)
    assert isinstance(provider, CIFAR10Dataset)
    assert provider.num_classes == 10

    # Negative case: Test that extra kwargs are filtered out
    # CIFAR10Dataset does not accept 'some_invalid_arg'
    cifar_split_config_extra = ResolvedSplitConfig(
        identifier='cifar10',
        num_classes=10,
        constructor_args={'num_classes': 10, 'some_invalid_arg': True}
    )
    # This should not raise an error because filter_kwargs_for_callable will remove it
    try:
        _create_provider(cifar_split_config_extra, DatasetType.CIFAR10)
    except TypeError as e:
        pytest.fail(f"_create_provider raised an unexpected TypeError: {e}")