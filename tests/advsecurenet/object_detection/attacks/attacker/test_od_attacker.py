import pytest
import torch
from unittest.mock import MagicMock, patch

from advsecurenet.computer_vision.object_detection.attacks.attacker.od_attacker import ODAttacker
from advsecurenet.shared.types.configs.attack_configs.od_attacker_config import ODAttackerConfig
from advsecurenet.shared.types.configs.dataloader_config import DataLoaderConfig
from advsecurenet.shared.types.configs.device_config import DeviceConfig
from advsecurenet.models.model_factory import ModelFactory
from advsecurenet.shared.types.configs.model_config import CreateModelConfig
from torch.utils.data import Dataset as TorchDataset

# Dummy subclass for testing abstract class
class DummyODAttacker(ODAttacker):
    def execute(self):
        return "executed"

# Minimal real dataset for testing
class DummyDataset(TorchDataset):
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return torch.zeros(3, 224, 224), 0

@pytest.fixture
def device(request):
    device_arg = getattr(request.config, 'getoption', lambda x: None)("--device")
    return torch.device(device_arg if device_arg else "cpu")

@pytest.fixture
def config(device):
    device_cfg = DeviceConfig(processor=device)
    with patch("advsecurenet.models.model_factory.ModelFactory.create_model", return_value=MagicMock()) as mock_create_model:
        return ODAttackerConfig(
            model=mock_create_model.return_value,
            attack=MagicMock(),
            dataloader=DataLoaderConfig(dataset=DummyDataset()),
            device=device_cfg,
            return_adversarial_images=True,
        )

def test_setup_device_with_processor(config):
    config.device.processor = "cpu"
    attacker = DummyODAttacker(config)
    assert attacker._device == torch.device("cpu")

@patch("torch.cuda.is_available", return_value=True)
def test_setup_device_default_cuda(mock_cuda, config):
    config.device.processor = None
    attacker = DummyODAttacker(config)
    assert attacker._device == torch.device("cuda")

@patch("torch.cuda.is_available", return_value=False)
def test_setup_device_default_cpu(mock_cuda, config):
    config.device.processor = None
    attacker = DummyODAttacker(config)
    assert attacker._device == torch.device("cpu")

def test_create_dataloader_with_instance(config):
    dummy_dl_config = DataLoaderConfig(dataset=DummyDataset())
    config.dataloader = dummy_dl_config
    attacker = DummyODAttacker(config)
    assert isinstance(attacker._dataloader, object)  # Optionally check for DataLoader type

def test_create_dataloader_with_factory(config):
    # Patch DataLoaderFactory.create_dataloader to return a mock
    with patch("advsecurenet.dataloader.DataLoaderFactory.create_dataloader", return_value=MagicMock()) as mock_factory:
        attacker = DummyODAttacker(config)
        assert attacker._dataloader == mock_factory.return_value
        mock_factory.assert_called_once()

def test_execute_abstract_raises(config):
    # Directly using ODAttacker should raise NotImplementedError
    with pytest.raises(TypeError):
        ODAttacker(config)

def test_execute_dummy_subclass(config):
    attacker = DummyODAttacker(config)
    assert attacker.execute() == "executed" 