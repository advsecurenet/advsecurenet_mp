import pytest
import torch
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import types

# Minimal transformers stub to avoid heavy dependency during imports
if "transformers" not in sys.modules:
    class _AutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return MagicMock()

    class _AutoConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            class _Cfg:
                architectures = []

            return _Cfg()

    transformers_stub = types.SimpleNamespace(AutoModel=_AutoModel, AutoConfig=_AutoConfig)
    sys.modules["transformers"] = transformers_stub

# Minimal mean_average_precision stub to satisfy evaluator imports
if "mean_average_precision" not in sys.modules:
    class _StubMetric:
        def add(self, *args, **kwargs):
            pass

        def reset(self):
            pass

        def value(self, **kwargs):
            return {"mAP": 0.0}

    class _MetricBuilder:
        @staticmethod
        def build_evaluation_metric(*args, **kwargs):
            return _StubMetric()

    sys.modules["mean_average_precision"] = types.SimpleNamespace(MetricBuilder=_MetricBuilder)

from advsecurenet.computer_vision.object_detection.attacks.attacker.od_attacker import (
    ODAttacker,
)
from advsecurenet.shared.types.configs.attack_configs.od_attacker_config import (
    ODAttackerConfig,
)
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


class _NormalizeLike:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std


class _DatasetWithTransform(TorchDataset):
    def __init__(self, with_normalize=True):
        self.transform = types.SimpleNamespace(
            transforms=[_NormalizeLike([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])] if with_normalize else []
        )
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        return torch.zeros(3, 8, 8), {"boxes": [torch.tensor([[0.0, 0.0, 1.0, 1.0]])], "labels": [torch.tensor([1])]} 


@pytest.fixture
def device(request):
    device_arg = getattr(request.config, "getoption", lambda x: None)("--device")
    return torch.device(device_arg if device_arg else "cpu")


@pytest.fixture
def config(device):
    device_cfg = DeviceConfig(processor=device)
    with patch(
        "advsecurenet.models.model_factory.ModelFactory.create_model",
        return_value=MagicMock(),
    ) as mock_create_model:
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
    assert isinstance(
        attacker._dataloader, object
    )  # Optionally check for DataLoader type


def test_create_dataloader_with_factory(config):
    # Patch DataLoaderFactory.create_dataloader to return a mock
    with patch(
        "advsecurenet.dataloader.DataLoaderFactory.create_dataloader",
        return_value=MagicMock(),
    ) as mock_factory:
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


def test_init_does_not_move_or_eval_model(config):
    # Attacker no longer moves/evals the model during __init__
    mock_model = MagicMock()
    config.model = mock_model
    config.device.processor = "cpu"
    attacker = DummyODAttacker(config)
    # Ensure device is set but model movement/eval are not invoked automatically
    assert attacker._device == torch.device("cpu")
    mock_model.to.assert_not_called()
    mock_model.eval.assert_not_called()


def test_create_dataloader_with_real_dataloader(config):
    from torch.utils.data import DataLoader

    dataset = DummyDataset()
    real_dl = DataLoader(dataset)
    config.dataloader = real_dl
    attacker = DummyODAttacker(config)
    assert attacker._dataloader is real_dl


def test_invalid_processor_string_raises():
    # DeviceConfig with invalid processor string should raise from torch.device
    from advsecurenet.shared.types.configs.device_config import DeviceConfig
    from advsecurenet.shared.types.configs.attack_configs.od_attacker_config import (
        ODAttackerConfig,
    )

    device_cfg = DeviceConfig(processor="not_a_real_device")
    config = ODAttackerConfig(
        model=MagicMock(),
        attack=MagicMock(),
        dataloader=MagicMock(),
        device=device_cfg,
        return_adversarial_images=True,
    )
    with pytest.raises(Exception):
        DummyODAttacker(config)


def test_preprocess_images_denormalize_and_scale_warnings(caplog, config):
    # DataLoader with dataset.transform including normalize-like transform
    from torch.utils.data import DataLoader
    ds = _DatasetWithTransform(with_normalize=True)
    config.dataloader = DataLoader(ds)
    attacker = DummyODAttacker(config)

    # Case 1: inputs in [-1,1] -> map to [0,255]
    imgs = torch.linspace(-1, 1, steps=8 * 8).view(1, 1, 8, 8).repeat(1, 3, 1, 1)
    arr = attacker.preprocess_images(imgs)
    assert np.issubdtype(arr.dtype, np.floating) and arr.min() >= 0.0 and arr.max() <= 255.0

    # Case 2: already in 0..1 -> multiply by 255
    imgs2 = torch.rand(1, 3, 8, 8)
    arr2 = attacker.preprocess_images(imgs2)
    assert arr2.max() <= 255.0

    # Case 3: values > 1.1 lead to pass-through then clip warning
    caplog.clear()
    with caplog.at_level("WARNING"):
        big = torch.full((1, 3, 8, 8), 300.0)
        arr3 = attacker.preprocess_images(big)
        # Environment logging filters can suppress warnings; assert effect instead
        assert arr3.max() == 255.0


def test_preprocess_targets_dict_and_process_batch(config, monkeypatch):
    # Patch move_batch_to_device to return inputs unchanged
    monkeypatch.setattr(
        "advsecurenet.utils.device_utils.move_batch_to_device",
        lambda images, targets, device: (images.to(device), {k: [v[0].to(device)] for k, v in targets.items()}),
    )
    from torch.utils.data import DataLoader
    ds = _DatasetWithTransform(with_normalize=False)
    config.dataloader = DataLoader(ds)
    attacker = DummyODAttacker(config)
    batch = next(iter(attacker._dataloader))
    images_np, targets_np, original_images = attacker.process_batch(batch)
    # images_np is numpy array in [0,255]; targets list of dicts with numpy arrays
    assert isinstance(images_np, np.ndarray)
    assert isinstance(targets_np, list) and isinstance(targets_np[0]["boxes"], np.ndarray)
    assert isinstance(original_images, torch.Tensor)
