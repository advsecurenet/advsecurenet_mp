import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
from torch.utils.data import Dataset as TorchDataset

from advsecurenet.computer_vision.object_detection.attacks.adversarial_patch_based.dpatch import DPatch
from advsecurenet.shared.types.configs.attack_configs.dpatch_attack_config import DPatchAttackConfig

# Minimal dummy object detector for DPatch
def make_dummy_detector():
    class DummyObjectDetector:
        def __init__(self):
            self.model = MagicMock()
            self.model.eval = MagicMock()
        def predict(self, x):
            return [{"boxes": np.array([[0,0,1,1]]), "labels": np.array([1]), "scores": np.array([1.0])} for _ in range(len(x))]
        def filter_boxes(self, t, threshold):
            return t
        def loss_gradient(self, x, y, standardise_output=True):
            return np.ones((len(x), 3, 10, 10), dtype=np.float32)
    return DummyObjectDetector()

# Dummy dataset for DPatch
class DummyDataset(TorchDataset):
    def __len__(self):
        return 2
    def __getitem__(self, idx):
        image = torch.zeros((3, 10, 10), dtype=torch.float32)
        targets_dict = {
            "boxes": torch.zeros((1, 4), dtype=torch.float32),
            "labels": torch.zeros((1,), dtype=torch.int64),
        }
        return image, targets_dict

@pytest.fixture
def dpatch_config():
    return DPatchAttackConfig(
        object_detector=make_dummy_detector(),
        patch_shape=(3, 10, 10),
        learning_rate=1.0,
        max_iter=1,
        target_label=1,
        device=MagicMock(processor="cpu", use_ddp=False),
        targeted=True,
    )

def test_dpatch_instantiation(dpatch_config):
    dpatch = DPatch(dpatch_config)
    assert isinstance(dpatch, DPatch)

def test_dpatch_attack_runs(dpatch_config):
    dpatch = DPatch(dpatch_config)
    dataloader = torch.utils.data.DataLoader(DummyDataset(), batch_size=2)
    mask = None  # Fix: avoid transforms/mask conflict
    device = torch.device("cpu")
    patch = dpatch.attack(dataloader, mask, device)
    assert isinstance(patch, torch.Tensor)
    assert patch.shape == torch.Size([3, 10, 10])

def test_dpatch_apply_patch(dpatch_config):
    dpatch = DPatch(dpatch_config)
    x = np.zeros((2, 3, 10, 10), dtype=np.float32)  # Pass as numpy array
    patch = np.ones((3, 10, 10), dtype=np.float32)
    patched = dpatch.apply_patch(x, patch_external=patch)
    assert isinstance(patched, (np.ndarray, torch.Tensor))
    assert patched.shape == (2, 3, 10, 10)

def test_dpatch_attack_step(dpatch_config):
    dpatch = DPatch(dpatch_config)
    x = np.zeros((2, 3, 10, 10), dtype=np.float32)
    y = [{"boxes": np.array([[0,0,1,1]]), "labels": np.array([1]), "scores": np.array([1.0])}, {"boxes": np.array([[0,0,1,1]]), "labels": np.array([1]), "scores": np.array([1.0])}]
    mask = None  # Fix: avoid transforms/mask conflict
    grad, suppress = dpatch._attack_step(x, y, mask)
    assert isinstance(grad, torch.Tensor)
    assert grad.shape == torch.Size([3, 10, 10])
    assert isinstance(suppress, bool)

def test_dpatch_augment_images_with_patch():
    x = torch.zeros((2, 3, 10, 10))
    patch = torch.ones((3, 5, 5))
    random_location = False
    patched, transforms = DPatch._augment_images_with_patch(x, patch, random_location)
    assert isinstance(patched, torch.Tensor)
    assert patched.shape == (2, 3, 10, 10)
    assert isinstance(transforms, list)
    assert all(isinstance(t, dict) for t in transforms)
