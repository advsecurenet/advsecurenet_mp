import pytest
import numpy as np
import torch
from unittest.mock import MagicMock

from advsecurenet.computer_vision.object_detection.attacks.pixel_perturbation_based.tog import (
    TOG,
    TOGAttackType,
)
from advsecurenet.shared.types.configs.attack_configs.tog_attack_config import (
    TOGAttackConfig,
)


# Dummy object detector for TOG
class DummyObjectDetector:
    def __init__(self):
        self.model = MagicMock()
        self.model.parameters = lambda: iter([torch.zeros(1)])

    def compute_object_vanishing_gradient(self, x_adv, training=False):
        return np.ones_like(x_adv)

    def compute_object_fabrication_gradient(self, x_adv):
        return np.ones_like(x_adv)

    def compute_object_mislabeling_gradient(self, x_adv, detections=None, mode=None):
        return np.ones_like(x_adv)

    def compute_object_untargeted_gradient(self, x_adv, detections=None):
        return np.ones_like(x_adv)

    def predict(self, x):
        # Return dummy detections
        batch_size = x.shape[0] if hasattr(x, "shape") else len(x)
        return [
            {
                "boxes": np.array([[0, 0, 1, 1]]),
                "labels": np.array([1]),
                "scores": np.array([1.0]),
                "logits": np.ones((1, 3)),
            }
            for _ in range(batch_size)
        ]


@pytest.fixture
def tog_config():
    return TOGAttackConfig(
        object_detector=DummyObjectDetector(),  # type: ignore
        max_iter=2,
        eps=0.03,
        eps_iter=0.01,
        attack_type="mislabeling",
        mislabeling_mode="ml",
        device=MagicMock(processor="cpu", use_ddp=False),
        targeted=False,
    )


def make_dummy_images(batch=2, c=3, h=10, w=10):
    return np.zeros((batch, c, h, w), dtype=np.float32)


def make_dummy_detections(batch=2, num_classes=3):
    return [
        {
            "boxes": np.array([[0, 0, 1, 1]]),
            "labels": np.array([1]),
            "scores": np.array([1.0]),
            "logits": np.ones((1, num_classes)),
        }
        for _ in range(batch)
    ]


def test_tog_instantiation(tog_config):
    tog = TOG(tog_config)
    assert isinstance(tog, TOG)


def test_generate_attack_targets_ml_ll():
    detections = make_dummy_detections()
    arr_ml = TOG.generate_attack_targets(detections, mode="ml")
    arr_ll = TOG.generate_attack_targets(detections, mode="ll")
    assert arr_ml.shape[1] == 7
    assert arr_ll.shape[1] == 7
    # With class_id
    arr_class = TOG.generate_attack_targets(detections, mode="ml", class_id=1)
    assert arr_class.shape[1] == 7


def test_tog_attack_vanishing(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    out = tog.attack(x, mask=None, tog_variant=TOGAttackType.VANISHING)
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape


def test_tog_attack_fabrication(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    out = tog.attack(x, mask=None, tog_variant=TOGAttackType.FABRICATION)
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape


def test_tog_attack_mislabeling_ml(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    out = tog.attack(
        x, mask=None, tog_variant=TOGAttackType.MISLABELING, tog_mislabeling_mode="ml"
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape


def test_tog_attack_mislabeling_ll(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    out = tog.attack(
        x, mask=None, tog_variant=TOGAttackType.MISLABELING, tog_mislabeling_mode="ll"
    )
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape


def test_tog_attack_untargeted(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    out = tog.attack(x, mask=None, tog_variant=TOGAttackType.UNTARGETED)
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape


def test_tog_vanishing(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    out = tog.tog_vanishing(x)
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape


def test_tog_fabrication(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    out = tog.tog_fabrication(x)
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape


def test_tog_mislabeling_ml(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    out = tog.tog_mislabeling(x, mode="ml")
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape


def test_tog_mislabeling_ll(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    out = tog.tog_mislabeling(x, mode="ll")
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape


def test_tog_untargeted(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    out = tog.tog_untargeted(x)
    assert isinstance(out, np.ndarray)
    assert out.shape == x.shape


def test_generate_attack_targets_empty():
    arr = TOG.generate_attack_targets([], mode="ml")
    assert arr.shape == (0, 7)


def test_generate_attack_targets_no_boxes():
    detections = [
        {
            "boxes": np.zeros((0, 4)),
            "labels": np.zeros((0,)),
            "scores": np.zeros((0,)),
            "logits": np.ones((0, 3)),
        }
    ]
    arr = TOG.generate_attack_targets(detections, mode="ml")
    assert arr.shape == (0, 7)


def test_generate_attack_targets_out_of_bounds_label():
    # This test expects ValueError due to shape mismatch when skipping out-of-bounds label
    detections = [
        {
            "boxes": np.array([[0, 0, 1, 1]]),
            "labels": np.array([99]),
            "scores": np.array([1.0]),
            "logits": np.ones((1, 3)),
        }
    ]
    with pytest.raises(ValueError):
        TOG.generate_attack_targets(detections, mode="ml")


def test_generate_attack_targets_background_class():
    detections = [
        {
            "boxes": np.array([[0, 0, 1, 1]]),
            "labels": np.array([0]),
            "scores": np.array([1.0]),
            "logits": np.ones((1, 11)),
        }
    ]
    arr = TOG.generate_attack_targets(detections, mode="ll")
    assert arr.shape == (1, 7)


def test_generate_attack_targets_class_id_not_present():
    detections = make_dummy_detections()
    arr = TOG.generate_attack_targets(detections, mode="ml", class_id=99)
    assert arr.shape == (0, 7)


def test_generate_attack_targets_class_id_present():
    detections = make_dummy_detections()
    arr = TOG.generate_attack_targets(detections, mode="ml", class_id=1)
    assert arr.shape[1] == 7


def test_generate_attack_targets_logits_shape():
    detections = [
        {
            "boxes": np.array([[0, 0, 1, 1]]),
            "labels": np.array([1]),
            "scores": np.array([1.0]),
            "logits": np.ones((1, 5)),
        }
    ]
    arr = TOG.generate_attack_targets(detections, mode="ml")
    assert arr.shape == (1, 7)


# --- Attack input variations ---
def test_tog_attack_x_range_0_255(tog_config):
    tog = TOG(tog_config)
    x = (make_dummy_images() * 255).astype(np.float32)
    out = tog.attack(x, mask=None, tog_variant=TOGAttackType.VANISHING)
    assert out.max() <= 1.0
    assert out.min() >= 0.0


def test_tog_attack_x_range_0_1(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    out = tog.attack(x, mask=None, tog_variant=TOGAttackType.VANISHING)
    assert out.max() <= 1.0
    assert out.min() >= 0.0


def test_tog_attack_batch_size_1(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images(batch=1)
    out = tog.attack(x, mask=None, tog_variant=TOGAttackType.FABRICATION)
    assert out.shape[0] == 1


def test_tog_attack_batch_size_3(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images(batch=3)
    out = tog.attack(x, mask=None, tog_variant=TOGAttackType.FABRICATION)
    assert out.shape[0] == 3


def test_tog_attack_noncontiguous_input(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()[..., ::-1].copy()
    out = tog.attack(x, mask=None, tog_variant=TOGAttackType.FABRICATION)
    assert out.shape == x.shape


def test_tog_attack_with_mask_ignored(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    mask = np.ones_like(x)
    out = tog.attack(x, mask=mask, tog_variant=TOGAttackType.FABRICATION)
    assert out.shape == x.shape


def test_tog_attack_unknown_variant(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()

    class FakeVariant:
        pass

    result = tog.attack(x, mask=None, tog_variant=FakeVariant())  # type: ignore
    assert result is None


# --- Mislabeling mode and gradient edge cases ---
def test_tog_mislabeling_unknown_mode(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    # Should fallback to 'ml' and not raise
    out = tog.tog_mislabeling(x, mode="unknown")
    assert out.shape == x.shape


def test_tog_mislabeling_small_gradient_breaks_early(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    # Patch object_detector to return very small gradients
    tog.object_detector.compute_object_mislabeling_gradient = lambda *a, **kw: np.zeros_like(x)  # type: ignore
    out = tog.tog_mislabeling(x, mode="ml")
    assert out.shape == x.shape


def test_tog_mislabeling_empty_initial_detections(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    # Patch object_detector to return empty detections
    tog.object_detector.predict = lambda x: [{"boxes": np.zeros((0, 4)), "labels": np.zeros((0,)), "scores": np.zeros((0,)), "logits": np.ones((0, 3))} for _ in range(x.shape[0])]  # type: ignore
    out = tog.tog_mislabeling(x, mode="ml")
    assert out.shape == x.shape


# --- Error handling ---
def test_tog_object_detector_raises(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    tog.object_detector.compute_object_vanishing_gradient = MagicMock(side_effect=RuntimeError("fail"))  # type: ignore
    with pytest.raises(RuntimeError):
        tog.tog_vanishing(x)
    tog.object_detector.compute_object_fabrication_gradient = MagicMock(side_effect=RuntimeError("fail"))  # type: ignore
    with pytest.raises(RuntimeError):
        tog.tog_fabrication(x)
    tog.object_detector.compute_object_mislabeling_gradient = MagicMock(side_effect=RuntimeError("fail"))  # type: ignore
    with pytest.raises(RuntimeError):
        tog.tog_mislabeling(x, mode="ml")
    tog.object_detector.compute_object_untargeted_gradient = MagicMock(side_effect=RuntimeError("fail"))  # type: ignore
    with pytest.raises(RuntimeError):
        tog.tog_untargeted(x)


def test_tog_missing_keys_in_detection(tog_config):
    tog = TOG(tog_config)
    x = make_dummy_images()
    # Remove 'logits' key
    tog.object_detector.predict = lambda x: [{"boxes": np.array([[0, 0, 1, 1]]), "labels": np.array([1]), "scores": np.array([1.0])} for _ in range(x.shape[0])]  # type: ignore
    # Should not raise, just skip or return zeros
    out = tog.tog_mislabeling(x, mode="ml")
    assert out.shape == x.shape
