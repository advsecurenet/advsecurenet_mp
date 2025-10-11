from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from advsecurenet.computer_vision.object_detection.defenses.adversarial_od_training import (
    AdversarialODTraining,
)
from advsecurenet.computer_vision.base.adversarial_attack import AdversarialAttack
from advsecurenet.models.base_model import BaseModel


class MockModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.model_name = "mock_model"
        self._was_training = True  # mirror of torch.nn.Module.training
        self._training_set_calls = 0

    def forward(self, x, *args, **kwargs):  # type: ignore[override]
        return x

    def train(self, mode: bool = True):  # type: ignore[override]
        # Delegate to torch.nn.Module to set .training and propagate to children
        super().train(mode)
        # Keep our mirrors/counters for assertions
        self._was_training = mode
        self._training_set_calls += 1
        return self

    def to(self, device):  # type: ignore[override]
        return self

    def load_model(self) -> None:  # satisfy abstract method
        return None

    def models(self):  # satisfy abstract requirement
        return [self]


class MockAttackBase(AdversarialAttack):
    __test__ = False

    def __init__(self):
        # Do NOT call super().__init__ (it requires a full AttackConfig).
        # Provide just the attributes the code paths under test may touch.
        self._object_detector = None
        self.targeted = False


class MockDefaultAttack(MockAttackBase):
    def attack(self, model: BaseModel, x: torch.Tensor, y):  # type: ignore[override]
        return x + 1.0


class MockTOGAttack(MockAttackBase):
    # Class name must be "TOG" for the code path
    __name__ = "TOG"

    def attack(self, x: np.ndarray, tog_variant=None, tog_mislabeling_mode: str = "ml"):
        return (x + 2.0).astype(np.float32)

    @property
    def __class__(self):  # trick to return name "TOG"
        return type("TOG", (), {})


class MockDPatchAttack(MockAttackBase):
    # Class name must be "DPatch" for the code path
    __name__ = "DPatch"

    def __init__(self):
        super().__init__()

    def attack(self, dataloader, mask=None, device=None):  # returns optimized patch
        return torch.zeros(1, 3, 4, 4, device=device)

    def apply_patch(self, x: np.ndarray, patch_external: np.ndarray, random_location: bool = True):
        return x  # identity for simplicity

    @property
    def __class__(self):  # trick to return name "DPatch"
        return type("DPatch", (), {})


def _make_detection_like_loader(num: int = 3):
    # images: Tensor [N, C, H, W], targets: dict[str, list[Tensor]]
    images = torch.randn(num, 3, 4, 4)
    boxes = [torch.tensor([[0.0, 0.0, 1.0, 1.0]]) for _ in range(num)]
    labels = [torch.tensor([1]) for _ in range(num)]
    # store as dataset of tuples (image, targets_dict)
    class _DS:
        def __len__(self):
            return num

        def __getitem__(self, idx):
            return images[idx], {"boxes": boxes, "labels": labels}

    ds = _DS()
    # Use a real DataLoader to satisfy base config checks; keep a collate_fn
    # that returns the single sample as-is so shapes remain consistent.
    return DataLoader(ds, batch_size=1, collate_fn=lambda batch: batch[0])


@pytest.fixture
def base_instance():
    inst = object.__new__(AdversarialODTraining)
    inst.config = SimpleNamespace(train_loader=_make_detection_like_loader())
    inst._device = torch.device("cpu")
    inst._trainable = MockModel()
    inst._model = inst._trainable
    inst._optimizer = SimpleNamespace(param_groups=[{"params": [torch.tensor(1.0, requires_grad=True)]}])
    inst._scheduler = None
    inst._wrapper_name = "dummy"
    inst._od_wrapper = SimpleNamespace(
        prepare_training_inputs=lambda imgs, t: (imgs, t),
        extract_total_loss=lambda out: torch.tensor(1.0, requires_grad=True),
    )
    return inst


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_config_accepts_dict_targets():
    loader = _make_detection_like_loader()
    cfg = SimpleNamespace(model=MockModel(), models=[MockModel()], attacks=[MockDefaultAttack()], train_loader=loader)
    inst = object.__new__(AdversarialODTraining)
    # Should not raise
    inst._check_config(cfg)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_check_config_rejects_wrong_structure():
    # Dataset that yields wrong shape
    ds = TensorDataset(torch.randn(2, 3, 4, 4), torch.randn(2))  # y is Tensor, not dict/list[dict]
    loader = DataLoader(ds, batch_size=1)
    cfg = SimpleNamespace(model=MockModel(), models=[MockModel()], attacks=[MockDefaultAttack()], train_loader=loader)
    inst = object.__new__(AdversarialODTraining)
    with pytest.raises(ValueError, match=r"expects dataset samples"):
        inst._check_config(cfg)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_shuffle_data_with_dict_targets(base_instance):
    images = torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2)
    targets = {"ids": [torch.tensor(0), torch.tensor(1)]}
    out_imgs, out_tgts = base_instance._shuffle_data(images, targets)
    assert isinstance(out_tgts, dict)
    assert set(v.item() for v in out_tgts["ids"]) == {0, 1}
    assert out_imgs.shape == images.shape


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_combine_clean_and_adversarial_data(base_instance):
    images = torch.randn(2, 3, 4, 4)
    adv_images = torch.randn(2, 3, 4, 4)
    targets = {"labels": [torch.tensor(1), torch.tensor(2)]}
    combined_imgs, combined_targets = base_instance._combine_clean_and_adversarial_data(images, adv_images, targets)
    assert combined_imgs.shape[0] == 4
    assert isinstance(combined_targets, dict)
    assert len(combined_targets["labels"]) == 4


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_move_to_device_with_list_images(base_instance):
    images_list = [torch.randn(3, 4, 4), torch.randn(3, 4, 4)]
    targets = {"ids": [torch.tensor(0), torch.tensor(1)]}
    imgs, tgts = base_instance._move_to_device(images_list, targets)
    assert isinstance(imgs, torch.Tensor) and imgs.shape[0] == 2
    assert all(t.device == base_instance._device for t in tgts["ids"])


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_perform_attack_default_path_returns_tensor_on_device(base_instance):
    attack = MockDefaultAttack()
    images = torch.zeros(2, 3, 4, 4)
    targets = {"labels": [torch.tensor(1), torch.tensor(1)]}
    out = base_instance._perform_attack(attack, base_instance._trainable, images, targets)
    assert isinstance(out, torch.Tensor)
    assert out.device == base_instance._device
    assert torch.allclose(out, images + 1.0)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_perform_attack_tog_numpy_to_torch(base_instance):
    attack = MockTOGAttack()
    images = torch.zeros(2, 3, 4, 4)
    targets = {"labels": [torch.tensor(1), torch.tensor(1)]}
    out = base_instance._perform_attack(attack, base_instance._trainable, images, targets)
    assert isinstance(out, torch.Tensor)
    assert out.device == base_instance._device
    assert torch.allclose(out, torch.zeros_like(images) + 2.0)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_perform_attack_dpatch_path_with_cached_patch(base_instance):
    attack = MockDPatchAttack()
    images = torch.zeros(1, 3, 4, 4)
    targets = {"labels": [torch.tensor(1)]}
    out = base_instance._perform_attack(attack, base_instance._trainable, images, targets)
    assert isinstance(out, torch.Tensor)
    assert out.shape == images.shape


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_generate_adversarial_batch_switches_model_modes(base_instance, monkeypatch):
    # Force a deterministic attack selection
    attack = MockDefaultAttack()
    base_instance.config.attacks = [attack]
    images = torch.randn(2, 3, 4, 4)
    targets = {"labels": [torch.tensor(1), torch.tensor(2)]}

    # Model starts in training mode
    base_instance._trainable.train(True)
    adv_images, adv_targets = base_instance._generate_adversarial_batch(images, targets)

    # After generation, training mode should be restored
    assert base_instance._trainable.training is True
    assert adv_images.shape == images.shape
    assert adv_targets == targets


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_shuffle_data_with_list_and_tensor_targets(base_instance):
    data_list = [torch.tensor([i]) for i in range(4)]
    targets_tensor = torch.tensor([0, 1, 2, 3])
    out_data, out_targets = base_instance._shuffle_data(data_list, targets_tensor)
    assert isinstance(out_data, list)
    assert isinstance(out_targets, torch.Tensor)
    assert sorted([int(x.item()) for x in out_data]) == [0, 1, 2, 3]
    assert sorted([int(x.item()) for x in out_targets]) == [0, 1, 2, 3]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_shuffle_data_tensor_with_list_targets(base_instance):
    data = torch.arange(8).view(4, 2)
    targets = [torch.tensor(i) for i in range(4)]
    out_data, out_targets = base_instance._shuffle_data(data, targets)
    assert isinstance(out_data, torch.Tensor)
    assert isinstance(out_targets, list)
    assert sorted(int(t.item()) for t in out_targets) == [0, 1, 2, 3]
    assert sorted(int(d.item()) for d in out_data.view(-1)[::2]) == [0, 2, 4, 6]


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_combine_raises_on_shape_mismatch(base_instance):
    images = torch.randn(2, 3, 4, 4)
    adv_images = torch.randn(1, 3, 4, 4)
    targets = {"labels": [torch.tensor(1), torch.tensor(2)]}
    with pytest.raises(AssertionError):
        base_instance._combine_clean_and_adversarial_data(images, adv_images, targets)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_move_to_device_with_tensor_images(base_instance):
    images = torch.randn(2, 3, 4, 4)
    targets = {"ids": [torch.tensor(0), torch.tensor(1)]}
    imgs, tgts = base_instance._move_to_device(images, targets)
    assert isinstance(imgs, torch.Tensor) and imgs.shape[0] == 2
    assert all(t.device == base_instance._device for t in tgts["ids"])


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_perform_attack_resolves_object_detector_success(base_instance, monkeypatch):
    # Prepare attack with unresolved detector (string)
    class Attack(MockDefaultAttack):
        def __init__(self):
            super().__init__()
            self._object_detector = "yolov5"

    # Fake wrapper returned by get_object_detector
    class _Wrapper:
        def __init__(self):
            self.model = None

    import advsecurenet.computer_vision.object_detection.defenses.adversarial_od_training as mod

    # Resolver is called with keyword argument existing_model
    def _resolver(*args, **kwargs):
        return _Wrapper()
    monkeypatch.setattr(mod, "get_object_detector", _resolver)

    attack = Attack()
    images = torch.zeros(2, 3, 4, 4)
    targets = {"labels": [torch.tensor(1), torch.tensor(1)]}
    out = base_instance._perform_attack(attack, base_instance._trainable, images, targets)
    assert isinstance(attack._object_detector, _Wrapper)
    assert isinstance(out, torch.Tensor)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_perform_attack_resolves_object_detector_fallback_to_model_predict(base_instance, monkeypatch):
    # get_object_detector fails; model has predict -> fallback
    import advsecurenet.computer_vision.object_detection.defenses.adversarial_od_training as mod

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "get_object_detector", _raise)

    class ModelWithPredict(MockModel):
        def predict(self, x):  # type: ignore[no-redef]
            return x

    attack = MockDefaultAttack()
    attack._object_detector = "anything"
    images = torch.zeros(1, 3, 4, 4)
    targets = {"labels": [torch.tensor(1)]}
    out = base_instance._perform_attack(attack, ModelWithPredict(), images, targets)
    assert isinstance(out, torch.Tensor)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_perform_attack_resolves_object_detector_raises_without_predict(base_instance, monkeypatch):
    import advsecurenet.computer_vision.object_detection.defenses.adversarial_od_training as mod

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(mod, "get_object_detector", _raise)

    attack = MockDefaultAttack()
    attack._object_detector = "anything"
    images = torch.zeros(1, 3, 4, 4)
    targets = {"labels": [torch.tensor(1)]}
    # Provide a model object that definitely lacks a 'predict' attribute
    class NoPredict:
        pass
    with pytest.raises(RuntimeError, match=r"Cannot resolve object detector"):
        base_instance._perform_attack(attack, NoPredict(), images, targets)


@pytest.mark.advsecurenet
@pytest.mark.essential
def test_run_epoch_logs_and_uses_batches(monkeypatch):
    # Create a fresh instance with minimal wiring
    inst = object.__new__(AdversarialODTraining)
    inst._device = torch.device("cpu")
    inst._trainable = MockModel()
    inst._model = inst._trainable
    # simple optimizer on a dummy parameter
    param = torch.nn.Parameter(torch.tensor(1.0))
    inst._optimizer = torch.optim.SGD([param], lr=0.1)
    inst._scheduler = None
    # minimal config with one attack for _generate_adversarial_batch
    inst.config = SimpleNamespace(attacks=[MockDefaultAttack()])
    # wrapper returns constant loss tensor with grad
    inst._od_wrapper = SimpleNamespace(
        prepare_training_inputs=lambda imgs, t: (imgs, t),
        extract_total_loss=lambda out: torch.tensor(1.0, requires_grad=True),
    )

    # Provide two batches via _get_train_loader
    images = torch.zeros(1, 3, 4, 4)
    targets = {"labels": [torch.tensor(1)]}
    inst._get_train_loader = lambda epoch: [(images, targets), (images, targets)]
    inst._get_loss_divisor = lambda: 2

    logged = {}
    inst._log_loss = lambda ep, val: logged.update({"epoch": ep, "loss": val})

    inst._run_epoch(epoch=1)
    assert logged["epoch"] == 1
    assert isinstance(logged["loss"], float)


