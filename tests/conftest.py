# tests/conftest.py
import importlib
import sys
import types
import pytest

import torch

# ======================================================================================
# 0) Ensure patch targets actually exist at import time (before any test patches run)
# ======================================================================================
import types

# Ensure torch.utils package object is present
try:
    utils_pkg = sys.modules.get("torch.utils")
    if utils_pkg is None:
        import torch.utils as utils_pkg  # noqa: F401

        utils_pkg = sys.modules.get("torch.utils")
except Exception:
    utils_pkg = None

# Create/attach torch.utils.utils
if utils_pkg is not None and not hasattr(utils_pkg, "utils"):
    try:
        import torchvision.utils as tvutils

        # Mirror torchvision.utils into a module at torch.utils.utils
        tv_mod = types.ModuleType("torch.utils.utils")
        for name in dir(tvutils):
            setattr(tv_mod, name, getattr(tvutils, name))
    except Exception:
        # Fallback stub if torchvision isn't available for some reason
        tv_mod = types.ModuleType("torch.utils.utils")

        def _noop(*a, **k):
            return None

        for fn in ("save_image", "make_grid"):
            setattr(tv_mod, fn, _noop)

    # Register in sys.modules and as attribute on torch.utils
    sys.modules["torch.utils.utils"] = tv_mod
    setattr(sys.modules["torch.utils"], "utils", tv_mod)

# -- torch.backends.cudnn -------------------------------------------------------------
# Guarantee a 'cudnn' attribute on torch.backends so @patch("torch.backends.cudnn") works.
backends_mod = sys.modules.get("torch.backends")
if backends_mod is None:
    backends_mod = types.ModuleType("torch.backends")
    sys.modules["torch.backends"] = backends_mod
    setattr(torch, "backends", backends_mod)

if not hasattr(backends_mod, "cudnn"):
    backends_mod.cudnn = types.SimpleNamespace(
        enabled=False, benchmark=False, deterministic=False
    )

# -- torch.distributed.* --------------------------------------------------------------
# Some tests patch these directly at fixture setup time.
dist_mod = sys.modules.get("torch.distributed")
if dist_mod is None:
    dist_mod = types.ModuleType("torch.distributed")
    sys.modules["torch.distributed"] = dist_mod
    setattr(torch, "distributed", dist_mod)
else:
    # Keep torch.distributed attribute pointing at the module in sys.modules
    setattr(torch, "distributed", dist_mod)


def _noop(*a, **k):
    return None


_defaults = {
    "init_process_group": _noop,
    "destroy_process_group": _noop,
    "is_initialized": lambda: False,
    "get_world_size": lambda *a, **k: 1,
    "get_rank": lambda *a, **k: 0,
    "barrier": _noop,
    "all_reduce": _noop,
}
for name, fn in _defaults.items():
    if not hasattr(dist_mod, name):
        setattr(dist_mod, name, fn)


# ======================================================================================
# 1) Make CPU/Mac test env safe: dummy DistributedSampler
# ======================================================================================


class _DummyDistributedSampler:
    def __init__(self, dataset, num_replicas=None, rank=None, **kwargs):
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def set_epoch(self, epoch):
        pass

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)


@pytest.fixture(autouse=True)
def _neutralize_torch(monkeypatch):
    # Route DistributedSampler lookups to a harmless stub
    monkeypatch.setattr(
        "torch.utils.data.distributed.DistributedSampler",
        _DummyDistributedSampler,
        raising=False,
    )
    # Keep distributed helpers as no-ops even if code touches them
    import torch as _t

    monkeypatch.setattr(
        _t.distributed, "init_process_group", dist_mod.init_process_group, raising=False
    )
    monkeypatch.setattr(
        _t.distributed,
        "destroy_process_group",
        dist_mod.destroy_process_group,
        raising=False,
    )
    monkeypatch.setattr(
        _t.distributed, "is_initialized", dist_mod.is_initialized, raising=False
    )
    monkeypatch.setattr(
        _t.distributed, "get_world_size", dist_mod.get_world_size, raising=False
    )
    monkeypatch.setattr(_t.distributed, "get_rank", dist_mod.get_rank, raising=False)
    monkeypatch.setattr(_t.distributed, "barrier", dist_mod.barrier, raising=False)
    monkeypatch.setattr(
        _t.distributed, "all_reduce", dist_mod.all_reduce, raising=False
    )
    yield


# If cli.shared.utils.dataloader captured the real DistributedSampler in defaults,
# replace that default with our dummy so tests don't touch real dist.
try:
    import cli.shared.utils.dataloader as dl
    from torch.utils.data.distributed import DistributedSampler as _RealDS

    defaults = list(dl.get_dataloader.__defaults__ or ())
    for i, d in enumerate(defaults):
        if d is _RealDS:
            defaults[i] = _DummyDistributedSampler
            dl.get_dataloader.__defaults__ = tuple(defaults)
            break
except Exception:
    pass


# ======================================================================================
# 2) HF tests: align module identity + lenient AutoModel.from_config for MockHFConfig
# ======================================================================================


@pytest.fixture(autouse=True)
def _align_hf_module_identity_and_lenient_automodel(monkeypatch):
    """
    Ensure advsecurenet.models.huggingface_model sees the same `transformers`
    module object as the tests, so their getattr side_effect treats it as 'transformers'.
    Also relax AutoModel.from_config when the tests' MockHFConfig is used.
    """
    import transformers as tf  # real module
    import advsecurenet.models.huggingface_model as hf_mod

    monkeypatch.setattr(hf_mod, "transformers", tf, raising=False)

    try:
        from tests.advsecurenet.models.test_huggingface_model import (
            MockHFModel,
        )  # provided by tests
    except Exception:  # pragma: no cover

        class MockHFModel(torch.nn.Module):
            def __init__(self, *a, **k):
                super().__init__()

    original_from_config = tf.AutoModel.from_config

    def _lenient_from_config(cls, config, **kwargs):
        if getattr(config, "__class__", type(None)).__name__ == "MockHFConfig":
            return MockHFModel()
        return original_from_config.__func__(cls, config, **kwargs)

    monkeypatch.setattr(
        tf.AutoModel, "from_config", classmethod(_lenient_from_config), raising=False
    )
