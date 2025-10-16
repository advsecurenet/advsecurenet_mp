import sys
import types
from types import SimpleNamespace
from pathlib import Path
import importlib
import logging

import pytest
from pydantic import BaseModel


# ---------- helpers ----------

def import_utils():
    # import once; we patch attributes on the module directly in each test
    return importlib.import_module("advsecurenet.llm_finetuning.utils")


def make_fake_torch(
    *,
    cuda_available=True,
    capability_major=8,
    mem_alloc=3 * 1024**3,    # 3 GB
    mem_max=5 * 1024**3,      # 5 GB
    has_compile=True,
):
    calls = {"manual_seed": [], "cuda_manual_seed_all": [], "compile": []}

    torch = types.ModuleType("torch")
    # dtypes we compare against in utils
    torch.float16 = "float16"
    torch.float32 = "float32"
    torch.bfloat16 = "bfloat16"

    # RNG
    def manual_seed(seed):
        calls["manual_seed"].append(seed)

    torch.manual_seed = manual_seed

    # compile
    if has_compile:
        def _compile(model):
            calls["compile"].append(True)
            return SimpleNamespace(compiled=True, base=model)
        torch.compile = _compile  # type: ignore[attr-defined]

    # cuda
    def is_available():
        return cuda_available

    def manual_seed_all(seed):
        calls["cuda_manual_seed_all"].append(seed)

    def current_device():
        return 0

    def get_device_capability(_dev=None):
        return (capability_major, 0)

    def memory_allocated():
        return mem_alloc

    def max_memory_allocated():
        return mem_max

    torch.cuda = SimpleNamespace(
        is_available=is_available,
        manual_seed_all=manual_seed_all,
        current_device=current_device,
        get_device_capability=get_device_capability,
        memory_allocated=memory_allocated,
        max_memory_allocated=max_memory_allocated,
    )
    torch._calls = calls
    return torch


# ---------- tests ----------

def test_set_seed_all_cuda_and_deterministic(monkeypatch):
    import sys, types, importlib
    u = importlib.import_module("advsecurenet.llm_finetuning.utils")

    # Build a full fake torch package and install it into sys.modules
    torch_fake = types.ModuleType("torch")
    calls = {"manual_seed": [], "cuda_manual_seed_all": [], "use_det": []}

    def manual_seed(s): calls["manual_seed"].append(s)
    torch_fake.manual_seed = manual_seed

    # Optional deterministic API
    def use_deterministic_algorithms(flag): calls["use_det"].append(flag)
    torch_fake.use_deterministic_algorithms = use_deterministic_algorithms  # ok if unused

    # minimal dtype constants used by utils.default_dtype (not used here but harmless)
    torch_fake.float16 = "float16"
    torch_fake.float32 = "float32"
    torch_fake.bfloat16 = "bfloat16"

    # CUDA shim
    def is_available(): return True
    def manual_seed_all(s): calls["cuda_manual_seed_all"].append(s)
    def current_device(): return 0
    def get_device_capability(_dev=None): return (8, 0)

    torch_fake.cuda = types.SimpleNamespace(
        is_available=is_available,
        manual_seed_all=manual_seed_all,
        current_device=current_device,
        get_device_capability=get_device_capability,
        memory_allocated=lambda: 0,
        max_memory_allocated=lambda: 0,
    )

    # Provide torch.backends.cudnn module so the import in utils hits our stub
    backends_mod = types.ModuleType("torch.backends")
    cudnn_mod = types.ModuleType("torch.backends.cudnn")
    cudnn_mod.deterministic = False
    cudnn_mod.benchmark = True

    # Install fakes into the import system
    sys.modules["torch"] = torch_fake
    sys.modules["torch.backends"] = backends_mod
    sys.modules["torch.backends.cudnn"] = cudnn_mod

    # Also make utils use our fake torch object directly
    monkeypatch.setattr(u, "torch", torch_fake, raising=True)
    # Record Python/NumPy seeds
    py_calls, np_calls = [], []
    monkeypatch.setattr(u.random, "seed", lambda s: py_calls.append(s))
    monkeypatch.setattr(u.np.random, "seed", lambda s: np_calls.append(s))

    # Act
    u.set_seed_all(123, deterministic=True)

    # Assert
    assert py_calls == [123]
    assert np_calls == [123]
    assert calls["manual_seed"] == [123]
    assert calls["cuda_manual_seed_all"] == [123]
    assert cudnn_mod.deterministic is True
    assert cudnn_mod.benchmark is False

def test_set_seed_all_cpu_only(monkeypatch):
    u = import_utils()
    fake = make_fake_torch(cuda_available=False)
    monkeypatch.setattr(u, "torch", fake, raising=True)

    py_calls, np_calls = [], []
    monkeypatch.setattr(u.random, "seed", lambda s: py_calls.append(s))
    monkeypatch.setattr(u.np.random, "seed", lambda s: np_calls.append(s))
    u.set_seed_all(7, deterministic=True)  # should NOT touch cuda branch

    assert py_calls == [7]
    assert np_calls == [7]
    assert fake._calls["cuda_manual_seed_all"] == []


def test_dist_info_env_defaults_and_overrides(monkeypatch):
    u = import_utils()
    # Stub a safe torch.distributed that is *not* initialized
    dist = types.ModuleType("torch.distributed")
    dist.is_available = lambda: False
    dist.is_initialized = lambda: False
    sys.modules["torch.distributed"] = dist

    # clear env -> defaults
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    info = u.dist_info()
    assert info == {"world_size": 1, "rank": 0, "local_rank": 0}

    # set env -> should be reflected
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("LOCAL_RANK", "1")
    info = u.dist_info()
    assert info == {"world_size": 4, "rank": 2, "local_rank": 1}


def test_dist_info_when_distributed_initialized(monkeypatch):
    import sys, types, importlib
    u = importlib.import_module("advsecurenet.llm_finetuning.utils")

    # Install a fake torch + distributed so "import torch.distributed as dist" uses our stub
    torch_fake = types.ModuleType("torch")
    torch_fake.cuda = types.SimpleNamespace(is_available=lambda: False)  # irrelevant here
    sys.modules["torch"] = torch_fake

    dist_mod = types.ModuleType("torch.distributed")
    dist_mod.is_available = lambda: True
    dist_mod.is_initialized = lambda: True
    dist_mod.get_world_size = lambda: 8
    dist_mod.get_rank = lambda: 3
    sys.modules["torch.distributed"] = dist_mod

    # LOCAL_RANK preference
    monkeypatch.setenv("LOCAL_RANK", "7")

    info = u.dist_info()
    assert info == {"world_size": 8, "rank": 3, "local_rank": 7}
    assert u.is_rank_zero() is False




def test_rank_zero_only_decorator(monkeypatch, tmp_path):
    u = import_utils()
    ran = {"v": False}

    @u.rank_zero_only
    def f(p):
        ran["v"] = True

    monkeypatch.setattr(u, "is_rank_zero", lambda: False)
    f(tmp_path / "x.json")
    assert ran["v"] is False

    monkeypatch.setattr(u, "is_rank_zero", lambda: True)
    f(tmp_path / "x.json")
    assert ran["v"] is True


def test_bf16_supported_true_false_and_exception(monkeypatch):
    u = import_utils()
    # True case: cuda + Ampere+
    fake = make_fake_torch(cuda_available=True, capability_major=8)
    monkeypatch.setattr(u, "torch", fake, raising=True)
    assert u.bf16_supported() is True

    # False case: cuda but < Ampere
    fake = make_fake_torch(cuda_available=True, capability_major=7)
    monkeypatch.setattr(u, "torch", fake, raising=True)
    assert u.bf16_supported() is False

    # Exception in capability -> False
    fake = make_fake_torch(cuda_available=True)
    def boom(_dev=None): raise RuntimeError("boom")
    fake.cuda.get_device_capability = boom
    monkeypatch.setattr(u, "torch", fake, raising=True)
    assert u.bf16_supported() is False


def test_default_dtype_paths(monkeypatch):
    u = import_utils()
    fake = make_fake_torch(cuda_available=True)
    monkeypatch.setattr(u, "torch", fake, raising=True)

    monkeypatch.setattr(u, "bf16_supported", lambda: True)
    assert u.default_dtype() == "bfloat16"

    monkeypatch.setattr(u, "bf16_supported", lambda: False)
    assert u.default_dtype() == "float16"

    fake = make_fake_torch(cuda_available=False)
    monkeypatch.setattr(u, "torch", fake, raising=True)
    assert u.default_dtype() == "float32"


def test_device_map_auto(monkeypatch):
    u = import_utils()
    fake = make_fake_torch(cuda_available=True)
    monkeypatch.setattr(u, "torch", fake, raising=True)
    assert u.device_map_auto() == "auto"

    fake = make_fake_torch(cuda_available=False)
    monkeypatch.setattr(u, "torch", fake, raising=True)
    assert u.device_map_auto() == "cpu"


def test_ensure_padding_token_all_branches(monkeypatch):
    u = import_utils()

    # pad/eos present -> preserve pad and change side
    tok = SimpleNamespace(pad_token="<pad>", eos_token="<eos>", pad_token_id=0, eos_token_id=5, padding_side="left")
    u.ensure_padding_token(tok)
    assert tok.pad_token == "<pad>"
    assert tok.pad_token_id == 0
    assert tok.padding_side == "right"

    # pad missing, eos present -> set both pad token & id
    tok = SimpleNamespace(pad_token=None, eos_token="<eos>", pad_token_id=None, eos_token_id=42, padding_side="left")
    u.ensure_padding_token(tok)
    assert tok.pad_token == "<eos>"
    assert tok.pad_token_id == 42
    assert tok.padding_side == "right"

    # pad missing, eos missing -> only side flips
    tok = SimpleNamespace(pad_token=None, eos_token=None, pad_token_id=None, eos_token_id=None, padding_side="left")
    u.ensure_padding_token(tok)
    assert tok.pad_token is None and tok.pad_token_id is None and tok.padding_side == "right"


def test_lora_default_targets_found_and_fallback():
    u = import_utils()

    class M1:
        def named_modules(self):
            return [("x.attn.q_proj", object()), ("y.attn.v_proj", object()), ("z", object())]

    class M2:
        def named_modules(self):
            return [("foo.bar", object())]

    assert u.lora_default_targets(M1()) == ["x.attn.q_proj", "y.attn.v_proj"]
    assert u.lora_default_targets(M2()) == ["q_proj", "v_proj"]


def test_ensure_dir_and_save_json(monkeypatch, tmp_path):
    u = import_utils()
    # When not rank zero, save_json does nothing
    monkeypatch.setattr(u, "is_rank_zero", lambda: False)
    p = tmp_path / "sub" / "a.json"
    u.save_json({"k": 1}, p)
    assert not p.exists()

    # rank zero -> writes
    monkeypatch.setattr(u, "is_rank_zero", lambda: True)
    u.save_json({"k": 1}, p)
    assert p.exists() and '"k": 1' in p.read_text()


def test_save_run_config_both_branches(monkeypatch, tmp_path):
    u = import_utils()
    monkeypatch.setattr(u, "is_rank_zero", lambda: True)

    # BaseModel branch
    class M(BaseModel):
        a: int
    out = tmp_path / "o1"
    u.save_run_config(M(a=7), out)
    t = (out / "config.resolved.json").read_text()
    assert '"a": 7' in t

    # dict-like fallback
    out2 = tmp_path / "o2"
    u.save_run_config({"b": 9}, out2)
    t2 = (out2 / "config.resolved.json").read_text()
    assert '"b": 9' in t2


def test_resolve_path(tmp_path):
    u = import_utils()
    base = tmp_path / "base"
    rel = "x.txt"
    abs_p = tmp_path / "abs.txt"
    assert u.resolve_path(base, rel) == base / rel
    assert u.resolve_path(base, str(abs_p)) == abs_p
    assert u.resolve_path(base, None) is None


def test_time_block_logs_only_on_rank_zero(monkeypatch, caplog):
    u = import_utils()
    caplog.set_level(logging.INFO)

    # rank zero -> logs
    monkeypatch.setattr(u, "is_rank_zero", lambda: True)
    with u.time_block("unit-work-1"):
        pass
    assert any("unit-work-1" in rec.message for rec in caplog.records)

    # not rank zero -> no log
    caplog.clear()
    monkeypatch.setattr(u, "is_rank_zero", lambda: False)
    with u.time_block("unit-work-2"):
        pass
    assert not any("unit-work-2" in rec.message for rec in caplog.records)


def test_cuda_mem_available_and_fallback(monkeypatch):
    u = import_utils()

    # available path with numbers
    fake = make_fake_torch(cuda_available=True, mem_alloc=2 * 1024**3, mem_max=6 * 1024**3)
    monkeypatch.setattr(u, "torch", fake, raising=True)
    assert u.cuda_mem() == {"allocated_gb": 2.0, "max_allocated_gb": 6.0}

    # exception path -> zeros
    def boom():
        raise RuntimeError("boom")
    fake.cuda.memory_allocated = boom
    monkeypatch.setattr(u, "torch", fake, raising=True)
    assert u.cuda_mem() == {"allocated_gb": 0.0, "max_allocated_gb": 0.0}

    # not available -> zeros
    fake = make_fake_torch(cuda_available=False)
    monkeypatch.setattr(u, "torch", fake, raising=True)
    assert u.cuda_mem() == {"allocated_gb": 0.0, "max_allocated_gb": 0.0}


def test_count_trainable_params():
    u = import_utils()

    class P:
        def __init__(self, n, req): self._n, self.requires_grad = n, req
        def numel(self): return self._n

    class M:
        def parameters(self): return [P(10, True), P(30, False), P(60, True)]

    tr, pct = u.count_trainable_params(M())
    assert tr == 70 and abs(pct - 70.0) < 1e-6


def test_require_bitsandbytes_if_needed(monkeypatch):
    u = import_utils()
    # present -> ok
    sys.modules["bitsandbytes"] = types.ModuleType("bitsandbytes")
    u.require_bitsandbytes_if_needed(True)

    # absent -> raises
    sys.modules.pop("bitsandbytes", None)
    with pytest.raises(RuntimeError):
        u.require_bitsandbytes_if_needed(True)

    # disabled -> no-op
    u.require_bitsandbytes_if_needed(False)


def test_maybe_torch_compile_all_paths(monkeypatch):
    u = import_utils()
    model = object()

    # enabled + has compile
    fake = make_fake_torch(has_compile=True)
    monkeypatch.setattr(u, "torch", fake, raising=True)
    compiled = u.maybe_torch_compile(model, enabled=True)
    assert getattr(compiled, "compiled", False) is True
    assert fake._calls["compile"] == [True]

    # enabled + no compile -> returns unchanged
    fake2 = make_fake_torch(has_compile=False)
    monkeypatch.setattr(u, "torch", fake2, raising=True)
    same = u.maybe_torch_compile(model, enabled=True)
    assert same is model

    # disabled -> returns unchanged
    fake3 = make_fake_torch(has_compile=True)
    monkeypatch.setattr(u, "torch", fake3, raising=True)
    same2 = u.maybe_torch_compile(model, enabled=False)
    assert same2 is model