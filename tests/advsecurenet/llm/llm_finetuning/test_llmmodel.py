import os
import sys
from types import SimpleNamespace, ModuleType
import importlib
import builtins

import pytest


def import_model_module():
    # Always import the module object so we can patch internal helpers.
    return importlib.import_module("advsecurenet.llm_finetuning.model")


def test_load_tokenizer_sets_pad_and_side(monkeypatch):
    m = import_model_module()

    # Fake tokenizer with missing pad_token/pad_token_id, eos present
    tok_obj = SimpleNamespace(
        pad_token=None,
        eos_token="<eos>",
        pad_token_id=None,
        eos_token_id=42,
        padding_side="left",
        save_pretrained=lambda *a, **k: None,
    )

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, use_fast=True):
            assert model_name == "dummy-model"
            assert use_fast is True
            return tok_obj

    monkeypatch.setattr(m, "AutoTokenizer", FakeAutoTokenizer, raising=True)

    cfg = SimpleNamespace(train=SimpleNamespace(model_name="dummy-model"))
    out_tok = m.load_tokenizer(cfg)
    assert out_tok.pad_token == "<eos>"
    assert out_tok.pad_token_id == 42
    assert out_tok.padding_side == "right"


def test_load_tokenizer_preserves_existing_pad(monkeypatch):
    m = import_model_module()

    tok_obj = SimpleNamespace(
        pad_token="<pad>",
        eos_token="<eos>",
        pad_token_id=0,
        eos_token_id=5,
        padding_side="left",
        save_pretrained=lambda *a, **k: None,
    )

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, use_fast=True):
            return tok_obj

    monkeypatch.setattr(m, "AutoTokenizer", FakeAutoTokenizer, raising=True)
    cfg = SimpleNamespace(train=SimpleNamespace(model_name="dummy"))
    out_tok = m.load_tokenizer(cfg)
    # unchanged
    assert out_tok.pad_token == "<pad>"
    assert out_tok.pad_token_id == 0
    # but padding_side should be forced to right
    assert out_tok.padding_side == "right"


def test_in_distributed_via_env_world_size(monkeypatch):
    m = import_model_module()
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert m._in_distributed() is True
    # cleanup
    monkeypatch.delenv("WORLD_SIZE", raising=False)


def test_in_distributed_via_env_local_rank(monkeypatch):
    m = import_model_module()
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    assert m._in_distributed() is True
    monkeypatch.delenv("LOCAL_RANK", raising=False)


def test_in_distributed_via_torch_dist(monkeypatch):
    import importlib
    m = importlib.import_module("advsecurenet.llm_finetuning.model")

    # Force the environment-driven branch to report "distributed"
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.delenv("LOCAL_RANK", raising=False)

    try:
        assert m._in_distributed() is True
    finally:
        monkeypatch.delenv("WORLD_SIZE", raising=False)


def test_in_distributed_exception_path(monkeypatch):
    m = import_model_module()
    # Put an object without the expected attributes; calling is_available will raise
    sys.modules["torch.distributed"] = object()  # attribute error triggers except -> False
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    assert m._in_distributed() is False


def test_select_device_map_variants(monkeypatch):
    m = import_model_module()

    # distributed -> None
    monkeypatch.setattr(m, "_in_distributed", lambda: True, raising=True)
    assert m._select_device_map() is None

    # single process + cuda -> "auto"
    monkeypatch.setattr(m, "_in_distributed", lambda: False, raising=True)
    monkeypatch.setattr(m.torch.cuda, "is_available", lambda: True, raising=True)
    assert m._select_device_map() == "auto"

    # single process + no cuda -> "cpu"
    monkeypatch.setattr(m.torch.cuda, "is_available", lambda: False, raising=True)
    assert m._select_device_map() == "cpu"


def _fake_cfg(peft_enabled=False, quant="qlora"):
    train = SimpleNamespace(model_name="tiny-model")
    peft = SimpleNamespace(
        enabled=peft_enabled,
        quantization=quant,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
    )
    return SimpleNamespace(train=train, peft=peft)


class _DummyModel:
    def __init__(self, name):
        self.name = name


def test_load_model_single_process_no_peft_auto_device(monkeypatch):
    m = import_model_module()

    # Make device map "auto"
    monkeypatch.setattr(m, "_select_device_map", lambda: "auto", raising=True)

    captured = {}

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name, device_map=None, load_in_4bit=False):
            captured["args"] = (model_name,)
            captured["kwargs"] = {"device_map": device_map, "load_in_4bit": load_in_4bit}
            return _DummyModel("base")

    monkeypatch.setattr(m, "AutoModelForCausalLM", FakeAutoModel, raising=True)

    cfg = _fake_cfg(peft_enabled=False)
    model = m.load_model(cfg)

    assert isinstance(model, _DummyModel)
    assert captured["args"][0] == "tiny-model"
    assert captured["kwargs"]["device_map"] == "auto"
    assert captured["kwargs"]["load_in_4bit"] is False  # peft disabled -> q4 False


def test_load_model_distributed_peft_with_qlora(monkeypatch):
    m = import_model_module()

    # Distributed -> device_map None
    monkeypatch.setattr(m, "_select_device_map", lambda: None, raising=True)

    # Fake base model load
    base_model = _DummyModel("base")

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name, device_map=None, load_in_4bit=False):
            assert device_map is None  # distributed path
            assert load_in_4bit is True  # qlora -> 4bit
            return base_model

    monkeypatch.setattr(m, "AutoModelForCausalLM", FakeAutoModel, raising=True)

    # Track prepare & get_peft_model calls
    calls = {"prepare": 0, "get_peft": 0, "lora_args": None}

    def fake_prepare(model):
        calls["prepare"] += 1
        # Return a new wrapper to ensure the flow uses the prepared model
        return _DummyModel("prepared")

    class FakeLoraConfig:
        def __init__(self, **kw):
            calls["lora_args"] = kw

    def fake_get_peft_model(model, lora_cfg):
        calls["get_peft"] += 1
        # Return final wrapped model
        return _DummyModel("wrapped")

    monkeypatch.setattr(m, "prepare_model_for_kbit_training", fake_prepare, raising=True)
    monkeypatch.setattr(m, "LoraConfig", FakeLoraConfig, raising=True)
    monkeypatch.setattr(m, "get_peft_model", fake_get_peft_model, raising=True)

    cfg = _fake_cfg(peft_enabled=True, quant="qlora")
    out = m.load_model(cfg)

    assert isinstance(out, _DummyModel) and out.name == "wrapped"
    assert calls["prepare"] == 1
    assert calls["get_peft"] == 1
    # Check key LoRA args wired through
    assert calls["lora_args"]["r"] == 8
    assert calls["lora_args"]["lora_alpha"] == 16
    assert calls["lora_args"]["lora_dropout"] == 0.05
    assert calls["lora_args"]["target_modules"] == ["q_proj", "v_proj"]
    assert calls["lora_args"]["bias"] == "none"
    assert calls["lora_args"]["task_type"] == "CAUSAL_LM"


def test_load_model_peft_without_quantization(monkeypatch):
    m = import_model_module()

    # Non-distributed + no cuda to ensure "cpu" path (not strictly required here)
    monkeypatch.setattr(m, "_select_device_map", lambda: "cpu", raising=True)

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(model_name, device_map=None, load_in_4bit=False):
            assert load_in_4bit is False  # quant not in {"qlora","bnb-4bit"}
            return _DummyModel("base")

    monkeypatch.setattr(m, "AutoModelForCausalLM", FakeAutoModel, raising=True)

    calls = {"prepare": 0, "get_peft": 0}

    def fake_prepare(model):
        calls["prepare"] += 1
        return model

    def fake_get_peft(model, lora_cfg):
        calls["get_peft"] += 1
        return _DummyModel("wrapped-noq")

    monkeypatch.setattr(m, "prepare_model_for_kbit_training", fake_prepare, raising=True)
    monkeypatch.setattr(m, "get_peft_model", fake_get_peft, raising=True)

    cfg = _fake_cfg(peft_enabled=True, quant="none")
    out = m.load_model(cfg)

    assert isinstance(out, _DummyModel) and out.name == "wrapped-noq"
    # prepare should NOT be called when q4 is False
    assert calls["prepare"] == 0
    assert calls["get_peft"] == 1