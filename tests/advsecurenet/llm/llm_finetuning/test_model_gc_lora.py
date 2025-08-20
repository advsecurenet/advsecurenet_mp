from types import SimpleNamespace
from unittest.mock import patch
from advsecurenet.llm_finetuning.config import Config, TrainConfig, DataConfig, PEFTConfig
# tests/test_model.py
import types
import sys
import importlib
from types import SimpleNamespace
import pytest


class _FakeModel:
    def __init__(self, tag, base=None, config=None):
        self.tag = tag
        self.base = base
        self.config = config

    def __repr__(self):
        return f"<FakeModel tag={self.tag}>"


class _FakeAutoModelForCausalLM:
    last_call = None

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.last_call = (model_name, kwargs)
        # return a base model instance
        return _FakeModel(tag="base")


class _FakeAutoTokenizer:
    last_call = None

    def __init__(self):
        # These will be tweaked by tests before calling load_tokenizer
        self.pad_token = None
        self.eos_token = "<eos>"
        self.padding_side = "left"

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.last_call = (model_name, kwargs)
        return cls()


# peft fakes
class _FakeLoraConfig:
    last_init_args = None
    def __init__(self, *args, **kwargs):
        _FakeLoraConfig.last_init_args = (args, kwargs)
        # store for test introspection
        self.args = args
        self.kwargs = kwargs


class _FakePeft:
    prepared_calls = 0
    get_calls = []

    @staticmethod
    def prepare_model_for_kbit_training(model):
        _FakePeft.prepared_calls += 1
        return _FakeModel(tag="prepared", base=model)

    @staticmethod
    def get_peft_model(model, lora_cfg):
        _FakePeft.get_calls.append((model, lora_cfg))
        return _FakeModel(tag="peft", base=model, config=lora_cfg)


def _install_fake_modules(monkeypatch):
    """Inject fake 'transformers' and 'peft' into sys.modules so import works anywhere."""
    tfm = types.ModuleType("transformers")
    tfm.AutoModelForCausalLM = _FakeAutoModelForCausalLM
    tfm.AutoTokenizer = _FakeAutoTokenizer

    peft = types.ModuleType("peft")
    peft.LoraConfig = _FakeLoraConfig
    peft.get_peft_model = _FakePeft.get_peft_model
    peft.prepare_model_for_kbit_training = _FakePeft.prepare_model_for_kbit_training

    monkeypatch.setitem(sys.modules, "transformers", tfm)
    monkeypatch.setitem(sys.modules, "peft", peft)


def _import_model_module(monkeypatch):
    _install_fake_modules(monkeypatch)
    # Import your module under test (prefer your repo path)
    try:
        mod = importlib.import_module("advsecurenet.llm_finetuning.model")
    except ModuleNotFoundError:
        mod = importlib.import_module("llm_finetune.model")
    # Ensure the names inside the module reference our fakes
    # (safety: if real libs exist in environment)
    monkeypatch.setattr(mod, "AutoModelForCausalLM", _FakeAutoModelForCausalLM, raising=False)
    monkeypatch.setattr(mod, "AutoTokenizer", _FakeAutoTokenizer, raising=False)
    # Bind peft callables/classes
    monkeypatch.setattr(mod, "LoraConfig", _FakeLoraConfig, raising=False)
    monkeypatch.setattr(mod, "get_peft_model", _FakePeft.get_peft_model, raising=False)
    monkeypatch.setattr(mod, "prepare_model_for_kbit_training", _FakePeft.prepare_model_for_kbit_training, raising=False)
    return mod


def _cfg(**overrides):
    # Minimal config tree to satisfy annotations/attribute access
    defaults = dict(
        train=SimpleNamespace(model_name="fake-model"),
        peft=SimpleNamespace(
            enabled=False,
            quantization=None,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
        ),
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


# ------------------------------ Tests ------------------------------
def test_load_tokenizer_sets_pad_and_padding_side_when_missing(monkeypatch):
    mod = _import_model_module(monkeypatch)

    # Fresh tokenizer state
    _FakeAutoTokenizer.last_call = None
    tok = mod.load_tokenizer(_cfg())

    # Called with correct args
    assert _FakeAutoTokenizer.last_call[0] == "fake-model"
    assert _FakeAutoTokenizer.last_call[1]["use_fast"] is True

    # pad_token should be set to eos if missing, padding_side forced right
    assert tok.pad_token == tok.eos_token == "<eos>"
    assert tok.padding_side == "right"


def test_load_tokenizer_preserves_existing_pad_token(monkeypatch):
    mod = _import_model_module(monkeypatch)

    # Make from_pretrained return a tokenizer that already has a pad_token
    orig_from_pretrained = _FakeAutoTokenizer.from_pretrained

    def with_pad(*args, **kwargs):
        t = _FakeAutoTokenizer()
        t.pad_token = "<pad>"
        t.eos_token = "<eosX>"
        return t

    _FakeAutoTokenizer.from_pretrained = classmethod(with_pad)
    try:
        tok = mod.load_tokenizer(_cfg())
        assert tok.pad_token == "<pad>"          # preserved
        assert tok.eos_token == "<eosX>"         # untouched
        assert tok.padding_side == "right"       # still forced
    finally:
        # restore
        _FakeAutoTokenizer.from_pretrained = orig_from_pretrained


def test_load_model_no_peft(monkeypatch):
    mod = _import_model_module(monkeypatch)

    # Reset logs
    _FakeAutoModelForCausalLM.last_call = None
    _FakePeft.prepared_calls = 0
    _FakePeft.get_calls.clear()

    cfg = _cfg(peft=SimpleNamespace(enabled=False, quantization=None))
    model = mod.load_model(cfg)

    # Asserts
    name, kwargs = _FakeAutoModelForCausalLM.last_call
    assert name == "fake-model"
    assert kwargs["device_map"] == "auto"
    assert kwargs["load_in_4bit"] is False
    assert isinstance(model, _FakeModel) and model.tag == "base"
    assert _FakePeft.prepared_calls == 0
    assert _FakePeft.get_calls == []


@pytest.mark.parametrize("quant", ["qlora", "bnb-4bit"])
def test_load_model_with_peft_and_q4_true_triggers_prepare_and_get_peft(monkeypatch, quant):
    mod = _import_model_module(monkeypatch)

    # Reset logs
    _FakeAutoModelForCausalLM.last_call = None
    _FakePeft.prepared_calls = 0
    _FakePeft.get_calls.clear()
    _FakeLoraConfig.last_init_args = None

    peft_ns = SimpleNamespace(
        enabled=True,
        quantization=quant,
        r=32,
        lora_alpha=64,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj"],
    )
    cfg = _cfg(peft=peft_ns)

    model = mod.load_model(cfg)

    # Correct 4-bit flag on base load
    _, kwargs = _FakeAutoModelForCausalLM.last_call
    assert kwargs["load_in_4bit"] is True

    # prepare_model_for_kbit_training called
    assert _FakePeft.prepared_calls == 1
    # get_peft_model called with prepared model and a LoraConfig
    assert len(_FakePeft.get_calls) == 1
    called_model, called_cfg = _FakePeft.get_calls[0]
    assert isinstance(called_model, _FakeModel) and called_model.tag == "prepared"
    assert isinstance(called_cfg, _FakeLoraConfig)

    # LoraConfig built with expected params
    args, kws = _FakeLoraConfig.last_init_args
    assert kws["r"] == 32
    assert kws["lora_alpha"] == 64
    assert abs(kws["lora_dropout"] - 0.1) < 1e-9
    assert kws["target_modules"] == ["q_proj", "k_proj", "v_proj"]
    assert kws["bias"] == "none"
    assert kws["task_type"] == "CAUSAL_LM"

    # Final model is the peft-wrapped one
    assert isinstance(model, _FakeModel) and model.tag == "peft"


def test_load_model_with_peft_but_not_q4(monkeypatch):
    mod = _import_model_module(monkeypatch)

    _FakeAutoModelForCausalLM.last_call = None
    _FakePeft.prepared_calls = 0
    _FakePeft.get_calls.clear()
    _FakeLoraConfig.last_init_args = None

    peft_ns = SimpleNamespace(
        enabled=True,
        quantization="none",  # not in {"qlora", "bnb-4bit"} -> q4=False
        r=4,
        lora_alpha=8,
        lora_dropout=0.2,
        target_modules=["something"],
    )
    cfg = _cfg(peft=peft_ns)

    model = mod.load_model(cfg)

    # Base load should be in full precision (no 4-bit)
    _, kwargs = _FakeAutoModelForCausalLM.last_call
    assert kwargs["load_in_4bit"] is False

    # No kbit preparation
    assert _FakePeft.prepared_calls == 0

    # get_peft_model still called
    assert len(_FakePeft.get_calls) == 1
    called_model, called_cfg = _FakePeft.get_calls[0]
    assert isinstance(called_model, _FakeModel) and called_model.tag == "base"
    assert isinstance(called_cfg, _FakeLoraConfig)

    # LoraConfig params
    _args, kws = _FakeLoraConfig.last_init_args
    assert kws["r"] == 4
    assert kws["lora_alpha"] == 8
    assert abs(kws["lora_dropout"] - 0.2) < 1e-9
    assert kws["target_modules"] == ["something"]
    assert kws["bias"] == "none"
    assert kws["task_type"] == "CAUSAL_LM"

    assert isinstance(model, _FakeModel) and model.tag == "peft"
