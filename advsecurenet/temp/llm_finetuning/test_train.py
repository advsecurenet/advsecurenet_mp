# tests/test_train_main.py
import sys
import types
import runpy
from types import SimpleNamespace
import pytest


# ---- Minimal fakes ----
class FakeTok:
    saved = []

    def save_pretrained(self, path):
        self.__class__.saved.append(path)


class FakeModel:
    pass


class FakeTrainingArguments:
    last = None

    def __init__(self, **kwargs):
        self.__class__.last = kwargs


class FakeDataCollatorForLanguageModeling:
    last = None

    def __init__(self, tokenizer, mlm):
        self.__class__.last = {"tokenizer": tokenizer, "mlm": mlm}


class FakeTrainer:
    last_instance = None

    def __init__(self, **kwargs):
        self.__class__.last_instance = self
        self.kwargs = kwargs
        self.trained = False
        self.saved_path = None

    def train(self):
        self.trained = True

    def save_model(self, path):
        self.saved_path = path


def install_fake_transformers(monkeypatch):
    tfm = types.ModuleType("transformers")
    tfm.TrainingArguments = FakeTrainingArguments
    tfm.Trainer = FakeTrainer
    tfm.DataCollatorForLanguageModeling = FakeDataCollatorForLanguageModeling
    monkeypatch.setitem(sys.modules, "transformers", tfm)


def install_fake_torch(monkeypatch):
    torch = types.ModuleType("torch")
    calls = {"manual_seed": [], "cuda_manual_seed_all": []}

    def manual_seed(s):
        calls["manual_seed"].append(s)

    def is_available():
        return False  # no-CUDA path is fine here

    def manual_seed_all(s):
        calls["cuda_manual_seed_all"].append(s)

    torch.manual_seed = manual_seed
    torch.cuda = SimpleNamespace(
        is_available=is_available, manual_seed_all=manual_seed_all
    )
    torch._calls = calls
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


def install_stub_pkg(monkeypatch):
    # Ensure package namespace exists
    pkg = sys.modules.setdefault("advsecurenet", types.ModuleType("advsecurenet"))
    sub = sys.modules.setdefault(
        "advsecurenet.llm_finetuning", types.ModuleType("advsecurenet.llm_finetuning")
    )

    # Stub config with load_yaml returning a minimal cfg; record the path received
    cfg_calls = {}
    cfg_mod = types.ModuleType("advsecurenet.llm_finetuning.config")

    class Config:
        pass

    def load_yaml(path):
        cfg_calls["path"] = path
        return SimpleNamespace(
            data=SimpleNamespace(),
            train=SimpleNamespace(
                seed=1,
                output_dir="./out",
                learning_rate=1e-4,
                weight_decay=0.0,
                max_steps=1,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                logging_steps=1,
                save_steps=1,
                eval_steps=1,
                evaluation_strategy="steps",
                lr_scheduler_type="linear",
                warmup_ratio=0.0,
                bf16=False,
                fp16=False,
                gradient_checkpointing=False,
                push_to_hub=False,
            ),
        )

    cfg_mod.Config = Config
    cfg_mod.load_yaml = load_yaml
    monkeypatch.setitem(sys.modules, "advsecurenet.llm_finetuning.config", cfg_mod)

    # Stub model module
    model_mod = types.ModuleType("advsecurenet.llm_finetuning.model")
    model_mod.load_model = lambda cfg: FakeModel()
    model_mod.load_tokenizer = lambda cfg: FakeTok()
    monkeypatch.setitem(sys.modules, "advsecurenet.llm_finetuning.model", model_mod)

    # Stub data module
    data_mod = types.ModuleType("advsecurenet.llm_finetuning.data")
    data_mod.load_tokenized_datasets = lambda data_cfg, tok: {
        "train": ["t"],
        "validation": ["v"],
    }
    monkeypatch.setitem(sys.modules, "advsecurenet.llm_finetuning.data", data_mod)

    return cfg_calls


@pytest.mark.usefixtures()
def test_main_block_executes(monkeypatch):
    # Install fakes before executing as __main__
    install_fake_transformers(monkeypatch)
    install_fake_torch(monkeypatch)
    cfg_calls = install_stub_pkg(monkeypatch)

    # Ensure a fresh run of the module code
    sys.modules.pop("advsecurenet.llm_finetuning.train", None)

    # Execute the target module as a script -> hits lines 57–64
    result_globals = runpy.run_module(
        "advsecurenet.llm_finetuning.train", run_name="__main__"
    )

    # __main__ should have called load_yaml with the hardcoded path
    assert (
        cfg_calls["path"]
        == "advsecurenet_mp/advsecurenet/llm_finetuning/configs/instruction.yaml"
    )

    # run_training should have been executed (via our faked Trainer etc.)
    tr = FakeTrainer.last_instance
    assert tr is not None and tr.trained is True
    assert tr.saved_path == "./out"
    assert FakeTok.saved == ["./out"]
