from types import SimpleNamespace
import builtins
import pytest


from advsecurenet.llm_finetuning import data as data_mod


class FakeDataset:
    """Minimal in-memory stand-in for a HF Dataset with .map and .column_names."""
    def __init__(self, rows):
        self.rows = list(rows)

    @property
    def column_names(self):
        if not self.rows:
            return []
        return list(self.rows[0].keys())

    def map(self, func, batched=False, remove_columns=None):
        if not batched:
            new_rows = []
            for r in self.rows:
                out = func(r)  # dict
                if remove_columns:
                    new_rows.append(dict(out))
                else:
                    merged = dict(r)
                    merged.update(out)
                    new_rows.append(merged)
            return FakeDataset(new_rows)

        # batched=True
        if not self.rows:
            return FakeDataset([])
        batch = {"text": [r["text"] for r in self.rows]}
        out = func(batch)  # dict[str, list]
        keys = list(out.keys())
        n = len(out[keys[0]]) if keys else 0
        new_rows = []
        for i in range(n):
            row = {k: out[k][i] for k in keys}
            if not remove_columns:
                row.update(self.rows[i])
            new_rows.append(row)
        return FakeDataset(new_rows)


class FakeDatasetDict:
    """Dict-like stand-in for datasets.DatasetDict with .map across splits."""
    def __init__(self, data=None):
        self._data = dict(data or {})

    def __getitem__(self, k):
        return self._data[k]

    def __setitem__(self, k, v):
        self._data[k] = v

    def __contains__(self, k):
        return k in self._data

    def items(self):
        return self._data.items()

    def map(self, func, batched=False, remove_columns=None):
        return FakeDatasetDict({k: ds.map(func, batched=batched, remove_columns=remove_columns)
                                for k, ds in self._data.items()})


class FakeTokenizer:
    """Records calls; returns deterministic token shapes."""
    def __init__(self):
        self.calls = []

    def __call__(self, texts, **kwargs):
        assert isinstance(texts, list), "tokenizer should get a batch (list of strings)"
        self.calls.append((list(texts), dict(kwargs)))
        max_len = kwargs.get("max_length", 128)
        lens = [min(max_len, len(t)) for t in texts]
        input_ids = [[1]*L for L in lens]
        attention_mask = [[1]*L for L in lens]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def make_cfg(**overrides):
    """Fake DataConfig using SimpleNamespace — python won't enforce the type."""
    defaults = dict(
        train_file=None,
        eval_file=None,
        prompt_field=None,
        response_field=None,
        text_field=None,
        max_seq_len=8,
        hub_name=None,
        hub_config=None,
        hub_train_split=None,
        hub_eval_split=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)

def test_jsonl_train_and_eval_with_prompt_response(monkeypatch):
    # Patch load_dataset for the JSON path only
    def fake_load_dataset_json(name, **kwargs):
        assert name == "json"
        files = kwargs["data_files"]
        out = {}
        for split in files:
            out[split] = FakeDataset([
                {"prompt": "P1", "response": "R1"},
                {"prompt": "P2", "response": "R2"},
            ])
        return FakeDatasetDict(out)

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset_json, raising=True)

    tok = FakeTokenizer()
    cfg = make_cfg(
        train_file="train.jsonl",
        eval_file="eval.jsonl",
        prompt_field="prompt",
        response_field="response",
        max_seq_len=5,
    )

    tokenized = data_mod.load_tokenized_datasets(cfg, tok)
    assert "train" in tokenized and "validation" in tokenized

    # Columns should be removed, only token fields remain
    for _, ds in tokenized.items():
        for row in ds.rows:
            assert set(row.keys()) == {"input_ids", "attention_mask"}
            assert len(row["input_ids"]) <= 5

    # Tokenizer called once per split; texts should contain a newline between prompt/response
    assert len(tok.calls) == 2
    for texts, kwargs in tok.calls:
        assert kwargs["truncation"] is True and kwargs["max_length"] == 5
        assert all("\n" in t for t in texts)


def test_jsonl_only_train_with_text_field(monkeypatch):
    def fake_load_dataset_json(name, **kwargs):
        assert name == "json"
        assert "train" in kwargs["data_files"] and "validation" not in kwargs["data_files"]
        return FakeDatasetDict({
            "train": FakeDataset([
                {"body": "hello"},
                {"body": "world!"},
            ])
        })

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset_json, raising=True)

    tok = FakeTokenizer()
    cfg = make_cfg(train_file="train.jsonl", text_field="body", max_seq_len=10)

    tokenized = data_mod.load_tokenized_datasets(cfg, tok)
    assert "train" in tokenized and "validation" not in tokenized
    texts, kwargs = tok.calls[0]
    assert texts == ["hello", "world!"]
    assert kwargs["truncation"] is True and kwargs["max_length"] == 10
    assert all(set(r.keys()) == {"input_ids", "attention_mask"} for r in tokenized["train"].rows)


def test_hub_gsm8k_builds_QA_text(monkeypatch):
    # For hub path, the code constructs DatasetDict() directly; patch that to our fake.
    monkeypatch.setattr(data_mod, "DatasetDict", FakeDatasetDict, raising=True)

    def fake_load_dataset_hub(name, config=None, split=None, **_):
        assert name == "gsm8k"
        if split == "train":
            return FakeDataset([{"question": "q1", "answer": "a1"}])
        elif split == "test":
            return FakeDataset([{"question": "q2", "answer": "a2"}])
        raise AssertionError("unexpected split")

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset_hub, raising=True)

    tok = FakeTokenizer()
    cfg = make_cfg(
        hub_name="gsm8k",
        hub_config=None,
        hub_train_split="train",
        hub_eval_split="test",
        max_seq_len=50,
    )

    tokenized = data_mod.load_tokenized_datasets(cfg, tok)
    assert "train" in tokenized and "validation" in tokenized

    for texts, _kwargs in tok.calls:
        assert all(t.startswith("Q: ") and "\nA: " in t for t in texts)
    for _, ds in tokenized.items():
        assert all(set(r.keys()) == {"input_ids", "attention_mask"} for r in ds.rows)


def test_generic_hub_without_fields_falls_back_to_str_ex(monkeypatch):
    monkeypatch.setattr(data_mod, "DatasetDict", FakeDatasetDict, raising=True)

    def fake_load_dataset_hub(name, config=None, split=None, **_):
        assert name == "some_dataset"
        assert split == "train"
        return FakeDataset([{"foo": 123}])

    monkeypatch.setattr(data_mod, "load_dataset", fake_load_dataset_hub, raising=True)
    cfg = make_cfg()

    tok = FakeTokenizer(
        hub_name="some_dataset",
        hub_train_split="train",
        max_seq_len=20,
    )

    tokenized = data_mod.load_tokenized_datasets(cfg, tok)
    assert "train" in tokenized and "validation" not in tokenized
    texts, kwargs = tok.calls[0]
    assert texts == [str({"foo": 123})]
    assert kwargs["truncation"] is True and kwargs["max_length"] == 20


def test_raises_if_neither_jsonl_nor_hub():
    tok = FakeTokenizer()
    cfg = make_cfg(train_file="data.csv", hub_name=None)
    with pytest.raises(ValueError) as ei:
        data_mod.load_tokenized_datasets(cfg, tok)
    assert "Provide either JSONL files or a hub dataset" in str(ei.value)