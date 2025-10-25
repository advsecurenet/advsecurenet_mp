# tests/test_config.py
import textwrap
import pytest

from advsecurenet.llm_finetuning.config import (
    Config,
    TrainConfig,
    load_yaml,
    merge,
)


def test_yaml_loads_and_defaults(tmp_path):
    yml = tmp_path / "ok.yaml"
    yml.write_text(
        textwrap.dedent(
            """
    train:
      model_name: gpt2
      output_dir: outputs/test
    data:
      train_file: data/train.jsonl
    peft:
      enabled: false
    """
        )
    )
    cfg = load_yaml(str(yml))

    # structure + values
    assert isinstance(cfg, Config)
    assert cfg.train.model_name == "gpt2"
    assert cfg.train.output_dir == "outputs/test"
    assert cfg.data.train_file == "data/train.jsonl"

    # defaults exist
    assert cfg.train.seed == 42
    assert cfg.peft.quantization == "qlora"
    assert cfg.peft.target_modules == ["q_proj", "v_proj"]


def test_one_of_validation_error(tmp_path):
    # no data.train_file and no data.hub_name -> should raise
    yml = tmp_path / "bad.yaml"
    yml.write_text(
        textwrap.dedent(
            """
    train:
      model_name: gpt2
    data: {}
    """
        )
    )
    with pytest.raises(Exception):
        load_yaml(str(yml))


def test_hub_fields_load_with_current_keys(tmp_path):
    # use the actual field names supported by DataConfig (no legacy aliasing)
    yml = tmp_path / "hub.yaml"
    yml.write_text(
        textwrap.dedent(
            """
    train:
      model_name: gpt2
      output_dir: out
    data:
      hub_name: wikitext
      hub_config: wikitext-2-raw-v1
      hub_train_split: train[:10]
      hub_eval_split: validation[:5]
      max_seq_len: 64
    """
        )
    )
    cfg = load_yaml(str(yml))
    assert cfg.data.hub_name == "wikitext"
    assert cfg.data.hub_config == "wikitext-2-raw-v1"
    assert cfg.data.hub_train_split == "train[:10]"
    assert cfg.data.hub_eval_split == "validation[:5]"
    assert cfg.data.max_seq_len == 64


def test_hub_only_satisfies_one_of_rule(tmp_path):
    # Providing just hub_name (and no train_file) is valid
    yml = tmp_path / "hub_only.yaml"
    yml.write_text(
        textwrap.dedent(
            """
    train:
      model_name: gpt2
    data:
      hub_name: gsm8k
    """
        )
    )
    cfg = load_yaml(str(yml))
    assert cfg.data.hub_name == "gsm8k"
    assert cfg.data.train_file is None


def test_merge_ignores_top_level_nones_and_replaces_with_model_instance(tmp_path):
    # Base config
    yml = tmp_path / "base.yaml"
    yml.write_text(
        textwrap.dedent(
            """
    train:
      model_name: gpt2
      output_dir: outputs/base
    data:
      train_file: data/base.jsonl
    peft:
      enabled: true
      r: 8
    """
        )
    )
    base = load_yaml(str(yml))

    # Prepare a *model instance* for replacement (since merge is shallow)
    new_train = base.train.model_copy(update={"output_dir": "outputs/merged"})

    # Top-level None entries should be ignored; train replaced by our instance
    merged = merge(
        base,
        {
            "train": new_train,
            "peft": None,
            "data": None,
        },
    )

    assert isinstance(merged, Config)
    assert isinstance(merged.train, TrainConfig)
    assert merged.train.output_dir == "outputs/merged"
    # unchanged
    assert merged.peft.enabled is True
    assert merged.data.train_file == "data/base.jsonl"


def test_merge_noop_when_all_nones(tmp_path):
    yml = tmp_path / "base2.yaml"
    yml.write_text(
        textwrap.dedent(
            """
    train:
      model_name: gpt2
      output_dir: outputs/base2
    data:
      train_file: data/base2.jsonl
    """
        )
    )
    base = load_yaml(str(yml))

    merged = merge(base, {"train": None, "data": None, "peft": None})
    # identical object contents
    assert merged.train.output_dir == base.train.output_dir
    assert merged.data.train_file == base.data.train_file
    assert merged.peft.enabled == base.peft.enabled


def test_peft_default_factory_independent_lists(tmp_path):
    # Ensure target_modules default is a fresh list per instance (no shared reference)
    yml1 = tmp_path / "c1.yaml"
    yml1.write_text(
        textwrap.dedent(
            """
    train:
      model_name: gpt2
    data:
      train_file: data/a.jsonl
    """
        )
    )
    yml2 = tmp_path / "c2.yaml"
    yml2.write_text(
        textwrap.dedent(
            """
    train:
      model_name: gpt2
    data:
      train_file: data/b.jsonl
    """
        )
    )

    c1 = load_yaml(str(yml1))
    c2 = load_yaml(str(yml2))

    assert c1.peft.target_modules == ["q_proj", "v_proj"]
    assert c2.peft.target_modules == ["q_proj", "v_proj"]

    # mutate one; the other remains unchanged
    c1.peft.target_modules.append("k_proj")
    assert c1.peft.target_modules == ["q_proj", "v_proj", "k_proj"]
    assert c2.peft.target_modules == ["q_proj", "v_proj"]
