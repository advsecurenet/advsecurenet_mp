# tests/advsecurenet/llm/llm_finetuning/test_cli.py
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner


def _load_cli_with_stubbed_train(run_return: dict | None = None):
    """
    Insert a dummy module at advsecurenet.llm_finetuning.train so that
    'from .train import run_training' inside cli.py binds to our stub.
    Then import the cli module safely.
    """
    dummy_train = types.ModuleType("advsecurenet.llm_finetuning.train")
    dummy_train.run_training = Mock(
        return_value=run_return or {"output_dir": "outputs/run"}
    )

    # Ensure the package path exists in sys.modules to support the dotted name.
    # (Usually already present, but harmless to set defensively.)
    pkg_name = "advsecurenet.llm_finetuning"
    if pkg_name not in sys.modules:
        # Import the package (without pulling cli) so the module path exists.
        importlib.import_module(pkg_name)

    with patch.dict(sys.modules, {"advsecurenet.llm_finetuning.train": dummy_train}):
        cli = importlib.import_module("advsecurenet.llm_finetuning.cli")

    return cli, dummy_train


def test_cli_with_yaml_invokes_training(tmp_path: Path):
    # Create a real file so Click's Path(exists=True) validation passes
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("just: a-file\n")

    # Load CLI with a stubbed train module first
    cli, dummy_train = _load_cli_with_stubbed_train({"output_dir": "outputs/run"})

    # Build a typed Config object using the real config classes
    from advsecurenet.llm_finetuning.config import (
        Config,
        TrainConfig,
        DataConfig,
        PEFTConfig,
    )

    cfg_obj = Config(
        train=TrainConfig(model_name="gpt2", output_dir="outputs/run"),
        data=DataConfig(train_file="data/train.jsonl"),
        peft=PEFTConfig(enabled=False),
    )

    # Patch load_yaml to return our typed config (no YAML parsing needed)
    with patch(
        "advsecurenet.llm_finetuning.cli.load_yaml", return_value=cfg_obj
    ) as load:
        res = CliRunner().invoke(cli.app, ["train", "--config", str(cfg_path)])

        assert res.exit_code == 0, res.output
        assert "Saved to: outputs/run" in res.output
        load.assert_called_once_with(str(cfg_path))
        dummy_train.run_training.assert_called_once()


def test_cli_flags_only_builds_typed_config(tmp_path: Path):
    # Minimal dummy training data so Click's file validation can pass if used
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"text": "hello"}\n')

    cli, dummy_train = _load_cli_with_stubbed_train({"output_dir": "o"})

    res = CliRunner().invoke(
        cli.app,
        [
            "train",
            "--model-name",
            "gpt2",
            "--train-file",
            str(train_file),
            "--output-dir",
            "o",
            "--no-peft",
        ],
    )

    assert res.exit_code == 0, res.output
    # Pull the config object that was passed into run_training(...)
    called_cfg = dummy_train.run_training.call_args[0][0]
    assert called_cfg.train.model_name == "gpt2"
    assert called_cfg.data.train_file == str(train_file)
    assert called_cfg.peft.enabled is False


def test_cli_no_yaml_missing_flags_errors():
    cli, _ = _load_cli_with_stubbed_train()
    res = CliRunner().invoke(cli.app, ["train"])
    assert res.exit_code != 0
    assert "must pass" in res.output.lower()
