from click.testing import CliRunner
from unittest.mock import patch
from pathlib import Path
from advsecurenet.llm_finetuning.cli import app
from advsecurenet.llm_finetuning import cli as cli_mod  # to build typed Configs


def _write(p: Path, s: str):
    p.write_text(s.strip() + "\n")


def test_cli_with_yaml_invokes_training(tmp_path):
    # create a real file so Click's Path(exists=True) passes
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("just: a-file\n")

    # Build a fully-typed Config object that load_yaml will return
    cfg_obj = cli_mod.Config(
        train=cli_mod.TrainConfig(model_name="gpt2", output_dir="outputs/run"),
        data=cli_mod.DataConfig(train_file="data/train.jsonl"),
        peft=cli_mod.PEFTConfig(enabled=False),
    )

    with patch("advsecurenet.llm_finetuning.cli.load_yaml", return_value=cfg_obj) as load, \
         patch("advsecurenet.llm_finetuning.cli.run_training") as run:
        run.return_value = {"output_dir": "outputs/run"}

        r = CliRunner().invoke(app, ["train", "--config", str(cfg_path)])

        assert r.exit_code == 0, r.output
        assert "Saved to: outputs/run" in r.output
        load.assert_called_once_with(str(cfg_path))
        run.assert_called_once()


def test_cli_flags_only_builds_typed_config(tmp_path):
    train_file = tmp_path / "train.jsonl"
    train_file.write_text('{"text": "hello"}\n')

    with patch("advsecurenet.llm_finetuning.cli.run_training") as run:
        run.return_value = {"output_dir": "o"}
        r = CliRunner().invoke(
            app,
            ["train",
             "--model-name","gpt2",
             "--train-file", str(train_file),
             "--output-dir","o",
             "--no-peft"]
        )
        assert r.exit_code == 0, r.output
        # ensure overrides reached the config
        called_cfg = run.call_args[0][0]
        assert called_cfg.train.model_name == "gpt2"
        assert called_cfg.data.train_file == str(train_file)
        assert called_cfg.peft.enabled is False

def test_cli_no_yaml_missing_flags_errors():
    r = CliRunner().invoke(app, ["train"])
    assert r.exit_code != 0
    assert "must pass" in r.output.lower()
