# llm_finetune/cli.py
import click
from advsecurenet.llm_finetuning.config import load_yaml, Config
from advsecurenet.llm_finetuning.train import run_training

@click.group()
def app():
    """LLM fine-tuning CLI"""

def _apply_overrides(cfg: Config, **kwargs) -> Config:
    # map flat CLI flags to nested cfg fields (only a few common ones shown)
    overrides = {}
    if kwargs.get("model_name"): overrides.setdefault("train", {})["model_name"] = kwargs["model_name"]
    if kwargs.get("train_file"): overrides.setdefault("data", {})["train_file"] = kwargs["train_file"]
    if kwargs.get("eval_file") is not None: overrides.setdefault("data", {})["eval_file"] = kwargs["eval_file"]
    if kwargs.get("output_dir"): overrides.setdefault("train", {})["output_dir"] = kwargs["output_dir"]
    if kwargs.get("peft") is not None: overrides.setdefault("peft", {})["enabled"] = kwargs["peft"]
    # cheap deep-merge via model_copy:
    return cfg.model_copy(update=overrides)

@app.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="YAML config file.")
@click.option("--model-name", type=str)
@click.option("--train-file", type=str)
@click.option("--eval-file", type=str, default=None)
@click.option("--output-dir", type=str)
@click.option("--peft/--no-peft", "peft", default=None, help="Enable/disable LoRA/QLoRA.")
def train(config_path, **overrides):
    """Fine-tune a causal LM"""
    if config_path:
        cfg = load_yaml(config_path)
    else:
        # minimal inlined config if user skips YAML
        cfg = Config(
            train={"model_name": overrides.get("model_name"), "output_dir": overrides.get("output_dir") or "outputs/sft"},
            data={"train_file": overrides.get("train_file"), "eval_file": overrides.get("eval_file")},
            peft={"enabled": True},
        )
    cfg = _apply_overrides(cfg, **overrides)
    res = run_training(cfg)
    click.echo(f"Saved to: {res['output_dir']}")
