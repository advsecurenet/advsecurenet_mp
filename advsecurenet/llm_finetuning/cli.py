import click

from advsecurenet.llm_finetuning.config import (
    load_yaml,
    Config,
    TrainConfig,
    DataConfig,
    PEFTConfig,
)
from advsecurenet.llm_finetuning.train import run_training


@click.group()
def app():
    """LLM fine-tuning CLI"""


def _apply_overrides(cfg: Config, **kw) -> Config:
    """
    Safely apply flat CLI flags onto nested Pydantic models
    without downgrading them to dicts.
    """
    train_upd, data_upd, peft_upd = {}, {}, {}

    if kw.get("model_name"):
        train_upd["model_name"] = kw["model_name"]
    if kw.get("output_dir"):
        train_upd["output_dir"] = kw["output_dir"]

    if kw.get("train_file"):
        data_upd["train_file"] = kw["train_file"]
    if kw.get("eval_file") is not None:
        data_upd["eval_file"] = kw["eval_file"]

    if kw.get("peft") is not None:
        peft_upd["enabled"] = kw["peft"]

    # Apply updates while preserving model types
    if train_upd:
        cfg = cfg.model_copy(update={"train": cfg.train.model_copy(update=train_upd)})
    if data_upd:
        cfg = cfg.model_copy(update={"data": cfg.data.model_copy(update=data_upd)})
    if peft_upd:
        cfg = cfg.model_copy(update={"peft": cfg.peft.model_copy(update=peft_upd)})

    return cfg


@app.command()
@click.option("--config", "config_path", type=click.Path(exists=True), help="Path to YAML config.")
@click.option("--model-name", type=str, help="HF model id or local path (flags-only mode).")
@click.option("--train-file", type=str, help="Path to JSONL train file (flags-only mode).")
@click.option("--eval-file", type=str, default=None, help="Optional JSONL eval file (flags-only mode).")
@click.option("--output-dir", type=str, help="Where to save outputs/checkpoints.")
@click.option("--peft/--no-peft", "peft", default=None, help="Enable/disable LoRA/QLoRA.")
def train(config_path, **overrides):
    """Fine-tune a causal LM."""
    if config_path:
        # Load a fully-typed Config from YAML
        cfg = load_yaml(config_path)
    else:
        # Flags-only: require bare minimum
        missing = []
        if not overrides.get("model_name"):
            missing.append("--model-name")
        if not overrides.get("train_file"):
            missing.append("--train-file")
        if missing:
            raise click.UsageError(
                "When --config is not provided, you must pass: " + ", ".join(missing)
            )

        # Build typed nested configs (NOT dicts)
        cfg = Config(
            train=TrainConfig(
                model_name=overrides["model_name"],
                output_dir=overrides.get("output_dir") or "outputs/sft",
            ),
            data=DataConfig(
                train_file=overrides["train_file"],
                eval_file=overrides.get("eval_file"),
            ),
            peft=PEFTConfig(enabled=True),
        )

    # Apply CLI overrides safely (preserve submodel types)
    cfg = _apply_overrides(cfg, **overrides)

    # Optional sanity print (uncomment if you want to verify types once)
    # print("train type:", type(cfg.train), "| data type:", type(cfg.data), "| peft type:", type(cfg.peft))

    res = run_training(cfg)
    click.echo(f"Saved to: {res['output_dir']}")


if __name__ == "__main__":
    app()