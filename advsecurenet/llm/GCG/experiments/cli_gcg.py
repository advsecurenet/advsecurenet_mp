import click
import subprocess
import sys
import os
from pathlib import Path


@click.group()
@click.version_option(version="1.0.0")
def gcg():
    """Universal GCG Attack CLI - Run adversarial attacks on any HuggingFace model."""
    pass


@gcg.command()
@click.option(
    "--config",
    "-c",
    default="configs/universal_config.py",
    help="Config file to use (default: universal_config.py)",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="HuggingFace model name or path (overrides config)",
)
@click.option(
    "--attack-type",
    default=None,
    type=click.Choice(["individual", "transfer"]),
    help="Type of attack to run (overrides config)",
)
@click.option(
    "--data-type",
    default=None,
    type=click.Choice(["behaviors", "strings"]),
    help="Data type for attack (overrides config)",
)
@click.option("--device", default=None, help="Device to use (overrides config)")
@click.option(
    "--steps",
    "-s",
    default=None,
    help="Number of optimization steps (overrides config)",
)
@click.option(
    "--train-data", default=None, help="Number of training examples (overrides config)"
)
@click.option(
    "--batch-size",
    "-b",
    default=None,
    help="Batch size for optimization (overrides config)",
)
@click.option(
    "--learning-rate", "-lr", default=None, help="Learning rate (overrides config)"
)
@click.option(
    "--control-init", default=None, help="Initial control string (overrides config)"
)
@click.option(
    "--data-offset", default=None, help="Data offset for experiments (overrides config)"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--config-override",
    multiple=True,
    help="Override config parameters (format: key=value)",
)
def attack(
    config,
    model,
    attack_type,
    data_type,
    device,
    steps,
    train_data,
    batch_size,
    learning_rate,
    control_init,
    data_offset,
    verbose,
    config_override,
):
    """Run universal GCG attack on specified model."""

    # Get the current directory (should be experiments directory)
    gcg_dir = Path.cwd()

    # Verify config file exists - handle both relative and absolute paths
    if not config.startswith("/"):  # It's a relative path
        config_path = gcg_dir / config
    else:
        config_path = Path(config)

    if not config_path.exists():
        click.echo(f"❌ Config file not found: {config_path}", err=True)

        # List available configs
        configs_dir = gcg_dir / "configs"
        if configs_dir.exists():
            available_configs = [
                f.name for f in configs_dir.glob("*.py") if f.name != "__init__.py"
            ]
            click.echo(f"📁 Available configs: {', '.join(available_configs)}")

        sys.exit(1)

    click.echo(f"🚀 Running Universal GCG Attack")
    click.echo(f"================================")
    click.echo(f"Config: {config_path}")
    if model:
        click.echo(f"Model Override: {model}")
    if attack_type:
        click.echo(f"Attack Type Override: {attack_type}")
    if data_type:
        click.echo(f"Data Type Override: {data_type}")
    if device:
        click.echo(f"Device Override: {device}")
    if steps:
        click.echo(f"Steps Override: {steps}")
    if train_data:
        click.echo(f"Training Data Override: {train_data}")
    if batch_size:
        click.echo(f"Batch Size Override: {batch_size}")
    if learning_rate:
        click.echo(f"Learning Rate Override: {learning_rate}")
    click.echo(f"Working Directory: {gcg_dir}")
    click.echo(f"================================")

    # Build command - MATCH THE EXACT SHELL SCRIPT SYNTAX
    cmd = [
        sys.executable,
        "main.py",
        "--config",
        str(config_path),
        f"--config.verbose={verbose}",
    ]

    # Only add overrides if values are provided - USE THE EXACT SHELL SCRIPT SYNTAX
    if model:
        cmd.extend(
            [
                f'--config.model_name="{model}"',  # FIXED: Add quotes like shell script
                f"--config.model_paths=\"('{model}',)\"",  # FIXED: Exact shell script syntax
                f"--config.tokenizer_paths=\"('{model}',)\"",  # FIXED: Exact shell script syntax
            ]
        )

    if device:
        cmd.append(f'--config.device="{device}"')  # FIXED: Add quotes
    if attack_type:
        cmd.append(f'--config.attack_type="{attack_type}"')  # FIXED: Add quotes
    if data_type:
        cmd.append(f'--config.data_type="{data_type}"')  # FIXED: Add quotes
    if steps:
        cmd.append(f"--config.n_steps={steps}")
    if train_data:
        cmd.append(f"--config.n_train_data={train_data}")
    if batch_size:
        cmd.append(f"--config.batch_size={batch_size}")
    if learning_rate:
        cmd.append(f"--config.lr={learning_rate}")
    if control_init:
        cmd.append(f'--config.control_init="{control_init}"')  # FIXED: Add quotes
    if data_offset is not None:
        cmd.append(f"--config.data_offset={data_offset}")

    # Add any additional config overrides
    for override in config_override:
        if "=" in override:
            key, value = override.split("=", 1)
            cmd.append(f"--config.{key}={value}")
        else:
            click.echo(
                f"⚠️  Invalid config override format: {override} (use key=value)",
                err=True,
            )

    # Run the attack
    try:
        if verbose:
            click.echo(f"🔧 Executing: {' '.join(cmd)}")

        # Use subprocess.run without capture_output to show real-time progress
        result = subprocess.run(cmd, cwd=gcg_dir, text=True)

        if result.returncode == 0:
            click.echo("✅ Attack completed successfully!")
        else:
            click.echo(
                f"❌ Attack failed with exit code: {result.returncode}", err=True
            )
            sys.exit(result.returncode)

    except KeyboardInterrupt:
        click.echo("\n🛑 Attack interrupted by user")
        sys.exit(1)
    except FileNotFoundError:
        click.echo(f"❌ Python executable not found: {sys.executable}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"❌ Unexpected error: {e}", err=True)
        sys.exit(1)


@gcg.command()
def list_configs():
    """List available configuration files."""

    configs_dir = Path.cwd() / "configs"

    if not configs_dir.exists():
        click.echo("❌ Configs directory not found")
        return

    click.echo("📁 Available Configuration Files:")
    click.echo("=" * 35)

    config_files = sorted(
        [f for f in configs_dir.glob("*.py") if f.name != "__init__.py"]
    )

    for config_file in config_files:
        click.echo(f"• {config_file.name}")

    click.echo(f"\n💡 Usage: python3 cli_gcg.py attack --config configs/my_config.py")


@gcg.command()
def test():
    """Test if GCG environment is properly set up."""

    gcg_dir = Path.cwd()

    click.echo("🧪 Testing GCG Environment")
    click.echo("=" * 30)

    # Check directory structure
    required_items = [
        ("main.py", "file"),
        ("configs", "directory"),
        ("configs/universal_config.py", "file"),
        ("data", "directory"),
        ("data/advbench", "directory"),
        ("data/advbench/harmful_behaviors.csv", "file"),
    ]

    all_good = True
    for item, item_type in required_items:
        path = gcg_dir / item
        if item_type == "file" and path.is_file():
            click.echo(f"✅ {item} (file)")
        elif item_type == "directory" and path.is_dir():
            click.echo(f"✅ {item} (directory)")
        else:
            click.echo(f"❌ {item} (missing {item_type})")
            all_good = False

    # Check Python environment
    try:
        import torch

        click.echo(f"✅ PyTorch: {torch.__version__}")
    except ImportError:
        click.echo("❌ PyTorch not installed")
        all_good = False

    try:
        import transformers

        click.echo(f"✅ Transformers: {transformers.__version__}")
    except ImportError:
        click.echo("❌ Transformers not installed")
        all_good = False

    if all_good:
        click.echo("\n🎉 Environment is properly configured!")
        click.echo("You can now run: python3 cli_gcg.py attack --model gpt2 --steps 10")
    else:
        click.echo("\n❌ Environment needs fixing before running attacks")
        sys.exit(1)


@gcg.command()
def quick():
    """Run a quick test attack with minimal settings."""

    click.echo("🚀 Running Quick Test Attack (10 steps)")

    ctx = click.Context(attack)
    ctx.invoke(
        attack,
        config="configs/universal_config.py",
        model="gpt2",
        attack_type="individual",
        data_type="behaviors",
        device="auto",
        steps=10,
        train_data=1,
        batch_size=8,
        learning_rate=0.01,
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        data_offset=0,
        verbose=True,
        config_override=(),
    )


@gcg.command()
@click.argument("models", nargs=-1)
def benchmark(models):
    """Benchmark GCG attack across multiple models."""

    if not models:
        models = ["gpt2", "t5-small", "distilgpt2"]

    click.echo(f"🏁 Benchmarking {len(models)} models...")

    results = []
    for i, model in enumerate(models, 1):
        click.echo(f"\n[{i}/{len(models)}] Testing {model}...")

        try:
            ctx = click.Context(attack)
            ctx.invoke(
                attack,
                config="configs/universal_config.py",
                model=model,
                attack_type="individual",
                data_type="behaviors",
                device="auto",
                steps=20,
                train_data=1,
                batch_size=8,
                learning_rate=0.01,
                control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
                data_offset=0,
                verbose=False,
                config_override=(),
            )
            results.append((model, "✅"))

        except Exception as e:
            click.echo(f"❌ {model} failed: {e}")
            results.append((model, "❌"))

    # Summary
    click.echo("\n📊 Benchmark Results:")
    click.echo("=" * 40)
    for model, status in results:
        click.echo(f"{status} {model}")

    success_rate = len([r for r in results if r[1] == "✅"]) / len(results)
    click.echo(f"\n🎯 Success Rate: {success_rate:.1%}")


@gcg.command()
@click.option(
    "--results-file",
    "-r",
    type=click.Path(exists=True),
    help="Path to GCG results JSON file (default: latest)",
)
@click.option("--model", "-m", help="Override model name (default: use from results)")
@click.option("--device", default="auto", help="Device to use (auto, cpu, cuda:0)")
@click.option(
    "--temperature", "-t", type=float, default=0.7, help="Generation temperature"
)
@click.option(
    "--num-samples", "-n", type=int, default=3, help="Number of response samples"
)
@click.option("--save-results", "-s", is_flag=True, help="Save evaluation results")
@click.option("--verbose", "-v", is_flag=True, help="Show all responses")
def evaluate_hf(
    results_file, model, device, temperature, num_samples, save_results, verbose
):
    """Evaluate GCG attack using HuggingFace model inference."""

    # Auto-find latest results file if not specified
    if not results_file:
        results_dir = Path.cwd() / "results"
        if results_dir.exists():
            json_files = list(results_dir.glob("*.json"))
            if json_files:
                results_file = max(json_files, key=lambda x: x.stat().st_mtime)
                click.echo(f"📁 Using latest results file: {results_file.name}")
            else:
                click.echo("❌ No results files found in results/ directory", err=True)
                sys.exit(1)
        else:
            click.echo("❌ Results directory not found", err=True)
            sys.exit(1)

    # Build evaluation command
    cmd = [
        sys.executable,
        "eval_scripts/evaluate_hf_attack.py",
        "--results-file",
        str(results_file),
        "--device",
        device,
        "--temperature",
        str(temperature),
        "--num-samples",
        str(num_samples),
    ]

    if model:
        cmd.extend(["--model", model])
    if save_results:
        cmd.append("--save-results")
    if verbose:
        cmd.append("--verbose")

    try:
        result = subprocess.run(cmd, cwd=Path.cwd(), text=True)
        if result.returncode == 0:
            click.echo("✅ Evaluation completed!")
        else:
            sys.exit(result.returncode)
    except Exception as e:
        click.echo(f"❌ Evaluation failed: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    gcg()
