"""Tests for the CLI GCG module."""

import pytest
import subprocess
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from click.testing import CliRunner
from advsecurenet.llm.GCG.experiments.cli_gcg import (
    gcg,
    attack,
    list_configs,
    test,
    quick,
    benchmark,
    evaluate_hf,
)


class TestGCGCLI:
    """Test the main CLI group."""

    def test_gcg_group_exists(self):
        """Test that the main CLI group exists."""
        runner = CliRunner()
        result = runner.invoke(gcg, ["--help"])

        assert result.exit_code == 0
        assert "Universal GCG Attack CLI" in result.output
        assert "Run adversarial attacks on any HuggingFace model" in result.output

    def test_gcg_version(self):
        """Test version option."""
        runner = CliRunner()
        result = runner.invoke(gcg, ["--version"])

        assert result.exit_code == 0
        assert "1.0.0" in result.output


class TestAttackCommand:
    """Test the attack command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.subprocess.run")
    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.Path.cwd")
    def test_attack_basic(self, mock_cwd, mock_subprocess):
        """Test basic attack command execution."""
        # Mock working directory
        mock_gcg_dir = MagicMock()
        mock_config_path = MagicMock()
        mock_config_path.exists.return_value = True
        mock_gcg_dir.__truediv__.return_value = mock_config_path
        mock_cwd.return_value = mock_gcg_dir

        # Mock subprocess
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        result = self.runner.invoke(
            attack, ["--config", "test_config.py", "--model", "gpt2", "--steps", "10"]
        )

        assert result.exit_code == 0
        assert "Attack completed successfully" in result.output

        # Verify subprocess call
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args

        # Check that main.py is called with correct arguments
        cmd = call_args[0][0]
        assert sys.executable in cmd
        assert "main.py" in cmd
        # Check for --config= format (not separate --config argument)
        config_arg = next((arg for arg in cmd if arg.startswith("--config=")), None)
        assert config_arg is not None, f"Expected --config= argument in command: {cmd}"
        assert "--config.verbose=false" in cmd
        assert '--config.model_name="gpt2"' in cmd
        assert "--config.n_steps=10" in cmd

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.subprocess.run")
    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.Path.cwd")
    def test_attack_with_all_options(self, mock_cwd, mock_subprocess):
        """Test attack command with all options."""
        # Mock working directory
        mock_gcg_dir = MagicMock()
        mock_config_path = MagicMock()
        mock_config_path.exists.return_value = True
        mock_gcg_dir.__truediv__.return_value = mock_config_path
        mock_cwd.return_value = mock_gcg_dir

        # Mock subprocess
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        result = self.runner.invoke(
            attack,
            [
                "--config",
                "test_config.py",
                "--model",
                "test_model",
                "--attack-type",
                "individual",
                "--data-type",
                "behaviors",
                "--device",
                "cuda:0",
                "--steps",
                "50",
                "--train-data",
                "25",
                "--batch-size",
                "128",
                "--learning-rate",
                "0.01",
                "--control-init",
                "! ! !",
                "--data-offset",
                "5",
                "--verbose",
                "--config-override",
                "param1=value1",
                "--config-override",
                "param2=value2",
            ],
        )

        assert result.exit_code == 0

        # Verify subprocess call includes all parameters
        cmd = mock_subprocess.call_args[0][0]
        assert "--config.verbose=true" in cmd  # Fixed: lowercase true, not True
        assert '--config.model_name="test_model"' in cmd
        assert '--config.attack_type="individual"' in cmd
        assert '--config.data_type="behaviors"' in cmd
        assert '--config.device="cuda:0"' in cmd
        assert "--config.n_steps=50" in cmd
        assert "--config.n_train_data=25" in cmd
        assert "--config.batch_size=128" in cmd
        assert "--config.lr=0.01" in cmd
        assert '--config.control_init="! ! !"' in cmd
        assert "--config.data_offset=5" in cmd
        assert "--config.param1=value1" in cmd
        assert "--config.param2=value2" in cmd

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.Path.cwd")
    def test_attack_config_not_found(self, mock_cwd):
        """Test attack command when config file not found."""
        # Mock working directory
        mock_gcg_dir = MagicMock()
        mock_config_path = MagicMock()
        mock_config_path.exists.return_value = False
        mock_config_path.__str__.return_value = "/fake/path/nonexistent.py"
        mock_gcg_dir.__truediv__.return_value = mock_config_path
        mock_cwd.return_value = mock_gcg_dir

        # Mock configs directory
        mock_configs_dir = MagicMock()
        mock_configs_dir.exists.return_value = True
        mock_config_file1 = MagicMock()
        mock_config_file1.name = "config1.py"
        mock_config_file2 = MagicMock()
        mock_config_file2.name = "config2.py"
        mock_configs_dir.glob.return_value = [mock_config_file1, mock_config_file2]

        # Setup the correct return value for "configs" subdirectory
        def mock_truediv(path_str):
            if path_str == "configs":
                return mock_configs_dir
            else:
                return mock_config_path

        mock_gcg_dir.__truediv__.side_effect = mock_truediv

        result = self.runner.invoke(attack, ["--config", "nonexistent.py"])

        assert result.exit_code == 1
        assert "Config file not found" in result.output

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.subprocess.run")
    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.Path.cwd")
    def test_attack_subprocess_failure(self, mock_cwd, mock_subprocess):
        """Test attack command when subprocess fails."""
        # Mock working directory
        mock_gcg_dir = MagicMock()
        mock_config_path = MagicMock()
        mock_config_path.exists.return_value = True
        mock_gcg_dir.__truediv__.return_value = mock_config_path
        mock_cwd.return_value = mock_gcg_dir

        # Mock subprocess failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_subprocess.return_value = mock_result

        result = self.runner.invoke(attack, ["--config", "test_config.py"])

        assert result.exit_code == 1
        assert "Attack failed with exit code: 1" in result.output

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.subprocess.run")
    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.Path.cwd")
    def test_attack_keyboard_interrupt(self, mock_cwd, mock_subprocess):
        """Test attack command handling of keyboard interrupt."""
        # Mock working directory
        mock_gcg_dir = MagicMock()
        mock_config_path = MagicMock()
        mock_config_path.exists.return_value = True
        mock_gcg_dir.__truediv__.return_value = mock_config_path
        mock_cwd.return_value = mock_gcg_dir

        # Mock keyboard interrupt
        mock_subprocess.side_effect = KeyboardInterrupt()

        result = self.runner.invoke(attack, ["--config", "test_config.py"])

        assert result.exit_code == 1
        assert "Attack interrupted by user" in result.output

    def test_attack_invalid_config_override(self):
        """Test attack command with invalid config override format."""
        # Create a temporary valid config file for this test
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("# test config")
            temp_config = f.name

        try:
            result = self.runner.invoke(
                attack, ["--config", temp_config, "--config-override", "invalid_format"]
            )

            # Command should show warning about invalid format
            assert "Invalid config override format" in result.output
        finally:
            os.unlink(temp_config)


class TestListConfigsCommand:
    """Test the list-configs command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.Path.cwd")
    def test_list_configs_success(self, mock_cwd):
        """Test successful listing of config files."""
        # Mock working directory
        mock_gcg_dir = MagicMock()
        mock_cwd.return_value = mock_gcg_dir

        # Mock configs directory
        mock_configs_dir = MagicMock()
        mock_configs_dir.exists.return_value = True

        # Create real path objects for proper sorting
        config1_path = Path("config1.py")
        config2_path = Path("config2.py")
        init_path = Path("__init__.py")

        # Mock the actual file paths with proper attributes
        mock_configs_dir.glob.return_value = [config1_path, config2_path, init_path]
        mock_gcg_dir.__truediv__.return_value = mock_configs_dir

        result = self.runner.invoke(list_configs)

        assert result.exit_code == 0
        assert "Available Configuration Files" in result.output
        assert "config1.py" in result.output
        assert "config2.py" in result.output
        assert "__init__.py" not in result.output  # Should be filtered out

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.Path.cwd")
    def test_list_configs_no_directory(self, mock_cwd):
        """Test list-configs when configs directory doesn't exist."""
        # Mock working directory
        mock_gcg_dir = MagicMock()
        mock_cwd.return_value = mock_gcg_dir

        # Mock configs directory not existing
        mock_configs_dir = MagicMock()
        mock_configs_dir.exists.return_value = False
        mock_gcg_dir.__truediv__.return_value = mock_configs_dir

        result = self.runner.invoke(list_configs)

        assert result.exit_code == 0
        assert "Configs directory not found" in result.output


class TestTestCommand:
    """Test the test command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.Path.cwd")
    def test_test_command_success(self, mock_cwd):
        """Test successful environment test."""
        # Mock working directory
        mock_gcg_dir = MagicMock()
        mock_cwd.return_value = mock_gcg_dir

        # Mock all required paths existing
        def mock_path_div(path_str):
            mock_path = MagicMock()
            if path_str in [
                "main.py",
                "configs/universal_config.py",
                "data/advbench/harmful_behaviors.csv",
            ]:
                mock_path.is_file.return_value = True
                mock_path.is_dir.return_value = False
            elif path_str in ["configs", "data", "data/advbench"]:
                mock_path.is_file.return_value = False
                mock_path.is_dir.return_value = True
            else:
                mock_path.is_file.return_value = False
                mock_path.is_dir.return_value = False
            return mock_path

        mock_gcg_dir.__truediv__.side_effect = mock_path_div

        with patch("torch.__version__", "1.9.0"):
            with patch("transformers.__version__", "4.20.0"):
                result = self.runner.invoke(test)

        assert result.exit_code == 0
        assert "Environment is properly configured" in result.output
        assert "[OK] main.py (file)" in result.output
        assert "[OK] PyTorch: 1.9.0" in result.output
        assert "[OK] Transformers: 4.20.0" in result.output

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.Path.cwd")
    def test_test_command_missing_files(self, mock_cwd):
        """Test environment test with missing files."""
        # Mock working directory
        mock_gcg_dir = MagicMock()
        mock_cwd.return_value = mock_gcg_dir

        # Mock missing files
        def mock_path_div(path_str):
            mock_path = MagicMock()
            mock_path.is_file.return_value = False
            mock_path.is_dir.return_value = False
            return mock_path

        mock_gcg_dir.__truediv__.side_effect = mock_path_div

        result = self.runner.invoke(test)

        assert result.exit_code == 1
        assert "Environment needs fixing" in result.output
        assert "[MISSING]" in result.output

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.Path.cwd")
    def test_test_command_missing_dependencies(self, mock_cwd):
        """Test environment test with missing Python dependencies."""
        # Mock working directory and files
        mock_gcg_dir = MagicMock()
        mock_cwd.return_value = mock_gcg_dir

        def mock_path_div(path_str):
            mock_path = MagicMock()
            mock_path.is_file.return_value = True  # All files exist
            mock_path.is_dir.return_value = True  # All dirs exist
            return mock_path

        mock_gcg_dir.__truediv__.side_effect = mock_path_div

        # Use patch.dict to make torch and transformers unavailable
        # Then patch the import to raise ImportError for these modules
        import builtins

        original_import = builtins.__import__

        def mock_failing_import(name, *args, **kwargs):
            if name in ["torch", "transformers"]:
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        # Remove modules from sys.modules and mock __import__
        with patch.dict(
            "sys.modules", {"torch": None, "transformers": None}, clear=False
        ):
            with patch("builtins.__import__", side_effect=mock_failing_import):
                result = self.runner.invoke(test)

        # Should show missing dependencies and exit with code 1
        assert result.exit_code == 1
        assert "[MISSING] PyTorch not installed" in result.output
        assert "[MISSING] Transformers not installed" in result.output


class TestQuickCommand:
    """Test the quick command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.click.Context")
    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.attack")
    def test_quick_command(self, mock_attack, mock_context):
        """Test quick command execution."""
        mock_ctx = MagicMock()
        mock_context.return_value = mock_ctx

        result = self.runner.invoke(quick)

        assert result.exit_code == 0
        assert "Running Quick Test Attack (10 steps)" in result.output

        # Verify that attack is called with correct parameters
        mock_ctx.invoke.assert_called_once_with(
            mock_attack,
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


class TestBenchmarkCommand:
    """Test the benchmark command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.click.Context")
    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.attack")
    def test_benchmark_default_models(self, mock_attack, mock_context):
        """Test benchmark with default models."""
        mock_ctx = MagicMock()
        mock_context.return_value = mock_ctx

        # Mock successful attack calls
        mock_ctx.invoke.return_value = None  # No exception means success

        result = self.runner.invoke(benchmark)

        assert result.exit_code == 0
        assert "Benchmarking 3 models" in result.output
        assert "gpt2" in result.output
        assert "t5-small" in result.output
        assert "distilgpt2" in result.output
        assert "Success Rate:" in result.output

        # Should call attack 3 times
        assert mock_ctx.invoke.call_count == 3

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.click.Context")
    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.attack")
    def test_benchmark_custom_models(self, mock_attack, mock_context):
        """Test benchmark with custom models."""
        mock_ctx = MagicMock()
        mock_context.return_value = mock_ctx

        result = self.runner.invoke(benchmark, ["model1", "model2"])

        assert result.exit_code == 0
        assert "Benchmarking 2 models" in result.output
        assert "model1" in result.output
        assert "model2" in result.output

        # Should call attack 2 times
        assert mock_ctx.invoke.call_count == 2

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.click.Context")
    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.attack")
    def test_benchmark_with_failures(self, mock_attack, mock_context):
        """Test benchmark with some model failures."""
        mock_ctx = MagicMock()
        mock_context.return_value = mock_ctx

        # Mock first call success, second call failure
        mock_ctx.invoke.side_effect = [None, Exception("Model failed"), None]

        result = self.runner.invoke(benchmark)

        assert result.exit_code == 0
        assert "[OK]" in result.output  # Should have some successes
        assert "[FAILED]" in result.output  # Should have some failures


class TestEvaluateHFCommand:
    """Test the evaluate-hf command."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.subprocess.run")
    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.Path.cwd")
    def test_evaluate_hf_with_results_file(self, mock_cwd, mock_subprocess):
        """Test evaluate-hf with explicit results file."""
        # Mock working directory
        mock_gcg_dir = MagicMock()
        mock_cwd.return_value = mock_gcg_dir

        # Mock subprocess success
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        # Create a real temporary results file (click.Path(exists=True) requires real file)
        import json

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"test_results": [{"prompt": "test", "response": "response"}]}, f)
            temp_file = f.name

        try:
            result = self.runner.invoke(
                evaluate_hf,
                [
                    "--results-file",
                    temp_file,
                    "--model",
                    "test_model",
                    "--temperature",
                    "0.8",
                ],
            )

            assert result.exit_code == 0
            assert "Evaluation completed" in result.output

            # Verify subprocess call
            mock_subprocess.assert_called_once()
            call_args = mock_subprocess.call_args[0][0]
            assert "eval_scripts/evaluate_hf_attack.py" in call_args
            assert "--results-file" in call_args
            assert temp_file in call_args
            assert "--model" in call_args
        finally:
            # Clean up temporary file
            import os

            if os.path.exists(temp_file):
                os.unlink(temp_file)
        assert "test_model" in call_args
        assert "--temperature" in call_args
        assert "0.8" in call_args

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.subprocess.run")
    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.Path.cwd")
    def test_evaluate_hf_auto_find_results(self, mock_cwd, mock_subprocess):
        """Test evaluate-hf with auto-finding latest results file."""
        # Mock working directory
        mock_gcg_dir = MagicMock()
        mock_cwd.return_value = mock_gcg_dir

        # Mock results directory
        mock_results_dir = MagicMock()
        mock_results_dir.exists.return_value = True

        # Mock JSON files with different timestamps
        mock_file1 = MagicMock()
        mock_file1.stat.return_value.st_mtime = 1000
        mock_file1.name = "old_results.json"

        mock_file2 = MagicMock()
        mock_file2.stat.return_value.st_mtime = 2000
        mock_file2.name = "latest_results.json"

        mock_results_dir.glob.return_value = [mock_file1, mock_file2]
        mock_gcg_dir.__truediv__.return_value = mock_results_dir

        # Mock subprocess success
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_subprocess.return_value = mock_result

        result = self.runner.invoke(evaluate_hf)

        assert result.exit_code == 0
        assert "Using latest results file: latest_results.json" in result.output

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.Path.cwd")
    def test_evaluate_hf_no_results_found(self, mock_cwd):
        """Test evaluate-hf when no results files found."""
        # Mock working directory
        mock_gcg_dir = MagicMock()
        mock_cwd.return_value = mock_gcg_dir

        # Mock results directory with no files
        mock_results_dir = MagicMock()
        mock_results_dir.exists.return_value = True
        mock_results_dir.glob.return_value = []
        mock_gcg_dir.__truediv__.return_value = mock_results_dir

        result = self.runner.invoke(evaluate_hf)

        assert result.exit_code == 1
        assert "No results files found" in result.output

    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.subprocess.run")
    @patch("advsecurenet.llm.GCG.experiments.cli_gcg.Path.cwd")
    def test_evaluate_hf_subprocess_failure(self, mock_cwd, mock_subprocess):
        """Test evaluate-hf when subprocess fails."""
        # Mock working directory and results
        mock_gcg_dir = MagicMock()
        mock_cwd.return_value = mock_gcg_dir

        mock_results_dir = MagicMock()
        mock_results_dir.exists.return_value = True
        mock_file = MagicMock()
        mock_file.name = "test.json"
        mock_results_dir.glob.return_value = [mock_file]
        mock_gcg_dir.__truediv__.return_value = mock_results_dir

        # Mock subprocess failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_subprocess.return_value = mock_result

        result = self.runner.invoke(evaluate_hf)

        assert result.exit_code == 1


class TestIntegration:
    """Integration tests for CLI components."""

    def test_command_registration(self):
        """Test that all commands are properly registered."""
        runner = CliRunner()
        result = runner.invoke(gcg, ["--help"])

        assert result.exit_code == 0

        # Check that all commands are listed
        assert "attack" in result.output
        assert "list-configs" in result.output
        assert "test" in result.output
        assert "quick" in result.output
        assert "benchmark" in result.output
        assert "evaluate-hf" in result.output

    def test_parameter_validation(self):
        """Test parameter validation for commands."""
        runner = CliRunner()

        # Test invalid attack type
        result = runner.invoke(attack, ["--attack-type", "invalid"])
        assert result.exit_code != 0  # Should fail validation

        # Test invalid data type
        result = runner.invoke(attack, ["--data-type", "invalid"])
        assert result.exit_code != 0  # Should fail validation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
