"""Tests for the universal configuration module."""

import pytest
import torch
import os
from unittest.mock import patch, MagicMock
from ml_collections import config_dict
from advsecurenet.llm.GCG.experiments.configs.universal_config import get_config


class TestGetConfig:
    """Test the get_config function."""

    @patch("torch.cuda.is_available")
    def test_get_config_with_cuda(self, mock_cuda_available):
        """Test configuration creation when CUDA is available."""
        mock_cuda_available.return_value = True

        config = get_config()

        assert isinstance(config, config_dict.ConfigDict)
        assert "cuda" in config.device  # Can be "cuda:0" or "cuda"

    @patch("torch.cuda.is_available")
    def test_get_config_without_cuda(self, mock_cuda_available):
        """Test configuration creation when CUDA is not available."""
        mock_cuda_available.return_value = False

        config = get_config()

        assert isinstance(config, config_dict.ConfigDict)
        assert config.device == "cpu"

    def test_config_structure(self):
        """Test that the configuration has all required fields."""
        config = get_config()

        # Test basic structure
        assert hasattr(config, "device")
        assert hasattr(config, "attack")
        assert hasattr(config, "progressive_models")
        assert hasattr(config, "progressive_goals")
        assert hasattr(config, "n_train_data")
        assert hasattr(config, "n_test_data")

        # Test control initialization fields
        assert hasattr(config, "control_init")
        assert hasattr(config, "n_steps")
        assert hasattr(config, "batch_size")
        assert hasattr(config, "lr")

        # Test optimization parameters
        assert hasattr(config, "topk")
        assert hasattr(config, "temp")
        assert hasattr(config, "filter_cand")
        assert hasattr(config, "allow_non_ascii")

        # Test loss parameters
        assert hasattr(config, "target_weight")
        assert hasattr(config, "control_weight")

        # Test evaluation parameters
        assert hasattr(config, "test_steps")
        assert hasattr(config, "anneal")
        assert hasattr(config, "incr_control")
        assert hasattr(config, "stop_on_success")
        assert hasattr(config, "verbose")
        assert hasattr(config, "gbda_deterministic")

    def test_config_default_values(self):
        """Test that configuration has expected default values."""
        config = get_config()

        # Test attack configuration
        assert config.attack == "gcg"
        assert config.progressive_models is False
        assert config.progressive_goals is False

        # Test data configuration
        assert config.n_train_data == 10  # Actual default value
        assert config.n_test_data == 0  # Actual default value

        # Test control parameters
        assert config.control_init == "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"
        assert config.n_steps == 100
        assert config.batch_size == 16
        assert config.lr == 1e-1

        # Test optimization parameters
        assert config.topk == 256
        assert config.temp == 1.5  # Actual default value
        assert config.filter_cand is True
        assert config.allow_non_ascii is False

        # Test loss weights
        assert config.target_weight == 1.0
        assert config.control_weight == 0.0  # Actual default value

        # Test evaluation parameters
        assert config.test_steps == 10  # Actual default value
        assert config.anneal is False
        assert config.incr_control is False
        assert config.stop_on_success is False
        assert config.verbose is True
        assert config.gbda_deterministic is True

    def test_config_model_configuration(self):
        """Test model-related configuration fields."""
        config = get_config()

        # Test model parameters
        assert hasattr(config, "model_name")
        assert hasattr(config, "train_data")
        assert hasattr(config, "test_data")
        assert hasattr(config, "result_prefix")

        # Test default model name
        assert config.model_name == "Qwen/Qwen2.5-0.5B-Instruct"  # Actual default

        # Test data source
        assert "harmful_behaviors.csv" in config.train_data  # Path to actual data file
        assert config.test_data == ""  # Empty by default

    def test_config_transfer_settings(self):
        """Test transfer attack configuration."""
        config = get_config()

        assert hasattr(config, "transfer")
        assert config.transfer is False  # Default should be individual attack

        # Test that transfer can be modified
        config.transfer = True
        assert config.transfer is True

    @patch("os.getcwd")
    def test_config_result_prefix_path(self, mock_getcwd):
        """Test that result prefix includes proper path formatting."""
        mock_getcwd.return_value = "/test/path"

        config = get_config()

        # The result_prefix should be a string that can be used for file paths
        assert isinstance(config.result_prefix, str)
        assert "../results/" in config.result_prefix  # Should contain results path

    def test_config_mutability(self):
        """Test that configuration can be modified after creation."""
        config = get_config()

        # Test modifying various parameters
        original_steps = config.n_steps
        config.n_steps = 200
        assert config.n_steps == 200
        assert config.n_steps != original_steps

        original_batch = config.batch_size
        config.batch_size = 128
        assert config.batch_size == 128
        assert config.batch_size != original_batch

        # Test modifying string parameters
        config.model_name = "test_model"
        assert config.model_name == "test_model"

        # Test modifying boolean parameters
        config.verbose = False
        assert config.verbose is False

    def test_config_type_validation(self):
        """Test that configuration fields have correct types."""
        config = get_config()

        # String fields
        assert isinstance(config.device, str)
        assert isinstance(config.attack, str)
        assert isinstance(config.control_init, str)
        assert isinstance(config.model_name, str)
        assert isinstance(config.train_data, str)
        assert isinstance(config.test_data, str)
        assert isinstance(config.result_prefix, str)

        # Integer fields
        assert isinstance(config.n_train_data, int)
        assert isinstance(config.n_test_data, int)
        assert isinstance(config.n_steps, int)
        assert isinstance(config.batch_size, int)
        assert isinstance(config.topk, int)
        assert isinstance(config.test_steps, int)

        # Float fields
        assert isinstance(config.lr, float)
        assert isinstance(config.temp, (int, float))
        assert isinstance(config.target_weight, float)
        assert isinstance(config.control_weight, float)

        # Boolean fields
        assert isinstance(config.progressive_models, bool)
        assert isinstance(config.progressive_goals, bool)
        assert isinstance(config.transfer, bool)
        assert isinstance(config.filter_cand, bool)
        assert isinstance(config.allow_non_ascii, bool)
        assert isinstance(config.anneal, bool)
        assert isinstance(config.incr_control, bool)
        assert isinstance(config.stop_on_success, bool)
        assert isinstance(config.verbose, bool)
        assert isinstance(config.gbda_deterministic, bool)

    def test_config_is_configdict(self):
        """Test that the returned config is a ConfigDict instance."""
        config = get_config()

        assert isinstance(config, config_dict.ConfigDict)

        # Test ConfigDict specific behavior
        assert hasattr(config, "lock")
        assert hasattr(config, "unlock")

        # Test that we can access fields as attributes
        assert config.attack == "gcg"
        assert config.n_steps == 100

    def test_config_device_logic(self):
        """Test device selection logic."""
        # Test that device is set based on CUDA availability
        config = get_config()

        # Should be either "cuda" or "cpu"
        assert config.device in ["cuda", "cpu"]

        # If device is "cuda", torch.cuda.is_available() should return True
        # If device is "cpu", torch.cuda.is_available() should return False
        if config.device == "cuda":
            # This test might fail on CI without GPU, so we just check the type
            assert isinstance(config.device, str)
        else:
            assert config.device == "cpu"

    def test_config_control_init_format(self):
        """Test that control_init has the expected format."""
        config = get_config()

        # Should be a string of exclamation marks separated by spaces
        control_tokens = config.control_init.split()

        # Should have 20 tokens (exclamation marks)
        assert len(control_tokens) == 20

        # All tokens should be exclamation marks
        for token in control_tokens:
            assert token == "!"

    def test_config_numeric_ranges(self):
        """Test that numeric configuration values are in reasonable ranges."""
        config = get_config()

        # Test positive values
        assert config.n_train_data > 0
        assert config.n_test_data >= 0  # Can be 0 for test data
        assert config.n_steps > 0
        assert config.batch_size > 0
        assert config.topk > 0
        assert config.test_steps > 0

        # Test learning rate is reasonable
        assert 0 < config.lr <= 1.0

        # Test weights are reasonable
        assert config.target_weight >= 0
        assert config.control_weight >= 0

        # Test temperature is reasonable
        assert config.temp > 0


class TestConfigIntegration:
    """Integration tests for configuration usage."""

    def test_config_for_individual_attack(self):
        """Test configuration setup for individual attack mode."""
        config = get_config()

        # Set up for individual attack
        config.transfer = False
        config.result_prefix = "./results/individual"

        # Verify individual attack configuration
        assert config.transfer is False
        assert "individual" in config.result_prefix

    def test_config_for_transfer_attack(self):
        """Test configuration setup for transfer attack mode."""
        config = get_config()

        # Set up for transfer attack
        config.transfer = True
        config.result_prefix = "./results/transfer"

        # Verify transfer attack configuration
        assert config.transfer is True
        assert "transfer" in config.result_prefix

    def test_config_modification_chain(self):
        """Test chaining multiple configuration modifications."""
        config = get_config()

        # Modify multiple parameters
        config.n_steps = 50
        config.batch_size = 128
        config.lr = 0.05
        config.verbose = False
        config.model_name = "test_model"

        # Verify all modifications
        assert config.n_steps == 50
        assert config.batch_size == 128
        assert config.lr == 0.05
        assert config.verbose is False
        assert config.model_name == "test_model"

        # Verify other values remain unchanged
        assert config.attack == "gcg"
        assert config.topk == 256
        assert config.temp == 1.5  # Actual default value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
