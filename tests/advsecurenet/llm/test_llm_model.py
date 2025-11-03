import pytest
import torch
import os
from unittest.mock import Mock, patch, MagicMock
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from advsecurenet.llm_finetuning.model import (
    load_tokenizer,
    load_model,
    _in_distributed,
    _select_device_map,
)
from advsecurenet.llm_finetuning.config import Config


class TestLoadTokenizer:
    """Test cases for the load_tokenizer function."""

    @patch("advsecurenet.llm_finetuning.model.AutoTokenizer")
    def test_load_tokenizer_basic(self, mock_auto_tokenizer):
        """Test basic tokenizer loading with default settings."""
        # Setup
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        cfg = Mock()
        cfg.train.model_name = "test-model"

        # Execute
        result = load_tokenizer(cfg)

        # Verify
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(
            "test-model", use_fast=True
        )
        assert result.padding_side == "right"
        assert result == mock_tokenizer

    @patch("advsecurenet.llm_finetuning.model.AutoTokenizer")
    def test_load_tokenizer_no_pad_token_with_eos(self, mock_auto_tokenizer):
        """Test tokenizer loading when pad_token is None but eos_token exists."""
        # Setup
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer.pad_token_id = None
        mock_tokenizer.eos_token_id = 1
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        cfg = Mock()
        cfg.train.model_name = "test-model"

        # Execute
        result = load_tokenizer(cfg)

        # Verify
        assert result.pad_token == "[EOS]"
        assert result.pad_token_id == 1
        assert result.padding_side == "right"

    @patch("advsecurenet.llm_finetuning.model.AutoTokenizer")
    def test_load_tokenizer_no_pad_token_with_eos_pad_token_id_minus_one(
        self, mock_auto_tokenizer
    ):
        """Test tokenizer loading when pad_token_id is -1."""
        # Setup
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer.pad_token_id = -1
        mock_tokenizer.eos_token_id = 1
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        cfg = Mock()
        cfg.train.model_name = "test-model"

        # Execute
        result = load_tokenizer(cfg)

        # Verify
        assert result.pad_token == "[EOS]"
        assert result.pad_token_id == 1

    @patch("advsecurenet.llm_finetuning.model.AutoTokenizer")
    def test_load_tokenizer_no_eos_token(self, mock_auto_tokenizer):
        """Test tokenizer loading when both pad_token and eos_token are None."""
        # Setup
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = None
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        cfg = Mock()
        cfg.train.model_name = "test-model"

        # Execute
        result = load_tokenizer(cfg)

        # Verify - should not modify pad_token when eos_token is None
        assert result.padding_side == "right"
        # pad_token should remain None since eos_token is None
        assert result.pad_token is None

    @patch("advsecurenet.llm_finetuning.model.AutoTokenizer")
    def test_load_tokenizer_existing_pad_token(self, mock_auto_tokenizer):
        """Test tokenizer loading when pad_token already exists."""
        # Setup
        mock_tokenizer = Mock()
        mock_tokenizer.pad_token = "[PAD]"
        mock_tokenizer.eos_token = "[EOS]"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        cfg = Mock()
        cfg.train.model_name = "test-model"

        # Execute
        result = load_tokenizer(cfg)

        # Verify - should not change existing pad_token
        assert result.pad_token == "[PAD]"
        assert result.pad_token_id == 0
        assert result.padding_side == "right"


class TestInDistributed:
    """Test cases for the _in_distributed function."""

    def test_in_distributed_world_size_greater_than_one(self):
        """Test distributed detection when WORLD_SIZE > 1."""
        with patch.dict(os.environ, {"WORLD_SIZE": "2"}):
            assert _in_distributed() is True

    def test_in_distributed_local_rank_exists(self):
        """Test distributed detection when LOCAL_RANK is set."""
        with patch.dict(os.environ, {"LOCAL_RANK": "0"}, clear=True):
            assert _in_distributed() is True

    def test_in_distributed_torch_distributed_available_and_initialized(self):
        """Test distributed detection via torch.distributed."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("torch.distributed.is_available", return_value=True):
                with patch("torch.distributed.is_initialized", return_value=True):
                    assert _in_distributed() is True

    def test_in_distributed_torch_distributed_not_available(self):
        """Test when torch.distributed is not available."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("torch.distributed.is_available", return_value=False):
                assert _in_distributed() is False

    def test_in_distributed_torch_distributed_not_initialized(self):
        """Test when torch.distributed is available but not initialized."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("torch.distributed.is_available", return_value=True):
                with patch("torch.distributed.is_initialized", return_value=False):
                    assert _in_distributed() is False

    def test_in_distributed_import_exception(self):
        """Test when importing torch.distributed raises an exception."""
        with patch.dict(os.environ, {}, clear=True):
            # Patch torch.distributed methods directly instead of the import
            with patch("torch.distributed.is_available", side_effect=ImportError()):
                assert _in_distributed() is False

    def test_in_distributed_no_environment_variables(self):
        """Test when no relevant environment variables are set."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("torch.distributed.is_available", return_value=False):
                assert _in_distributed() is False


class TestSelectDeviceMap:
    """Test cases for the _select_device_map function."""

    @patch("advsecurenet.llm_finetuning.model._in_distributed")
    def test_select_device_map_distributed(self, mock_in_distributed):
        """Test device map selection in distributed mode."""
        mock_in_distributed.return_value = True

        result = _select_device_map()

        assert result is None

    @patch("advsecurenet.llm_finetuning.model._in_distributed")
    @patch("advsecurenet.llm_finetuning.model.torch.cuda.is_available")
    def test_select_device_map_single_process_cuda_available(
        self, mock_cuda_available, mock_in_distributed
    ):
        """Test device map selection in single process mode with CUDA available."""
        mock_in_distributed.return_value = False
        mock_cuda_available.return_value = True

        result = _select_device_map()

        assert result == "auto"

    @patch("advsecurenet.llm_finetuning.model._in_distributed")
    @patch("advsecurenet.llm_finetuning.model.torch.cuda.is_available")
    def test_select_device_map_single_process_cuda_not_available(
        self, mock_cuda_available, mock_in_distributed
    ):
        """Test device map selection in single process mode with CUDA not available."""
        mock_in_distributed.return_value = False
        mock_cuda_available.return_value = False

        result = _select_device_map()

        assert result == "cpu"


class TestLoadModel:
    """Test cases for the load_model function."""

    @patch("advsecurenet.llm_finetuning.model.get_peft_model")
    @patch("advsecurenet.llm_finetuning.model.LoraConfig")
    @patch("advsecurenet.llm_finetuning.model.prepare_model_for_kbit_training")
    @patch("advsecurenet.llm_finetuning.model.AutoModelForCausalLM")
    @patch("advsecurenet.llm_finetuning.model._select_device_map")
    def test_load_model_no_peft(
        self,
        mock_select_device_map,
        mock_auto_model,
        mock_prepare_model,
        mock_lora_config,
        mock_get_peft_model,
    ):
        """Test loading model without PEFT."""
        # Setup
        mock_select_device_map.return_value = "auto"
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model

        cfg = Mock()
        cfg.train.model_name = "test-model"
        cfg.peft.enabled = False

        # Execute
        result = load_model(cfg)

        # Verify
        mock_auto_model.from_pretrained.assert_called_once_with(
            "test-model",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=False,
        )
        mock_prepare_model.assert_not_called()
        mock_lora_config.assert_not_called()
        mock_get_peft_model.assert_not_called()
        assert result == mock_model

    @patch("advsecurenet.llm_finetuning.model.get_peft_model")
    @patch("advsecurenet.llm_finetuning.model.LoraConfig")
    @patch("advsecurenet.llm_finetuning.model.prepare_model_for_kbit_training")
    @patch("advsecurenet.llm_finetuning.model.AutoModelForCausalLM")
    @patch("advsecurenet.llm_finetuning.model._select_device_map")
    def test_load_model_with_peft_no_quantization(
        self,
        mock_select_device_map,
        mock_auto_model,
        mock_prepare_model,
        mock_lora_config,
        mock_get_peft_model,
    ):
        """Test loading model with PEFT but no quantization."""
        # Setup
        mock_select_device_map.return_value = "auto"
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_lora = Mock()
        mock_lora_config.return_value = mock_lora
        mock_peft_model = Mock()
        mock_get_peft_model.return_value = mock_peft_model

        cfg = Mock()
        cfg.train.model_name = "test-model"
        cfg.peft.enabled = True
        cfg.peft.quantization = "none"
        cfg.peft.r = 16
        cfg.peft.lora_alpha = 32
        cfg.peft.lora_dropout = 0.1
        cfg.peft.target_modules = ["q_proj", "v_proj"]

        # Execute
        result = load_model(cfg)

        # Verify
        mock_auto_model.from_pretrained.assert_called_once_with(
            "test-model",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=False,
        )
        mock_prepare_model.assert_not_called()
        mock_lora_config.assert_called_once_with(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        mock_get_peft_model.assert_called_once_with(mock_model, mock_lora)
        assert result == mock_peft_model

    @patch("advsecurenet.llm_finetuning.model.get_peft_model")
    @patch("advsecurenet.llm_finetuning.model.LoraConfig")
    @patch("advsecurenet.llm_finetuning.model.prepare_model_for_kbit_training")
    @patch("advsecurenet.llm_finetuning.model.AutoModelForCausalLM")
    @patch("advsecurenet.llm_finetuning.model._select_device_map")
    def test_load_model_with_qlora_quantization(
        self,
        mock_select_device_map,
        mock_auto_model,
        mock_prepare_model,
        mock_lora_config,
        mock_get_peft_model,
    ):
        """Test loading model with QLoRA quantization."""
        # Setup
        mock_select_device_map.return_value = "auto"
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_prepared_model = Mock()
        mock_prepare_model.return_value = mock_prepared_model
        mock_lora = Mock()
        mock_lora_config.return_value = mock_lora
        mock_peft_model = Mock()
        mock_get_peft_model.return_value = mock_peft_model

        cfg = Mock()
        cfg.train.model_name = "test-model"
        cfg.peft.enabled = True
        cfg.peft.quantization = "qlora"
        cfg.peft.r = 8
        cfg.peft.lora_alpha = 16
        cfg.peft.lora_dropout = 0.05
        cfg.peft.target_modules = ["q_proj"]

        # Execute
        result = load_model(cfg)

        # Verify
        mock_auto_model.from_pretrained.assert_called_once_with(
            "test-model",
            device_map="auto",
            torch_dtype=torch.float16,
            load_in_4bit=True,
        )
        mock_prepare_model.assert_called_once_with(mock_model)
        mock_lora_config.assert_called_once_with(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        mock_get_peft_model.assert_called_once_with(mock_prepared_model, mock_lora)
        assert result == mock_peft_model

    @patch("advsecurenet.llm_finetuning.model.get_peft_model")
    @patch("advsecurenet.llm_finetuning.model.LoraConfig")
    @patch("advsecurenet.llm_finetuning.model.prepare_model_for_kbit_training")
    @patch("advsecurenet.llm_finetuning.model.AutoModelForCausalLM")
    @patch("advsecurenet.llm_finetuning.model._select_device_map")
    def test_load_model_with_bnb_4bit_quantization(
        self,
        mock_select_device_map,
        mock_auto_model,
        mock_prepare_model,
        mock_lora_config,
        mock_get_peft_model,
    ):
        """Test loading model with BNB 4-bit quantization."""
        # Setup
        mock_select_device_map.return_value = "cpu"
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_prepared_model = Mock()
        mock_prepare_model.return_value = mock_prepared_model
        mock_lora = Mock()
        mock_lora_config.return_value = mock_lora
        mock_peft_model = Mock()
        mock_get_peft_model.return_value = mock_peft_model

        cfg = Mock()
        cfg.train.model_name = "test-model"
        cfg.peft.enabled = True
        cfg.peft.quantization = "bnb-4bit"
        cfg.peft.r = 4
        cfg.peft.lora_alpha = 8
        cfg.peft.lora_dropout = 0.0
        cfg.peft.target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

        # Execute
        result = load_model(cfg)

        # Verify
        mock_auto_model.from_pretrained.assert_called_once_with(
            "test-model", device_map="cpu", torch_dtype=torch.float16, load_in_4bit=True
        )
        mock_prepare_model.assert_called_once_with(mock_model)
        mock_lora_config.assert_called_once_with(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        mock_get_peft_model.assert_called_once_with(mock_prepared_model, mock_lora)
        assert result == mock_peft_model

    @patch("advsecurenet.llm_finetuning.model.get_peft_model")
    @patch("advsecurenet.llm_finetuning.model.LoraConfig")
    @patch("advsecurenet.llm_finetuning.model.prepare_model_for_kbit_training")
    @patch("advsecurenet.llm_finetuning.model.AutoModelForCausalLM")
    @patch("advsecurenet.llm_finetuning.model._select_device_map")
    def test_load_model_with_peft_other_quantization(
        self,
        mock_select_device_map,
        mock_auto_model,
        mock_prepare_model,
        mock_lora_config,
        mock_get_peft_model,
    ):
        """Test loading model with PEFT and other quantization method."""
        # Setup
        mock_select_device_map.return_value = None
        mock_model = Mock()
        mock_auto_model.from_pretrained.return_value = mock_model
        mock_lora = Mock()
        mock_lora_config.return_value = mock_lora
        mock_peft_model = Mock()
        mock_get_peft_model.return_value = mock_peft_model

        cfg = Mock()
        cfg.train.model_name = "test-model"
        cfg.peft.enabled = True
        cfg.peft.quantization = "int8"  # Not qlora or bnb-4bit
        cfg.peft.r = 32
        cfg.peft.lora_alpha = 64
        cfg.peft.lora_dropout = 0.2
        cfg.peft.target_modules = ["dense"]

        # Execute
        result = load_model(cfg)

        # Verify
        mock_auto_model.from_pretrained.assert_called_once_with(
            "test-model",
            device_map=None,
            torch_dtype=torch.float16,
            load_in_4bit=False,  # Should be False since quantization is not qlora or bnb-4bit
        )
        mock_prepare_model.assert_not_called()  # Should not be called for non-4bit quantization
        mock_lora_config.assert_called_once_with(
            r=32,
            lora_alpha=64,
            lora_dropout=0.2,
            target_modules=["dense"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        mock_get_peft_model.assert_called_once_with(mock_model, mock_lora)
        assert result == mock_peft_model


class TestIntegration:
    """Integration tests for edge cases and error conditions."""

    @patch("advsecurenet.llm_finetuning.model.AutoTokenizer")
    def test_load_tokenizer_with_missing_attributes(self, mock_auto_tokenizer):
        """Test tokenizer loading when tokenizer is missing some attributes."""
        # Setup tokenizer with missing attributes
        mock_tokenizer = Mock()
        # Simulate missing attributes by using spec
        mock_tokenizer.configure_mock(**{"pad_token": None, "eos_token": None})
        # Remove pad_token_id and eos_token_id attributes entirely
        del mock_tokenizer.pad_token_id
        del mock_tokenizer.eos_token_id
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        cfg = Mock()
        cfg.train.model_name = "test-model"

        # Execute
        result = load_tokenizer(cfg)

        # Verify - should handle missing attributes gracefully
        assert result.padding_side == "right"
        assert result == mock_tokenizer

    def test_in_distributed_with_world_size_one(self):
        """Test _in_distributed with WORLD_SIZE=1 (edge case)."""
        with patch.dict(os.environ, {"WORLD_SIZE": "1"}, clear=True):
            with patch("torch.distributed.is_available", return_value=False):
                assert _in_distributed() is False

    def test_in_distributed_with_empty_local_rank(self):
        """Test _in_distributed with empty LOCAL_RANK."""
        with patch.dict(os.environ, {"LOCAL_RANK": ""}, clear=True):
            with patch("torch.distributed.is_available", return_value=False):
                # Empty LOCAL_RANK still counts as being set
                assert _in_distributed() is True


# Additional edge case tests - FIXED VERSION
class TestEdgeCases:
    """Test edge cases and error conditions."""

    @patch("advsecurenet.llm_finetuning.model.AutoTokenizer")
    def test_load_tokenizer_getattr_behavior(self, mock_auto_tokenizer):
        """Test tokenizer behavior with getattr edge cases."""

        # Create a custom tokenizer class that simulates getattr behavior
        class CustomTokenizer:
            def __init__(self):
                self._pad_token = None
                self._eos_token = "[EOS]"
                self._pad_token_id = None
                self._eos_token_id = 1
                self.padding_side = None

            def __getattr__(self, name):
                if name == "pad_token":
                    return self._pad_token
                elif name == "eos_token":
                    return self._eos_token
                elif name == "pad_token_id":
                    return self._pad_token_id
                elif name == "eos_token_id":
                    return self._eos_token_id
                else:
                    raise AttributeError(
                        f"'{type(self).__name__}' object has no attribute '{name}'"
                    )

            def __setattr__(self, name, value):
                if name in [
                    "_pad_token",
                    "_eos_token",
                    "_pad_token_id",
                    "_eos_token_id",
                    "padding_side",
                ]:
                    super().__setattr__(name, value)
                elif name == "pad_token":
                    self._pad_token = value
                elif name == "pad_token_id":
                    self._pad_token_id = value
                else:
                    super().__setattr__(name, value)

        mock_tokenizer = CustomTokenizer()
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        cfg = Mock()
        cfg.train.model_name = "test-model"

        # Execute
        result = load_tokenizer(cfg)

        # Verify getattr behavior worked correctly
        assert result.padding_side == "right"
        assert result.pad_token == "[EOS]"  # Should be set from eos_token
        assert result.pad_token_id == 1  # Should be set from eos_token_id


# Test fixtures and utilities
@pytest.fixture
def mock_config():
    """Create a mock configuration object for testing."""
    cfg = Mock()
    cfg.train.model_name = "test-model"
    cfg.peft.enabled = False
    cfg.peft.quantization = "none"
    cfg.peft.r = 16
    cfg.peft.lora_alpha = 32
    cfg.peft.lora_dropout = 0.1
    cfg.peft.target_modules = ["q_proj", "v_proj"]
    return cfg


@pytest.fixture
def mock_tokenizer():
    """Create a mock tokenizer for testing."""
    tokenizer = Mock()
    tokenizer.pad_token = "[PAD]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    return tokenizer


if __name__ == "__main__":
    pytest.main(
        [__file__, "-v", "--cov=advsecurenet.llm_finetuning.model", "--cov-report=html"]
    )
