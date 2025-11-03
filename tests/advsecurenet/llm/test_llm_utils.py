import pytest
import os
import json
import tempfile
import time
import logging
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from pydantic import BaseModel

from advsecurenet.llm_finetuning.utils import (
    set_seed_all,
    dist_info,
    is_rank_zero,
    rank_zero_only,
    bf16_supported,
    default_dtype,
    device_map_auto,
    ensure_padding_token,
    lora_default_targets,
    ensure_dir,
    save_json,
    save_run_config,
    resolve_path,
    time_block,
    cuda_mem,
    count_trainable_params
)


class TestSetSeedAll:
    """Test set_seed_all function."""

    @patch("advsecurenet.llm_finetuning.utils.torch")
    @patch("advsecurenet.llm_finetuning.utils.np.random")
    @patch("advsecurenet.llm_finetuning.utils.random")
    def test_basic_seeding(self, mock_random, mock_np_random, mock_torch):
        """Test basic seeding without deterministic mode."""
        mock_torch.cuda.is_available.return_value = False

        set_seed_all(123, deterministic=False)

        mock_random.seed.assert_called_once_with(123)
        mock_np_random.seed.assert_called_once_with(123)
        mock_torch.manual_seed.assert_called_once_with(123)
        mock_torch.cuda.manual_seed_all.assert_not_called()

    @patch("advsecurenet.llm_finetuning.utils.torch")
    @patch("advsecurenet.llm_finetuning.utils.np.random")
    @patch("advsecurenet.llm_finetuning.utils.random")
    def test_cuda_seeding(self, mock_random, mock_np_random, mock_torch):
        """Test seeding with CUDA available."""
        mock_torch.cuda.is_available.return_value = True

        set_seed_all(456)

        mock_torch.cuda.manual_seed_all.assert_called_once_with(456)

    @patch("advsecurenet.llm_finetuning.utils.torch")
    @patch("advsecurenet.llm_finetuning.utils.np.random")
    @patch("advsecurenet.llm_finetuning.utils.random")
    def test_deterministic_mode(self, mock_random, mock_np_random, mock_torch):
        """Test deterministic mode enablement."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.use_deterministic_algorithms = Mock()

        with patch("torch.backends.cudnn") as mock_cudnn:
            set_seed_all(789, deterministic=True)

            mock_torch.use_deterministic_algorithms.assert_called_once_with(True)
            assert mock_cudnn.deterministic is True
            assert mock_cudnn.benchmark is False

    @patch("advsecurenet.llm_finetuning.utils.torch")
    @patch("advsecurenet.llm_finetuning.utils.np.random")
    @patch("advsecurenet.llm_finetuning.utils.random")
    def test_deterministic_exceptions_ignored(
        self, mock_random, mock_np_random, mock_torch
    ):
        """Test that exceptions in deterministic setup are ignored."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.use_deterministic_algorithms.side_effect = Exception("Not supported")

        # Should not raise exception
        set_seed_all(999, deterministic=True)

        mock_torch.use_deterministic_algorithms.assert_called_once_with(True)


class TestDistInfo:
    """Test distributed training info functions."""

    def test_dist_info_from_env_vars(self):
        """Test dist_info reads from environment variables."""
        env_vars = {"WORLD_SIZE": "4", "RANK": "2", "LOCAL_RANK": "1"}

        with patch.dict(os.environ, env_vars):
            info = dist_info()

        assert info["world_size"] == 4
        assert info["rank"] == 2
        assert info["local_rank"] == 1

    def test_dist_info_defaults(self):
        """Test dist_info defaults when env vars not set."""
        with patch.dict(os.environ, {}, clear=True):
            info = dist_info()

        assert info["world_size"] == 1
        assert info["rank"] == 0
        assert info["local_rank"] == 0

    @patch("torch.distributed")
    def test_dist_info_with_torch_distributed(self, mock_dist):
        """Test dist_info with torch.distributed initialized."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 8
        mock_dist.get_rank.return_value = 3

        with patch.dict(os.environ, {"LOCAL_RANK": "2"}):
            info = dist_info()

        assert info["world_size"] == 8
        assert info["rank"] == 3
        assert info["local_rank"] == 2

    def test_is_rank_zero_true(self):
        """Test is_rank_zero returns True for rank 0."""
        with patch(
            "advsecurenet.llm_finetuning.utils.dist_info", return_value={"rank": 0}
        ):
            assert is_rank_zero() is True

    def test_is_rank_zero_false(self):
        """Test is_rank_zero returns False for non-zero rank."""
        with patch(
            "advsecurenet.llm_finetuning.utils.dist_info", return_value={"rank": 1}
        ):
            assert is_rank_zero() is False

    def test_rank_zero_only_decorator(self):
        """Test rank_zero_only decorator."""

        @rank_zero_only
        def test_func(x):
            return x * 2

        # Test rank 0 - function should execute
        with patch("advsecurenet.llm_finetuning.utils.is_rank_zero", return_value=True):
            result = test_func(5)
            assert result == 10

        # Test non-zero rank - function should not execute
        with patch(
            "advsecurenet.llm_finetuning.utils.is_rank_zero", return_value=False
        ):
            result = test_func(5)
            assert result is None


class TestDeviceCapabilities:
    """Test device capability detection functions."""

    @patch("advsecurenet.llm_finetuning.utils.torch")
    def test_bf16_supported_no_cuda(self, mock_torch):
        """Test bf16_supported when CUDA not available."""
        mock_torch.cuda.is_available.return_value = False

        assert bf16_supported() is False

    @patch("advsecurenet.llm_finetuning.utils.torch")
    def test_bf16_supported_ampere_gpu(self, mock_torch):
        """Test bf16_supported with Ampere GPU (compute capability >= 8.0)."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_capability.return_value = (8, 6)  # Ampere

        assert bf16_supported() is True

    @patch("advsecurenet.llm_finetuning.utils.torch")
    def test_bf16_supported_older_gpu(self, mock_torch):
        """Test bf16_supported with older GPU (compute capability < 8.0)."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_capability.return_value = (7, 5)  # Pascal/Turing

        assert bf16_supported() is False

    @patch("advsecurenet.llm_finetuning.utils.torch")
    def test_bf16_supported_exception_handling(self, mock_torch):
        """Test bf16_supported handles exceptions gracefully."""
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.current_device.side_effect = Exception("GPU error")

        assert bf16_supported() is False

    @patch("advsecurenet.llm_finetuning.utils.torch")
    @patch("advsecurenet.llm_finetuning.utils.bf16_supported")
    def test_default_dtype_bf16_preferred(self, mock_bf16_supported, mock_torch):
        """Test default_dtype with bf16 preference and support."""
        mock_bf16_supported.return_value = True

        result = default_dtype(prefer_bf16=True)

        assert result == mock_torch.bfloat16

    @patch("advsecurenet.llm_finetuning.utils.torch")
    @patch("advsecurenet.llm_finetuning.utils.bf16_supported")
    def test_default_dtype_bf16_not_supported(self, mock_bf16_supported, mock_torch):
        """Test default_dtype when bf16 not supported."""
        mock_bf16_supported.return_value = False
        mock_torch.cuda.is_available.return_value = True

        result = default_dtype(prefer_bf16=True)

        assert result == mock_torch.float16

    @patch("advsecurenet.llm_finetuning.utils.torch")
    @patch("advsecurenet.llm_finetuning.utils.bf16_supported")
    def test_default_dtype_cpu_only(self, mock_bf16_supported, mock_torch):
        """Test default_dtype on CPU-only system."""
        mock_bf16_supported.return_value = False
        mock_torch.cuda.is_available.return_value = False

        result = default_dtype(prefer_bf16=False)

        assert result == mock_torch.float32

    @patch("advsecurenet.llm_finetuning.utils.torch")
    def test_device_map_auto_cuda(self, mock_torch):
        """Test device_map_auto with CUDA available."""
        mock_torch.cuda.is_available.return_value = True

        result = device_map_auto()

        assert result == "auto"

    @patch("advsecurenet.llm_finetuning.utils.torch")
    def test_device_map_auto_cpu(self, mock_torch):
        """Test device_map_auto on CPU-only system."""
        mock_torch.cuda.is_available.return_value = False

        result = device_map_auto()

        assert result == "cpu"


class TestTokenizerUtils:
    """Test tokenizer utility functions."""

    def test_ensure_padding_token_missing_pad_token(self):
        """Test ensure_padding_token when pad_token is missing."""
        tokenizer = Mock()
        tokenizer.pad_token = None
        tokenizer.eos_token = "<|endoftext|>"
        tokenizer.pad_token_id = None
        tokenizer.eos_token_id = 50256

        ensure_padding_token(tokenizer)

        assert tokenizer.pad_token == "<|endoftext|>"
        assert tokenizer.pad_token_id == 50256
        assert tokenizer.padding_side == "right"

    def test_ensure_padding_token_already_set(self):
        """Test ensure_padding_token when pad_token already exists."""
        tokenizer = Mock()
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = 0

        ensure_padding_token(tokenizer)

        assert tokenizer.pad_token == "<pad>"  # Unchanged
        assert tokenizer.pad_token_id == 0  # Unchanged
        assert tokenizer.padding_side == "right"

    def test_ensure_padding_token_no_eos_token(self):
        """Test ensure_padding_token when no eos_token available."""
        tokenizer = Mock()
        tokenizer.pad_token = None
        tokenizer.eos_token = None

        ensure_padding_token(tokenizer)

        assert tokenizer.pad_token is None  # Should remain None
        assert tokenizer.padding_side == "right"


class TestLoRATargets:
    """Test LoRA target detection."""

    def test_lora_default_targets_common_patterns(self):
        """Test LoRA target detection with common module names."""
        model = Mock()
        model.named_modules.return_value = [
            ("layer.0.attention.q_proj", Mock()),
            ("layer.0.attention.v_proj", Mock()),
            ("layer.1.attention.k_proj", Mock()),
            ("layer.1.attention.o_proj", Mock()),
            ("other.module", Mock()),
        ]

        targets = lora_default_targets(model)

        expected = [
            "layer.0.attention.q_proj",
            "layer.0.attention.v_proj",
            "layer.1.attention.k_proj",
            "layer.1.attention.o_proj",
        ]
        assert sorted(targets) == sorted(expected)

    def test_lora_default_targets_no_matches(self):
        """Test LoRA target detection when no common patterns found."""
        model = Mock()
        model.named_modules.return_value = [
            ("unknown.module1", Mock()),
            ("unknown.module2", Mock()),
        ]

        targets = lora_default_targets(model)

        assert targets == ["q_proj", "v_proj"]  # Fallback

    def test_lora_default_targets_mixed_patterns(self):
        """Test LoRA target detection with various naming patterns."""
        model = Mock()
        model.named_modules.return_value = [
            ("transformer.h.0.attn.c_proj", Mock()),
            ("transformer.h.1.Wqkv", Mock()),
            ("model.layers.0.self_attn.query_key_value", Mock()),
        ]

        targets = lora_default_targets(model)

        expected = [
            "model.layers.0.self_attn.query_key_value",
            "transformer.h.0.attn.c_proj",
            "transformer.h.1.Wqkv",
        ]
        assert sorted(targets) == sorted(expected)


class TestIOUtils:
    """Test I/O utility functions."""

    def test_ensure_dir_new_directory(self):
        """Test ensure_dir creates new directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            new_path = Path(tmp_dir) / "new" / "nested" / "dir"

            result = ensure_dir(new_path)

            assert result == new_path
            assert new_path.exists()
            assert new_path.is_dir()

    def test_ensure_dir_existing_directory(self):
        """Test ensure_dir with existing directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            existing_path = Path(tmp_dir)

            result = ensure_dir(existing_path)

            assert result == existing_path
            assert existing_path.exists()

    @patch("advsecurenet.llm_finetuning.utils.is_rank_zero")
    def test_save_json_rank_zero(self, mock_is_rank_zero):
        """Test save_json only executes on rank 0."""
        mock_is_rank_zero.return_value = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.json"
            test_data = {"key": "value", "number": 42}

            save_json(test_data, file_path)

            assert file_path.exists()
            with file_path.open() as f:
                loaded_data = json.load(f)
            assert loaded_data == test_data

    @patch("advsecurenet.llm_finetuning.utils.is_rank_zero")
    def test_save_json_non_zero_rank(self, mock_is_rank_zero):
        """Test save_json skips execution on non-zero rank."""
        mock_is_rank_zero.return_value = False

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test.json"

            class TestIOUtils:
                """Test I/O utility functions."""

                def test_ensure_dir_new_directory(self):
                    """Test ensure_dir creates new directory."""
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        new_path = Path(tmp_dir) / "new" / "nested" / "dir"

                        result = ensure_dir(new_path)

                        assert result == new_path
                        assert new_path.exists()
                        assert new_path.is_dir()

                def test_ensure_dir_existing_directory(self):
                    """Test ensure_dir with existing directory."""
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        existing_path = Path(tmp_dir)

                        result = ensure_dir(existing_path)

                        assert result == existing_path
                        assert existing_path.exists()

                @patch("advsecurenet.llm_finetuning.utils.is_rank_zero")
                def test_save_json_rank_zero(self, mock_is_rank_zero):
                    """Test save_json only executes on rank 0."""
                    mock_is_rank_zero.return_value = True

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        file_path = Path(tmp_dir) / "test.json"
                        test_data = {"key": "value", "number": 42}

                        save_json(test_data, file_path)

                        assert file_path.exists()
                        with file_path.open() as f:
                            loaded_data = json.load(f)
                        assert loaded_data == test_data

                @patch("advsecurenet.llm_finetuning.utils.is_rank_zero")
                def test_save_json_non_zero_rank(self, mock_is_rank_zero):
                    """Test save_json skips execution on non-zero rank."""
                    mock_is_rank_zero.return_value = False

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        file_path = Path(tmp_dir) / "test.json"

                        save_json({"key": "value"}, file_path)

                        assert not file_path.exists()  # Should not be created

                @patch("advsecurenet.llm_finetuning.utils.save_json")
                @patch("advsecurenet.llm_finetuning.utils.is_rank_zero")
                def test_save_run_config_pydantic_model(
                    self, mock_is_rank_zero, mock_save_json
                ):
                    """Test save_run_config with Pydantic model."""
                    mock_is_rank_zero.return_value = True

                    class TestModel(BaseModel):
                        name: str
                        value: int

                    config = TestModel(name="test", value=123)

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        save_run_config(config, tmp_dir)

                    mock_save_json.assert_called_once()
                    call_args = mock_save_json.call_args
                    saved_data = call_args[0][0]
                    saved_path = call_args[0][1]

                    assert saved_data == {"name": "test", "value": 123}
                    assert str(saved_path).endswith("config.resolved.json")

                @patch("advsecurenet.llm_finetuning.utils.save_json")
                @patch("advsecurenet.llm_finetuning.utils.is_rank_zero")
                def test_save_run_config_dict_like(
                    self, mock_is_rank_zero, mock_save_json
                ):
                    """Test save_run_config with dict-like object."""
                    mock_is_rank_zero.return_value = True

                    config = {"model": "gpt-2", "lr": 1e-4}

                    with tempfile.TemporaryDirectory() as tmp_dir:
                        save_run_config(config, tmp_dir)

                    mock_save_json.assert_called_once()
                    call_args = mock_save_json.call_args
                    saved_data = call_args[0][0]
                    saved_path = call_args[0][1]

                    assert saved_data == {"model": "gpt-2", "lr": 1e-4}
                    assert str(saved_path).endswith("config.resolved.json")

            class TestPathUtils:
                """Test path utility functions."""

                def test_resolve_path_absolute(self):
                    """Test resolve_path with absolute path."""
                    abs_path = "/absolute/path/to/file"
                    result = resolve_path(abs_path)
                    assert result == Path(abs_path).resolve()

                def test_resolve_path_relative(self):
                    """Test resolve_path with relative path."""
                    rel_path = "relative/path"
                    result = resolve_path(rel_path)
                    assert result == Path(rel_path).resolve()

                def test_resolve_path_home_expansion(self):
                    """Test resolve_path with home directory expansion."""
                    home_path = "~/documents/file.txt"
                    result = resolve_path(home_path)
                    assert result == Path(home_path).expanduser().resolve()

                def test_resolve_path_pathlib_object(self):
                    """Test resolve_path with Path object."""
                    path_obj = Path("some/path")
                    result = resolve_path(path_obj)
                    assert result == path_obj.resolve()

            class TestTimeBlock:
                """Test time_block context manager."""

                def test_time_block_basic(self, capsys):
                    """Test basic time_block functionality."""
                    with time_block("test operation"):
                        time.sleep(0.1)

                    captured = capsys.readouterr()
                    assert "test operation took" in captured.out
                    assert "seconds" in captured.out

                def test_time_block_with_exception(self, capsys):
                    """Test time_block still prints timing when exception occurs."""
                    with pytest.raises(ValueError):
                        with time_block("failing operation"):
                            time.sleep(0.05)
                            raise ValueError("test error")

                    captured = capsys.readouterr()
                    assert "failing operation took" in captured.out

            class TestCudaMemory:
                """Test CUDA memory utility functions."""

                @patch("advsecurenet.llm_finetuning.utils.torch")
                def test_cuda_mem_with_cuda(self, mock_torch):
                    """Test cuda_mem when CUDA is available."""
                    mock_torch.cuda.is_available.return_value = True
                    mock_torch.cuda.memory_allocated.return_value = (
                        1024 * 1024 * 512
                    )  # 512 MB
                    mock_torch.cuda.memory_reserved.return_value = (
                        1024 * 1024 * 1024
                    )  # 1 GB

                    result = cuda_mem()

                    expected = {"allocated_mb": 512.0, "reserved_mb": 1024.0}
                    assert result == expected

                @patch("advsecurenet.llm_finetuning.utils.torch")
                def test_cuda_mem_without_cuda(self, mock_torch):
                    """Test cuda_mem when CUDA is not available."""
                    mock_torch.cuda.is_available.return_value = False

                    result = cuda_mem()

                    expected = {"allocated_mb": 0.0, "reserved_mb": 0.0}
                    assert result == expected

                @patch("advsecurenet.llm_finetuning.utils.torch")
                def test_cuda_mem_exception_handling(self, mock_torch):
                    """Test cuda_mem handles exceptions gracefully."""
                    mock_torch.cuda.is_available.return_value = True
                    mock_torch.cuda.memory_allocated.side_effect = Exception(
                        "CUDA error"
                    )

                    result = cuda_mem()

                    expected = {"allocated_mb": 0.0, "reserved_mb": 0.0}
                    assert result == expected

            class TestModelUtils:
                """Test model utility functions."""

                def test_count_trainable_params_all_trainable(self):
                    """Test count_trainable_params with all parameters trainable."""
                    model = Mock()

                    # Mock parameters
                    param1 = Mock()
                    param1.requires_grad = True
                    param1.numel.return_value = 1000

                    param2 = Mock()
                    param2.requires_grad = True
                    param2.numel.return_value = 2000

                    model.parameters.return_value = [param1, param2]

                    result = count_trainable_params(model)

                    assert result == 3000

                def test_count_trainable_params_mixed(self):
                    """Test count_trainable_params with mixed trainable/frozen parameters."""
                    model = Mock()

                    # Mock parameters
                    param1 = Mock()
                    param1.requires_grad = True
                    param1.numel.return_value = 1000

                    param2 = Mock()
                    param2.requires_grad = False  # Frozen
                    param2.numel.return_value = 2000

                    param3 = Mock()
                    param3.requires_grad = True
                    param3.numel.return_value = 500

                    model.parameters.return_value = [param1, param2, param3]

                    result = count_trainable_params(model)

                    assert result == 1500  # Only trainable params

                def test_count_trainable_params_none_trainable(self):
                    """Test count_trainable_params with no trainable parameters."""
                    model = Mock()

                    param1 = Mock()
                    param1.requires_grad = False
                    param1.numel.return_value = 1000

                    model.parameters.return_value = [param1]

                    result = count_trainable_params(model)

                    assert result == 0


            class TestLoggingUtils:
                """Test logging-related utilities."""

                def test_rank_zero_only_with_logging(self):
                    """Test rank_zero_only decorator with logging function."""

                    @rank_zero_only
                    def log_message(msg):
                        logging.info(msg)
                        return f"logged: {msg}"

                    # Test on rank 0
                    with patch(
                        "advsecurenet.llm_finetuning.utils.is_rank_zero",
                        return_value=True,
                    ):
                        with patch("logging.info") as mock_log:
                            result = log_message("test message")
                            mock_log.assert_called_once_with("test message")
                            assert result == "logged: test message"

                    # Test on non-zero rank
                    with patch(
                        "advsecurenet.llm_finetuning.utils.is_rank_zero",
                        return_value=False,
                    ):
                        with patch("logging.info") as mock_log:
                            result = log_message("test message")
                            mock_log.assert_not_called()
                            assert result is None

            class TestEdgeCases:
                """Test edge cases and error conditions."""

                def test_ensure_dir_with_file_path(self):
                    """Test ensure_dir with a file path (should create parent directory)."""
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        file_path = Path(tmp_dir) / "subdir" / "file.txt"

                        result = ensure_dir(file_path.parent)

                        assert result == file_path.parent
                        assert file_path.parent.exists()

                def test_resolve_path_empty_string(self):
                    """Test resolve_path with empty string."""
                    result = resolve_path("")
                    assert result == Path("").resolve()

                def test_time_block_nested(self, capsys):
                    """Test nested time_block context managers."""
                    with time_block("outer operation"):
                        time.sleep(0.05)
                        with time_block("inner operation"):
                            time.sleep(0.05)

                    captured = capsys.readouterr()
                    assert "outer operation took" in captured.out
                    assert "inner operation took" in captured.out

                @patch("advsecurenet.llm_finetuning.utils.torch")
                def test_default_dtype_edge_cases(self, mock_torch):
                    """Test default_dtype with various edge cases."""
                    # Test when both CUDA and bf16 are unavailable
                    mock_torch.cuda.is_available.return_value = False

                    with patch(
                        "advsecurenet.llm_finetuning.utils.bf16_supported",
                        return_value=False,
                    ):
                        result = default_dtype(prefer_bf16=True)
                        assert result == mock_torch.float32

                def test_lora_default_targets_empty_model(self):
                    """Test LoRA target detection with empty model."""
                    model = Mock()
                    model.named_modules.return_value = []

                    targets = lora_default_targets(model)

                    assert targets == ["q_proj", "v_proj"]  # Should return fallback
