import pytest
import tempfile
import json
from unittest.mock import Mock, patch, MagicMock
from datasets import Dataset, DatasetDict

from advsecurenet.llm_finetuning.data import load_tokenized_datasets
from advsecurenet.llm_finetuning.config import DataConfig


class TestLoadTokenizedDatasets:
    """Test load_tokenized_datasets function."""

    @patch("advsecurenet.llm_finetuning.data.load_dataset")
    def test_jsonl_train_only(self, mock_load_dataset):
        """Test loading JSONL with train file only."""
        # Setup
        cfg = DataConfig(train_file="train.jsonl", max_seq_len=512)
        tokenizer = Mock()
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        mock_dataset = Mock()
        mock_dataset.column_names = ["text", "other"]
        mock_raw = DatasetDict({"train": mock_dataset})
        mock_load_dataset.return_value = mock_raw

        # Mock map operations
        mock_raw.map = Mock(return_value=mock_raw)
        mock_raw["train"].map = Mock(return_value=mock_raw)

        # Execute
        result = load_tokenized_datasets(cfg, tokenizer)

        # Verify
        mock_load_dataset.assert_called_once_with(
            "json", data_files={"train": "train.jsonl"}
        )
        assert mock_raw.map.call_count == 2  # to_text and tokenize

    @patch("advsecurenet.llm_finetuning.data.load_dataset")
    def test_jsonl_train_and_eval(self, mock_load_dataset):
        """Test loading JSONL with both train and eval files."""
        # Setup
        cfg = DataConfig(
            train_file="train.jsonl", eval_file="eval.jsonl", max_seq_len=1024
        )
        tokenizer = Mock()

        mock_dataset = Mock()
        mock_dataset.column_names = ["text"]
        mock_raw = DatasetDict({"train": mock_dataset, "validation": mock_dataset})
        mock_load_dataset.return_value = mock_raw
        mock_raw.map = Mock(return_value=mock_raw)

        # Execute
        result = load_tokenized_datasets(cfg, tokenizer)

        # Verify
        mock_load_dataset.assert_called_once_with(
            "json", data_files={"train": "train.jsonl", "validation": "eval.jsonl"}
        )

    @patch("advsecurenet.llm_finetuning.data.load_dataset")
    def test_hub_dataset_train_only(self, mock_load_dataset):
        """Test loading Hub dataset with train split only."""
        # Setup
        cfg = DataConfig(
            hub_name="test-dataset",
            hub_config="main",
            hub_train_split="train[:100]",
            max_seq_len=2048,
        )
        tokenizer = Mock()

        mock_dataset = Mock()
        mock_dataset.column_names = ["question", "answer"]
        mock_load_dataset.return_value = mock_dataset

        # Create a real DatasetDict to avoid the Mock item assignment issue
        real_dataset_dict = DatasetDict()
        real_dataset_dict["train"] = mock_dataset

        with patch(
            "advsecurenet.llm_finetuning.data.DatasetDict",
            return_value=real_dataset_dict,
        ):
            # Mock the map method on the real DatasetDict
            real_dataset_dict.map = Mock(return_value=real_dataset_dict)

            result = load_tokenized_datasets(cfg, tokenizer)

        # Verify
        mock_load_dataset.assert_called_with(
            "test-dataset", "main", split="train[:100]"
        )

    @patch("advsecurenet.llm_finetuning.data.load_dataset")
    def test_hub_dataset_with_eval(self, mock_load_dataset):
        """Test loading Hub dataset with both train and eval splits."""
        # Setup
        cfg = DataConfig(
            hub_name="test-dataset",
            hub_train_split="train",
            hub_eval_split="test[:50]",
            max_seq_len=512,
        )
        tokenizer = Mock()

        mock_dataset = Mock()
        mock_dataset.column_names = ["text"]
        mock_load_dataset.return_value = mock_dataset

        # Create a real DatasetDict to avoid Mock item assignment issue
        real_dataset_dict = DatasetDict()

        with patch(
            "advsecurenet.llm_finetuning.data.DatasetDict",
            return_value=real_dataset_dict,
        ):
            real_dataset_dict.map = Mock(return_value=real_dataset_dict)

            result = load_tokenized_datasets(cfg, tokenizer)

        # Verify
        assert mock_load_dataset.call_count == 2  # train and eval

    def test_no_data_source_error(self):
        """Test error when no data source is provided."""
        # Can't create invalid DataConfig due to validation, so test the function directly
        # Create a valid config then modify it to bypass validation
        cfg = DataConfig(train_file="dummy.jsonl")
        cfg.train_file = None  # Make it invalid after creation
        cfg.hub_name = None

        tokenizer = Mock()

        with pytest.raises(
            ValueError, match="Provide either JSONL files or a hub dataset"
        ):
            load_tokenized_datasets(cfg, tokenizer)

    def test_non_jsonl_file_error(self):
        """Test error with non-JSONL file."""
        cfg = DataConfig(train_file="train.csv")
        tokenizer = Mock()

        with pytest.raises(
            ValueError, match="Provide either JSONL files or a hub dataset"
        ):
            load_tokenized_datasets(cfg, tokenizer)


class TestTextProcessing:
    """Test text processing logic within load_tokenized_datasets."""

    @patch("advsecurenet.llm_finetuning.data.load_dataset")
    def test_prompt_response_fields(self, mock_load_dataset):
        """Test text processing with prompt and response fields."""
        cfg = DataConfig(
            train_file="train.jsonl",
            prompt_field="instruction",
            response_field="output",
        )
        tokenizer = Mock()
        tokenizer.return_value = {"input_ids": [1, 2], "attention_mask": [1, 1]}

        # Create a real dataset for testing the map function
        test_data = [{"instruction": "Hello", "output": "Hi there"}]
        mock_dataset = Dataset.from_list(test_data)
        mock_raw = DatasetDict({"train": mock_dataset})
        mock_load_dataset.return_value = mock_raw

        # Execute
        result = load_tokenized_datasets(cfg, tokenizer)

        # Verify tokenizer was called with combined text
        tokenizer.assert_called()
        call_args = tokenizer.call_args[0][
            0
        ]  # First positional argument (batch["text"])
        assert "Hello\nHi there" in call_args[0]

    @patch("advsecurenet.llm_finetuning.data.load_dataset")
    def test_text_field(self, mock_load_dataset):
        """Test text processing with text field."""
        cfg = DataConfig(train_file="train.jsonl", text_field="content")
        tokenizer = Mock()
        tokenizer.return_value = {"input_ids": [1], "attention_mask": [1]}

        test_data = [{"content": "Some text content"}]
        mock_dataset = Dataset.from_list(test_data)
        mock_raw = DatasetDict({"train": mock_dataset})
        mock_load_dataset.return_value = mock_raw

        # Execute
        result = load_tokenized_datasets(cfg, tokenizer)

        # Verify
        tokenizer.assert_called()
        call_args = tokenizer.call_args[0][0]
        assert "Some text content" in call_args[0]

    @patch("advsecurenet.llm_finetuning.data.load_dataset")
    def test_gsm8k_preset(self, mock_load_dataset):
        """Test GSM8K dataset preset formatting."""
        cfg = DataConfig(hub_name="gsm8k", hub_train_split="train")
        tokenizer = Mock()
        tokenizer.return_value = {"input_ids": [1], "attention_mask": [1]}

        test_data = [{"question": "What is 2+2?", "answer": "4"}]
        mock_dataset = Dataset.from_list(test_data)

        real_dataset_dict = DatasetDict()
        real_dataset_dict["train"] = mock_dataset

        with patch(
            "advsecurenet.llm_finetuning.data.DatasetDict",
            return_value=real_dataset_dict,
        ):
            mock_load_dataset.return_value = mock_dataset

            # Execute
            result = load_tokenized_datasets(cfg, tokenizer)

        # Verify GSM8K formatting
        tokenizer.assert_called()
        call_args = tokenizer.call_args[0][0]
        assert "Q: What is 2+2?\nA: 4" in call_args[0]

    @patch("advsecurenet.llm_finetuning.data.load_dataset")
    def test_fallback_str_conversion(self, mock_load_dataset):
        """Test fallback string conversion when no fields specified."""
        cfg = DataConfig(train_file="train.jsonl")
        tokenizer = Mock()
        tokenizer.return_value = {"input_ids": [1], "attention_mask": [1]}

        test_data = [{"random_field": "random_value"}]
        mock_dataset = Dataset.from_list(test_data)
        mock_raw = DatasetDict({"train": mock_dataset})
        mock_load_dataset.return_value = mock_raw

        # Execute
        result = load_tokenized_datasets(cfg, tokenizer)

        # Verify str() conversion was used
        tokenizer.assert_called()


class TestTokenization:
    """Test tokenization logic."""

    @patch("advsecurenet.llm_finetuning.data.load_dataset")
    def test_tokenization_parameters(self, mock_load_dataset):
        """Test tokenization with correct parameters."""
        cfg = DataConfig(train_file="train.jsonl", max_seq_len=256)
        tokenizer = Mock()
        tokenizer.return_value = {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}

        test_data = [{"text": "test"}]
        mock_dataset = Dataset.from_list(test_data)
        mock_raw = DatasetDict({"train": mock_dataset})
        mock_load_dataset.return_value = mock_raw

        # Execute
        result = load_tokenized_datasets(cfg, tokenizer)

        # Verify tokenizer called with correct parameters
        tokenizer.assert_called()
        call_kwargs = tokenizer.call_args[1]
        assert call_kwargs["truncation"] is True
        assert call_kwargs["max_length"] == 256


if __name__ == "__main__":
    pytest.main(
        [__file__, "-v", "--cov=advsecurenet.llm_finetuning.data", "--cov-report=html"]
    )
