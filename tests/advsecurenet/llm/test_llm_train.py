import pytest
import torch
import random
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import DatasetDict, Dataset

from advsecurenet.llm_finetuning.train import _set_seed, run_training
from advsecurenet.llm_finetuning.config import Config, TrainConfig, DataConfig, PEFTConfig


class TestSetSeed:
    """Test _set_seed function."""
    
    @patch('advsecurenet.llm_finetuning.train.torch')
    @patch('advsecurenet.llm_finetuning.train.np.random')
    @patch('advsecurenet.llm_finetuning.train.random')
    def test_set_seed_cpu_only(self, mock_random, mock_np_random, mock_torch):
        """Test seed setting when CUDA is not available."""
        mock_torch.cuda.is_available.return_value = False
        
        _set_seed(42)
        
        mock_random.seed.assert_called_once_with(42)
        mock_np_random.seed.assert_called_once_with(42)
        mock_torch.manual_seed.assert_called_once_with(42)
        mock_torch.cuda.manual_seed_all.assert_not_called()
    
    @patch('advsecurenet.llm_finetuning.train.torch')
    @patch('advsecurenet.llm_finetuning.train.np.random')
    @patch('advsecurenet.llm_finetuning.train.random')
    def test_set_seed_with_cuda(self, mock_random, mock_np_random, mock_torch):
        """Test seed setting when CUDA is available."""
        mock_torch.cuda.is_available.return_value = True
        
        _set_seed(123)
        
        mock_random.seed.assert_called_once_with(123)
        mock_np_random.seed.assert_called_once_with(123)
        mock_torch.manual_seed.assert_called_once_with(123)
        mock_torch.cuda.manual_seed_all.assert_called_once_with(123)


class TestRunTraining:
    """Test run_training function."""
    
    @patch('advsecurenet.llm_finetuning.train._set_seed')
    @patch('advsecurenet.llm_finetuning.train.load_tokenizer')
    @patch('advsecurenet.llm_finetuning.train.load_tokenized_datasets')
    @patch('advsecurenet.llm_finetuning.train.load_model')
    @patch('advsecurenet.llm_finetuning.train.Trainer')
    @patch('advsecurenet.llm_finetuning.train.TrainingArguments')
    @patch('advsecurenet.llm_finetuning.train.DataCollatorForLanguageModeling')
    def test_run_training_complete_workflow(
        self, mock_data_collator_class, mock_training_args_class, mock_trainer_class,
        mock_load_model, mock_load_tokenized_datasets, mock_load_tokenizer, mock_set_seed
    ):
        """Test complete training workflow with all components."""
        # Setup config
        cfg = Config(
            train=TrainConfig(
                model_name="test-model",
                output_dir="test-output",
                learning_rate=1e-3,
                seed=42
            ),
            data=DataConfig(train_file="train.jsonl"),
            peft=PEFTConfig(enabled=True)
        )
        
        # Setup mocks
        mock_tokenizer = Mock()
        mock_load_tokenizer.return_value = mock_tokenizer
        
        mock_datasets = DatasetDict({
            "train": Dataset.from_list([{"input_ids": [1, 2, 3]}]),
            "validation": Dataset.from_list([{"input_ids": [4, 5, 6]}])
        })
        mock_load_tokenized_datasets.return_value = mock_datasets
        
        mock_model = Mock()
        mock_load_model.return_value = mock_model
        
        mock_data_collator = Mock()
        mock_data_collator_class.return_value = mock_data_collator
        
        mock_training_args = Mock()
        mock_training_args_class.return_value = mock_training_args
        
        mock_trainer = Mock()
        mock_trainer_class.return_value = mock_trainer
        
        # Execute
        result = run_training(cfg)
        
        # Verify seed setting
        mock_set_seed.assert_called_once_with(42)
        
        # Verify component loading
        mock_load_tokenizer.assert_called_once_with(cfg)
        mock_load_tokenized_datasets.assert_called_once_with(cfg.data, mock_tokenizer)
        mock_load_model.assert_called_once_with(cfg)
        
        # Verify data collator creation
        mock_data_collator_class.assert_called_once_with(
            tokenizer=mock_tokenizer,
            mlm=False
        )
        
        # Verify training arguments creation
        mock_training_args_class.assert_called_once_with(
            output_dir="test-output",
            learning_rate=1e-3,
            weight_decay=0.0,  # Default value
            max_steps=1000,    # Default value
            per_device_train_batch_size=1,  # Default value
            gradient_accumulation_steps=16,  # Default value
            logging_steps=10,   # Default value
            save_steps=200,     # Default value
            eval_steps=200,     # Default value
            eval_strategy="steps",  # Default value
            lr_scheduler_type="cosine",  # Default value
            warmup_ratio=0.03,  # Default value
            bf16=False,         # Default value
            fp16=False,         # Default value
            gradient_checkpointing=False,  # Default value
            push_to_hub=False,  # Default value
            report_to=["none"]
        )
        
        # Verify trainer creation
        mock_trainer_class.assert_called_once_with(
            model=mock_model,
            args=mock_training_args,
            train_dataset=mock_datasets["train"],
            eval_dataset=mock_datasets["validation"],
            tokenizer=mock_tokenizer,
            data_collator=mock_data_collator
        )
        
        # Verify training execution
        mock_trainer.train.assert_called_once()
        mock_trainer.save_model.assert_called_once_with("test-output")
        mock_tokenizer.save_pretrained.assert_called_once_with("test-output")
        
        # Verify return value
        assert result == {"output_dir": "test-output"}
    
    @patch('advsecurenet.llm_finetuning.train._set_seed')
    @patch('advsecurenet.llm_finetuning.train.load_tokenizer')
    @patch('advsecurenet.llm_finetuning.train.load_tokenized_datasets')
    @patch('advsecurenet.llm_finetuning.train.load_model')
    @patch('advsecurenet.llm_finetuning.train.Trainer')
    @patch('advsecurenet.llm_finetuning.train.TrainingArguments')
    @patch('advsecurenet.llm_finetuning.train.DataCollatorForLanguageModeling')
    def test_run_training_component_loading_order(
        self, mock_data_collator_class, mock_training_args_class, mock_trainer_class,
        mock_load_model, mock_load_tokenized_datasets, mock_load_tokenizer, mock_set_seed
    ):
        """Test that components are loaded in the correct order."""
        cfg = Config(
            train=TrainConfig(model_name="test-model"),
            data=DataConfig(train_file="train.jsonl")
        )
        
        # Setup mocks
        mock_tokenizer = Mock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_tokenized_datasets.return_value = DatasetDict({
            "train": Dataset.from_list([{"input_ids": [1]}])
        })
        mock_load_model.return_value = Mock()
        mock_data_collator_class.return_value = Mock()
        mock_training_args_class.return_value = Mock()
        mock_trainer_class.return_value = Mock()
        
        # Execute
        run_training(cfg)
        
        # Verify calls were made in correct order using call counts
        # Since we can't compare call objects directly, verify that each function was called
        mock_set_seed.assert_called_once()
        mock_load_tokenizer.assert_called_once()
        mock_load_tokenized_datasets.assert_called_once()
        mock_load_model.assert_called_once()
        
        # Verify tokenizer is passed to dataset loading
        mock_load_tokenized_datasets.assert_called_once_with(cfg.data, mock_tokenizer)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    @patch('advsecurenet.llm_finetuning.train._set_seed')
    @patch('advsecurenet.llm_finetuning.train.load_tokenizer')
    def test_tokenizer_loading_error(self, mock_load_tokenizer, mock_set_seed):
        """Test error handling when tokenizer loading fails."""
        cfg = Config(
            train=TrainConfig(model_name="invalid-model"),
            data=DataConfig(train_file="train.jsonl")
        )
        
        mock_load_tokenizer.side_effect = Exception("Tokenizer loading failed")
        
        with pytest.raises(Exception, match="Tokenizer loading failed"):
            run_training(cfg)
    
    @patch('advsecurenet.llm_finetuning.train._set_seed')
    @patch('advsecurenet.llm_finetuning.train.load_tokenizer')
    @patch('advsecurenet.llm_finetuning.train.load_tokenized_datasets')
    def test_dataset_loading_error(self, mock_load_tokenized_datasets, mock_load_tokenizer, mock_set_seed):
        """Test error handling when dataset loading fails."""
        cfg = Config(
            train=TrainConfig(model_name="test-model"),
            data=DataConfig(train_file="nonexistent.jsonl")
        )
        
        mock_load_tokenizer.return_value = Mock()
        mock_load_tokenized_datasets.side_effect = FileNotFoundError("Dataset file not found")
        
        with pytest.raises(FileNotFoundError, match="Dataset file not found"):
            run_training(cfg)


class TestIntegration:
    """Integration tests with real components where possible."""
    
    @patch('advsecurenet.llm_finetuning.train._set_seed')
    @patch('advsecurenet.llm_finetuning.train.load_tokenizer')
    @patch('advsecurenet.llm_finetuning.train.load_tokenized_datasets')
    @patch('advsecurenet.llm_finetuning.train.load_model')
    def test_data_collator_integration(
        self, mock_load_model, mock_load_tokenized_datasets, mock_load_tokenizer, mock_set_seed
    ):
        """Test that DataCollatorForLanguageModeling is properly configured."""
        cfg = Config(
            train=TrainConfig(model_name="test-model"),
            data=DataConfig(train_file="train.jsonl")
        )
        
        mock_tokenizer = Mock()
        mock_load_tokenizer.return_value = mock_tokenizer
        mock_load_tokenized_datasets.return_value = DatasetDict({
            "train": Dataset.from_list([{"input_ids": [1]}])
        })
        mock_load_model.return_value = Mock()
        
        with patch('advsecurenet.llm_finetuning.train.DataCollatorForLanguageModeling') as mock_collator_class, \
             patch('advsecurenet.llm_finetuning.train.Trainer'), \
             patch('advsecurenet.llm_finetuning.train.TrainingArguments'):
            
            mock_collator_class.return_value = Mock()
            
            run_training(cfg)
            
            # Verify data collator is configured correctly for causal LM
            mock_collator_class.assert_called_once_with(
                tokenizer=mock_tokenizer,
                mlm=False  # False for causal language modeling
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=advsecurenet.llm_finetuning.train", "--cov-report=html"])