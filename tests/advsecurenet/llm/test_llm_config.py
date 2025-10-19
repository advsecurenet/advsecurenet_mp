import pytest
import tempfile
import yaml
from pathlib import Path
from pydantic import ValidationError

from advsecurenet.llm_finetuning.config import (
    PEFTConfig,
    DataConfig, 
    TrainConfig,
    Config,
    load_yaml,
    merge
)


class TestPEFTConfig:
    """Test PEFTConfig model."""
    
    def test_defaults(self):
        """Test default values."""
        config = PEFTConfig()
        assert config.enabled is True
        assert config.quantization == "qlora"
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.05
        assert config.target_modules == ["q_proj", "v_proj"]
    
    def test_custom_values(self):
        """Test custom values."""
        config = PEFTConfig(
            enabled=False,
            quantization="bnb-4bit",
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
        )
        assert config.enabled is False
        assert config.quantization == "bnb-4bit"
        assert config.r == 16
        assert config.lora_alpha == 32
        assert config.lora_dropout == 0.1
        assert config.target_modules == ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    def test_invalid_quantization(self):
        """Test invalid quantization value."""
        with pytest.raises(ValidationError):
            PEFTConfig(quantization="invalid")


class TestDataConfig:
    """Test DataConfig model."""
    
    def test_local_file_mode(self):
        """Test with local file configuration."""
        config = DataConfig(train_file="train.jsonl")
        assert config.train_file == "train.jsonl"
        assert config.eval_file is None
        assert config.max_seq_len == 2048
    
    def test_hub_mode(self):
        """Test with HuggingFace Hub configuration."""
        config = DataConfig(
            hub_name="gsm8k",
            hub_config="main",
            hub_train_split="train[:200]",
            hub_eval_split="test[:50]"
        )
        assert config.hub_name == "gsm8k"
        assert config.hub_config == "main"
        assert config.hub_train_split == "train[:200]"
        assert config.hub_eval_split == "test[:50]"
    
    def test_validation_error_no_source(self):
        """Test validation error when neither train_file nor hub_name is provided."""
        with pytest.raises(ValidationError, match="Provide either data.train_file or data.hub_name"):
            DataConfig()
    
    def test_validation_success_with_train_file(self):
        """Test validation passes with train_file."""
        config = DataConfig(train_file="train.jsonl")
        assert config.train_file == "train.jsonl"
    
    def test_validation_success_with_hub_name(self):
        """Test validation passes with hub_name."""
        config = DataConfig(hub_name="dataset")
        assert config.hub_name == "dataset"


class TestTrainConfig:
    """Test TrainConfig model."""
    
    def test_defaults(self):
        """Test default values."""
        config = TrainConfig(model_name="test-model")
        assert config.model_name == "test-model"
        assert config.output_dir == "outputs/sft"
        assert config.learning_rate == 2e-4
        assert config.weight_decay == 0.0
        assert config.max_steps == 1000
        assert config.evaluation_strategy == "steps"
        assert config.lr_scheduler_type == "cosine"
        assert config.seed == 42
    
    def test_custom_values(self):
        """Test custom values."""
        config = TrainConfig(
            model_name="custom-model",
            output_dir="custom-output",
            learning_rate=1e-3,
            max_steps=2000,
            evaluation_strategy="epoch",
            lr_scheduler_type="linear"
        )
        assert config.model_name == "custom-model"
        assert config.output_dir == "custom-output"
        assert config.learning_rate == 1e-3
        assert config.max_steps == 2000
        assert config.evaluation_strategy == "epoch"
        assert config.lr_scheduler_type == "linear"
    
    def test_invalid_evaluation_strategy(self):
        """Test invalid evaluation strategy."""
        with pytest.raises(ValidationError):
            TrainConfig(model_name="test", evaluation_strategy="invalid")
    
    def test_invalid_lr_scheduler_type(self):
        """Test invalid learning rate scheduler type."""
        with pytest.raises(ValidationError):
            TrainConfig(model_name="test", lr_scheduler_type="invalid")


class TestConfig:
    """Test main Config model."""
    
    def test_complete_config(self):
        """Test complete configuration."""
        config = Config(
            train=TrainConfig(model_name="test-model"),
            data=DataConfig(train_file="train.jsonl"),
            peft=PEFTConfig(enabled=False)
        )
        assert isinstance(config.train, TrainConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.peft, PEFTConfig)
        assert config.train.model_name == "test-model"
        assert config.data.train_file == "train.jsonl"
        assert config.peft.enabled is False
    
    def test_default_peft_config(self):
        """Test default PEFT configuration."""
        config = Config(
            train=TrainConfig(model_name="test-model"),
            data=DataConfig(train_file="train.jsonl")
        )
        assert isinstance(config.peft, PEFTConfig)
        assert config.peft.enabled is True  # Default value


class TestLoadYaml:
    """Test load_yaml function."""
    
    def test_load_valid_yaml(self):
        """Test loading valid YAML configuration."""
        yaml_content = {
            "train": {"model_name": "test-model"},
            "data": {"train_file": "train.jsonl"},
            "peft": {"enabled": False}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            config_path = f.name
        
        try:
            config = load_yaml(config_path)
            assert isinstance(config, Config)
            assert config.train.model_name == "test-model"
            assert config.data.train_file == "train.jsonl"
            assert config.peft.enabled is False
        finally:
            Path(config_path).unlink()
    
    def test_load_minimal_yaml(self):
        """Test loading minimal YAML configuration."""
        yaml_content = {
            "train": {"model_name": "test-model"},
            "data": {"hub_name": "dataset"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            config_path = f.name
        
        try:
            config = load_yaml(config_path)
            assert config.train.model_name == "test-model"
            assert config.data.hub_name == "dataset"
            assert config.peft.enabled is True  # Default
        finally:
            Path(config_path).unlink()
    
    def test_load_nonexistent_file(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_yaml("/nonexistent/config.yaml")
    
    def test_load_invalid_yaml(self):
        """Test loading invalid YAML."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = f.name
        
        try:
            with pytest.raises(yaml.YAMLError):
                load_yaml(config_path)
        finally:
            Path(config_path).unlink()
    
    def test_load_invalid_config_structure(self):
        """Test loading YAML with invalid config structure."""
        yaml_content = {"invalid": "structure"}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            config_path = f.name
        
        try:
            with pytest.raises(ValidationError):
                load_yaml(config_path)
        finally:
            Path(config_path).unlink()


class TestMerge:
    """Test merge function."""
    
    def test_merge_empty_override(self):
        """Test merge with empty override."""
        base = Config(
            train=TrainConfig(model_name="base-model"),
            data=DataConfig(train_file="base.jsonl")
        )
        result = merge(base, {})
        assert result.train.model_name == "base-model"
        assert result.data.train_file == "base.jsonl"
    
    def test_merge_with_none_values(self):
        """Test merge filters out None values."""
        base = Config(
            train=TrainConfig(model_name="base-model"),
            data=DataConfig(train_file="base.jsonl")
        )
        override = {"model_name": None, "learning_rate": 1e-3}
        result = merge(base, override)
        assert result.train.model_name == "base-model"  # None filtered out
        # The merge function only works at the top level, not nested
        # So learning_rate won't actually be updated since it's a nested field
        assert result.train.learning_rate == 2e-4      # Unchanged (default value)
    
    def test_merge_top_level_updates_only(self):
        """Test merge only updates top-level fields (based on actual implementation)."""
        base = Config(
            train=TrainConfig(model_name="base-model", learning_rate=2e-4),
            data=DataConfig(train_file="base.jsonl")
        )
        
        # The merge function appears to only work with top-level Config fields
        # Create new nested objects for testing
        new_train = TrainConfig(model_name="new-model", learning_rate=1e-3)
        override = {"train": new_train}
        result = merge(base, override)
        assert result.train.model_name == "new-model"
        assert result.train.learning_rate == 1e-3
    
    def test_merge_nested_config_replacement(self):
        """Test merge replaces entire nested config objects."""
        base = Config(
            train=TrainConfig(model_name="base-model"),
            data=DataConfig(train_file="base.jsonl"),
            peft=PEFTConfig(enabled=True, r=8)
        )
        
        # Test replacing entire nested configs
        new_data = DataConfig(train_file="new.jsonl", eval_file="eval.jsonl")
        new_peft = PEFTConfig(enabled=False, r=16)
        
        override = {"data": new_data, "peft": new_peft}
        result = merge(base, override)
        assert result.data.train_file == "new.jsonl"
        assert result.data.eval_file == "eval.jsonl"
        assert result.peft.enabled is False
        assert result.peft.r == 16
    
    def test_merge_preserves_unmodified_sections(self):
        """Test merge preserves sections not in override."""
        base = Config(
            train=TrainConfig(model_name="base-model", learning_rate=2e-4, max_steps=1000),
            data=DataConfig(train_file="base.jsonl", max_seq_len=2048)
        )
        
        # Only update train section
        new_train = TrainConfig(model_name="new-model", learning_rate=1e-3, max_steps=2000)
        override = {"train": new_train}
        result = merge(base, override)
        
        assert result.train.model_name == "new-model"      # Updated
        assert result.train.learning_rate == 1e-3         # Updated  
        assert result.train.max_steps == 2000             # Updated
        assert result.data.train_file == "base.jsonl"     # Preserved
        assert result.data.max_seq_len == 2048            # Preserved


class TestIntegration:
    """Integration tests combining multiple functions."""
    
    def test_yaml_load_and_merge_workflow(self):
        """Test typical workflow: load YAML then merge overrides."""
        yaml_content = {
            "train": {"model_name": "yaml-model", "learning_rate": 1e-4},
            "data": {"train_file": "yaml.jsonl"}
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            config_path = f.name
        
        try:
            # Load from YAML
            base_config = load_yaml(config_path)
            assert base_config.train.model_name == "yaml-model"
            
            # Merge CLI overrides (using full config replacement approach)
            new_train = TrainConfig(
                model_name="cli-model", 
                learning_rate=1e-4,  # Keep original
                max_steps=2000       # New value
            )
            override = {"train": new_train}
            final_config = merge(base_config, override)
            
            assert final_config.train.model_name == "cli-model"   # Overridden
            assert final_config.train.learning_rate == 1e-4      # From original
            assert final_config.train.max_steps == 2000          # From override
        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=advsecurenet.llm_finetuning.config", "--cov-report=html"])