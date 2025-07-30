import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from transformers import TrainingArguments

@dataclass
class ModelConfig:
    model_name: str = "bert-base-uncased"
    num_labels: int = 2
    model_type: str = "auto"  # auto, custom, pretrained
    checkpoint_path: Optional[str] = None

@dataclass
class DataConfig:
    dataset_name: Optional[str] = None
    dataset_path: Optional[str] = None
    text_column: str = "text"
    label_column: str = "label"
    validation_split: float = 0.2
    max_length: int = 128
    preprocessing_num_workers: int = 4

@dataclass
class TrainingConfig:
    output_dir: str = "./results"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    evaluation_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True
    save_total_limit: int = 3
    logging_dir: str = "./logs"

class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.model_config = ModelConfig()
        self.data_config = DataConfig()
        self.training_config = TrainingConfig()
        
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load configuration from YAML or JSON file."""
        with open(config_path, 'r') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        if 'model' in config_dict:
            self.model_config = ModelConfig(**config_dict['model'])
        if 'data' in config_dict:
            self.data_config = DataConfig(**config_dict['data'])
        if 'training' in config_dict:
            self.training_config = TrainingConfig(**config_dict['training'])
    
    
    def get_training_arguments(self) -> TrainingArguments:
        """Convert training config to HuggingFace TrainingArguments."""
        return TrainingArguments(**asdict(self.training_config))
