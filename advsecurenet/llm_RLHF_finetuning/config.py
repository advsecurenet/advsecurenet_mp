from dataclasses import dataclass
from typing import Optional

@dataclass
class DataConfig:
    dataset_name: str
    dataset_config: Optional[str] = None
    train_split: str = "train"
    test_split: str = "test"

@dataclass
class ModelConfig:
    model_name_or_path: str
    trust_remote_code: bool = True
    use_peft: bool = False

    # LoRA/PEFT arguments
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list = None
    lora_task_type: str = "CAUSAL_LM"
    use_rslora: bool = False
    use_dora: bool = False
    bias: str = "none"
    lora_modules_to_save: list = None
    init_lora_weights: bool = True

    gradient_checkpointing: bool = False
    quantize: bool = False

@dataclass
class TrainingConfig:
    output_dir: str
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    fp16: bool = False
    logging_steps: int = 10
    save_steps: int = 50
    evaluation_strategy: str = "steps"
    eval_steps: int = 100
    push_to_hub: bool = False
