# llm_finetune/config.py
from pydantic import BaseModel, Field
from typing import Optional, Literal, List
import yaml
from pathlib import Path

class PEFTConfig(BaseModel):
    enabled: bool = True
    quantization: Optional[Literal["qlora", "bnb-4bit"]] = "qlora"
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: List[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])

# config.py
from pydantic import BaseModel, Field, model_validator
from typing import Optional, List

class DataConfig(BaseModel):
    # Local JSONL option
    train_file: Optional[str] = None
    eval_file: Optional[str] = None
    text_field: Optional[str] = None
    prompt_field: Optional[str] = None
    response_field: Optional[str] = None

    max_seq_len: int = 2048

    # HF Hub option
    hub_name: Optional[str] = None        # e.g., "gsm8k"
    hub_config: Optional[str] = None      # e.g., "main"
    hub_train_split: Optional[str] = None # e.g., "train[:200]"
    hub_eval_split: Optional[str] = None  # e.g., "test[:50]"

    @model_validator(mode="after")
    def _one_of_source(self):
        if not self.train_file and not self.hub_name:
            raise ValueError("Provide either data.train_file or data.hub_name")
        return self

class TrainConfig(BaseModel):
    model_name: str
    output_dir: str = "outputs/sft"
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    max_steps: int = 1000
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    logging_steps: int = 10
    save_steps: int = 200
    eval_steps: int = 200
    evaluation_strategy: Literal["no", "steps", "epoch"] = "steps"
    lr_scheduler_type: Literal["cosine", "linear", "constant_with_warmup"] = "cosine"
    warmup_ratio: float = 0.03
    bf16: bool = False
    gradient_checkpointing: bool = False
    push_to_hub: bool = False
    seed: int = 42

class Config(BaseModel):
    train: TrainConfig
    data: DataConfig
    peft: PEFTConfig = PEFTConfig()

def load_yaml(path: str) -> Config:
    return Config(**yaml.safe_load(Path(path).read_text()))

def merge(base: Config, override: dict) -> Config:
    # override can come from CLI kwargs; ignore Nones
    o = {k: v for k, v in override.items() if v is not None}
    return base.model_copy(update=o)
