from datasets import load_dataset
from transformers import AutoTokenizer
from advsecurenet.llm_RLHF_finetuning.config import DataConfig

class DataLoader:
    def __init__(self, config: DataConfig, tokenizer_name_or_path: str):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # GPT-2 requires this

    def load(self):
        dataset = load_dataset(self.config.dataset_name, name=self.config.dataset_config)
        train_data = dataset[self.config.train_split]
        eval_data = dataset[self.config.test_split] if self.config.test_split in dataset else None
        return train_data, eval_data
