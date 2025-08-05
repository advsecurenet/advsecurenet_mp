from datasets import load_dataset
from advsecurenet.llm_RLHF_finetuning.config import DataConfig

class DataLoader:
    def __init__(self, config: DataConfig):
        self.config = config

    def load(self):
        dataset = load_dataset(self.config.dataset_name, name=self.config.dataset_config)
        train_data = dataset[self.config.train_split]
        test_data = dataset[self.config.test_split] if self.config.test_split in dataset else None
        return train_data, test_data
