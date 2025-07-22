from datasets import load_dataset, DatasetDict, Dataset
from torch.utils.data import Dataset as TorchDataset

class TextDataset(TorchDataset):
    def __init__(self, hf_dataset, tokenizer, text_column, label_column, max_length=128):
        self.encodings = tokenizer(
            hf_dataset[text_column],
            truncation=True,
            padding=True,
            max_length=max_length
        )
        self.labels = hf_dataset[label_column]

    def __getitem__(self, idx):
        return {
            **{k: v[idx] for k, v in self.encodings.items()},
            "labels": self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

class DataLoaderWrapper:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def load_data(
        self,
        dataset_input,
        text_column: str,
        label_column: str,
        split_keys=("train", "test")
    ):
        # Accept either a string path or DatasetDict
        if isinstance(dataset_input, str):
            dataset = load_dataset(dataset_input)
        elif isinstance(dataset_input, DatasetDict):
            dataset = dataset_input
        else:
            raise ValueError("Invalid dataset_input type. Must be str or DatasetDict.")

        train_data = dataset[split_keys[0]]
        eval_data = dataset[split_keys[1]]

        train_dataset = TextDataset(train_data, self.tokenizer, text_column, label_column)
        eval_dataset = TextDataset(eval_data, self.tokenizer, text_column, label_column)
        return train_dataset, eval_dataset
