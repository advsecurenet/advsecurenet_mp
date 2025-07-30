from datasets import load_dataset
from transformers import PreTrainedTokenizerBase
from typing import Tuple, Optional


class HuggingFaceDataLoader:
    """
    A simplified data loader for loading datasets from HuggingFace and preparing them for training.
    """

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int = 128):
        """
        Initialize the data loader.

        Args:
            tokenizer (PreTrainedTokenizerBase): The tokenizer to preprocess the text data.
            max_length (int): Maximum sequence length for tokenization.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, dataset_name: str, text_column: str, label_column: str, split_keys: Tuple[str, str] = ("train", "test")) -> Tuple:
        """
        Load and preprocess a dataset from HuggingFace.

        Args:
            dataset_name (str): The name of the dataset to load from HuggingFace.
            text_column (str): The column containing the text data.
            label_column (str): The column containing the labels.
            split_keys (Tuple[str, str]): The dataset splits to use (e.g., "train" and "test").

        Returns:
            Tuple: A tuple containing the tokenized train and validation datasets.
        """
        # Load the dataset from HuggingFace
        dataset = load_dataset(dataset_name)
        train_split, val_split = split_keys

        # Tokenize the dataset
        train_dataset = self._tokenize_dataset(dataset[train_split], text_column, label_column)
        val_dataset = self._tokenize_dataset(dataset[val_split], text_column, label_column)

        return train_dataset, val_dataset

    def _tokenize_dataset(self, dataset, text_column: str, label_column: str):
        """
        Tokenize the dataset.

        Args:
            dataset: The dataset to tokenize.
            text_column (str): The column containing the text data.
            label_column (str): The column containing the labels.

        Returns:
            The tokenized dataset.
        """
        def tokenize_function(examples):
            # Tokenize the text and align the labels
            return self.tokenizer(
                examples[text_column],
                truncation=True,
                padding="max_length",
            )

        # Apply tokenization
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Add labels to the tokenized dataset
        tokenized_dataset = tokenized_dataset.rename_column(label_column, "labels")
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        return tokenized_dataset