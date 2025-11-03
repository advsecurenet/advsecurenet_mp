from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase
from .config import DataConfig


def _format_example_to_text(example: dict, cfg: DataConfig) -> dict:
    """Convert dataset examples to text format for language modeling.

    Args:
        example (dict): Raw example from the dataset.
        cfg (DataConfig): Configuration specifying field mappings and formatting.

    Returns:
        dict: Formatted example with 'text' field ready for tokenization.
    """
    if cfg.prompt_field and cfg.response_field:
        return {"text": f"{example[cfg.prompt_field]}\n{example[cfg.response_field]}"}
    if cfg.text_field:
        return {"text": example[cfg.text_field]}
    # Hub presets:
    if getattr(cfg, "hub_name", None) == "gsm8k":
        return {"text": f"Q: {example['question']}\nA: {example['answer']}"}
    return {"text": str(example)}


def _tokenize_batch(
    batch: dict, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> dict:
    """Tokenize text batch with truncation.

    Args:
        batch (dict): Batch of examples with 'text' field.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for text processing.
        max_length (int): Maximum sequence length for truncation.

    Returns:
        dict: Tokenized batch with input_ids, attention_mask, etc.
    """
    return tokenizer(batch["text"], truncation=True, max_length=max_length)


def load_tokenized_datasets(
    cfg: DataConfig, tokenizer: PreTrainedTokenizerBase
) -> DatasetDict:
    """Load and tokenize datasets for language model fine-tuning.

    Supports local JSONL files and HuggingFace Hub datasets with automatic
    text formatting and tokenization.

    Args:
        cfg (DataConfig): Dataset configuration specifying source and preprocessing options.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for text processing.

    Returns:
        DatasetDict: Tokenized datasets with 'train' and optionally 'validation' splits.
    """
    # Load raw datasets
    if cfg.train_file and cfg.train_file.endswith(".jsonl"):
        files = {"train": cfg.train_file}
        if cfg.eval_file:
            files["validation"] = cfg.eval_file
        raw = load_dataset("json", data_files=files)
    elif getattr(cfg, "hub_name", None):
        # e.g., hub_name="gsm8k", hub_config="main", hub_train_split="train", hub_eval_split="test"
        raw = DatasetDict()
        raw["train"] = load_dataset(
            cfg.hub_name, cfg.hub_config or None, split=cfg.hub_train_split or "train"
        )
        if cfg.hub_eval_split:
            raw["validation"] = load_dataset(
                cfg.hub_name, cfg.hub_config or None, split=cfg.hub_eval_split
            )
    else:
        raise ValueError(
            "Provide either JSONL files or a hub dataset via data.hub_name"
        )

    # Apply text formatting using the dedicated function
    formatted = raw.map(lambda ex: _format_example_to_text(ex, cfg))

    # Apply tokenization using the dedicated function
    tokenized = formatted.map(
        lambda batch: _tokenize_batch(batch, tokenizer, cfg.max_seq_len),
        batched=True,
        remove_columns=formatted["train"].column_names,
    )

    return tokenized
