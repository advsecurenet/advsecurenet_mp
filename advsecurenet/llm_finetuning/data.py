from datasets import load_dataset, DatasetDict
from transformers import PreTrainedTokenizerBase
from .config import DataConfig

def load_tokenized_datasets(cfg: DataConfig, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    if cfg.train_file and cfg.train_file.endswith(".jsonl"):
        files = {"train": cfg.train_file}
        if cfg.eval_file:
            files["validation"] = cfg.eval_file
        raw = load_dataset("json", data_files=files)
    elif getattr(cfg, "hub_name", None):
        # e.g., hub_name="gsm8k", hub_config="main", hub_train_split="train", hub_eval_split="test"
        raw = DatasetDict()
        raw["train"] = load_dataset(cfg.hub_name, cfg.hub_config or None, split=cfg.hub_train_split or "train")
        if cfg.hub_eval_split:
            raw["validation"] = load_dataset(cfg.hub_name, cfg.hub_config or None, split=cfg.hub_eval_split)
    else:
        raise ValueError("Provide either JSONL files or a hub dataset via data.hub_name")

    def to_text(ex):
        if cfg.prompt_field and cfg.response_field:
            return {"text": f"{ex[cfg.prompt_field]}\n{ex[cfg.response_field]}"}
        if cfg.text_field:
            return {"text": ex[cfg.text_field]}
        # hub presets:
        if getattr(cfg, "hub_name", None) == "gsm8k":
            return {"text": f"Q: {ex['question']}\nA: {ex['answer']}"}
        return {"text": str(ex)}

    raw = raw.map(to_text)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=cfg.max_seq_len)

    tokenized = raw.map(tok, batched=True, remove_columns=raw["train"].column_names)
    return tokenized