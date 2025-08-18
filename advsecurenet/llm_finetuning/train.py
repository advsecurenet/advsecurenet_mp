# llm_finetune/train.py
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling
from advsecurenet.llm_finetuning.config import Config
from advsecurenet.llm_finetuning.model import load_model, load_tokenizer
from advsecurenet.llm_finetuning.data import load_tokenized_datasets

import torch, random, numpy as np

def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def run_training(cfg: Config):
    _set_seed(cfg.train.seed)
    tok = load_tokenizer(cfg)
    dsets = load_tokenized_datasets(cfg.data, tok)
    model = load_model(cfg)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tok,
        mlm=False
    )

    args = TrainingArguments(
        output_dir=cfg.train.output_dir,
        learning_rate=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
        max_steps=cfg.train.max_steps,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        logging_steps=cfg.train.logging_steps,
        save_steps=cfg.train.save_steps,
        eval_steps=cfg.train.eval_steps,
        eval_strategy=cfg.train.evaluation_strategy,
        lr_scheduler_type=cfg.train.lr_scheduler_type,
        warmup_ratio=cfg.train.warmup_ratio,
        bf16=cfg.train.bf16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        push_to_hub=cfg.train.push_to_hub,
        report_to=["none"],  # swap to "wandb" if you wire it
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dsets["train"],
        eval_dataset=dsets.get("validation"),
        tokenizer=tok,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_model(cfg.train.output_dir)
    tok.save_pretrained(cfg.train.output_dir)
    return {"output_dir": cfg.train.output_dir}

if __name__ == "__main__":
    from advsecurenet.llm_finetuning.config import load_yaml
    import pathlib

    # Example path to your config file
    cfg_path = "advsecurenet_mp/advsecurenet/llm_finetuning/configs/instruction.yaml"

    cfg = load_yaml(str(cfg_path))
    run_training(cfg)
