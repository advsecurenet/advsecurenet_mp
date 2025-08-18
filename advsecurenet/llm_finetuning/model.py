# llm_finetune/model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from advsecurenet.llm_finetuning.config import Config

def load_tokenizer(cfg: Config):
    tok = AutoTokenizer.from_pretrained(cfg.train.model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok

def load_model(cfg: Config):
    q4 = cfg.peft.enabled and (cfg.peft.quantization in {"qlora", "bnb-4bit"})
    model = AutoModelForCausalLM.from_pretrained(
        cfg.train.model_name,
        device_map="auto",
        load_in_4bit=q4,
    )
    if cfg.peft.enabled:
        if q4:
            model = prepare_model_for_kbit_training(model)
        lora = LoraConfig(
            r=cfg.peft.r,
            lora_alpha=cfg.peft.lora_alpha,
            lora_dropout=cfg.peft.lora_dropout,
            target_modules=cfg.peft.target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    return model
