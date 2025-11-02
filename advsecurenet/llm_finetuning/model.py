from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from advsecurenet.llm_finetuning.config import Config
import os
import torch


def load_tokenizer(cfg: Config):
    """Load and configure a tokenizer for the specified model.

    Args:
        cfg (Config): Configuration object containing model settings.

    Returns:
        AutoTokenizer: Configured tokenizer with proper padding settings.
    """
    tok = AutoTokenizer.from_pretrained(cfg.train.model_name, use_fast=True)
    if (
        getattr(tok, "pad_token", None) is None
        and getattr(tok, "eos_token", None) is not None
    ):
        tok.pad_token = tok.eos_token
        if (
            getattr(tok, "pad_token_id", None) in (None, -1)
            and getattr(tok, "eos_token_id", None) is not None
        ):
            tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "right"
    return tok


def _in_distributed() -> bool:
    """Detect if we're in a distributed/multi-process context (Accelerate/DDP).
    
    Returns:
        bool: True if running in distributed mode, False otherwise.
    """
    if int(os.environ.get("WORLD_SIZE", "1")) > 1 or "LOCAL_RANK" in os.environ:
        return True
    try:
        import torch.distributed as dist  # local import to avoid hard dependency at import time

        return dist.is_available() and dist.is_initialized()
    except Exception:
        return False


def _select_device_map():
    """Select appropriate device mapping strategy for model loading.

    Use 'auto' only in single-process mode. In distributed training, 
    let Accelerator/Trainer handle device placement.

    Returns:
        str | None: Device mapping strategy ("auto", "cpu", or None for distributed).
    """
    if _in_distributed():
        return None
    return "auto" if torch.cuda.is_available() else "cpu"


def load_model(cfg: Config):
    """Load and configure a language model for fine-tuning.

    Handles model loading with optional quantization and PEFT configuration.
    Supports both full fine-tuning and parameter-efficient methods like LoRA.

    Args:
        cfg (Config): Configuration object with model and PEFT settings.

    Returns:
        torch.nn.Module: Configured model ready for training, optionally wrapped with PEFT adapters.
    """
    use_4bit_quantization = cfg.peft.enabled and (cfg.peft.quantization in {"qlora", "bnb-4bit"})
    model = AutoModelForCausalLM.from_pretrained(
        cfg.train.model_name,
        device_map=_select_device_map(),
        load_in_4bit=use_4bit_quantization,
    )

    if cfg.peft.enabled:
        if use_4bit_quantization:
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

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    return model
