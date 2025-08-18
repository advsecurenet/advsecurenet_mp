# llm_finetune/utils.py
from __future__ import annotations
import os, json, time, functools, contextlib, random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch

# ---------- Repro & distributed ----------

def set_seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def dist_info() -> dict:
    """Return world_size/rank/local_rank even if torch.distributed isn't initialized."""
    info = {
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
    }
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            info["world_size"] = dist.get_world_size()
            info["rank"] = dist.get_rank()
            info["local_rank"] = int(os.environ.get("LOCAL_RANK", info["rank"]))
    except Exception:
        pass
    return info

def is_rank_zero() -> bool:
    return dist_info()["rank"] == 0

def rank_zero_only(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if is_rank_zero():
            return fn(*args, **kwargs)
    return wrapper

# ---------- Device / dtype / capability ----------

def bf16_supported() -> bool:
    """True if bfloat16 training is supported on this GPU."""
    if not torch.cuda.is_available():
        return False
    cc_major, _ = torch.cuda.get_device_capability()
    # Ampere (8.0+) or newer have native bf16
    return cc_major >= 8

def default_dtype(prefer_bf16: bool = True) -> torch.dtype:
    if prefer_bf16 and bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32

def device_map_auto() -> str | dict:
    """Return a safe default device map string for HF .from_pretrained."""
    return "auto" if torch.cuda.is_available() else "cpu"

# ---------- Tokenizer / model small fixes ----------

def ensure_padding_token(tokenizer) -> None:
    """Set pad_token to eos_token if missing (common for causal LMs)."""
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

def lora_default_targets(model) -> Iterable[str]:
    """
    Heuristic target modules for LoRA based on common LLMs.
    Falls back to q_proj/v_proj if present.
    """
    names = [n for n, _ in model.named_modules()]
    # common patterns
    candidates = [
        "q_proj", "v_proj", "k_proj", "o_proj",
        "Wqkv", "W_pack", "query_key_value",
        "attn.c_proj", "attn.q_proj", "attn.v_proj",
    ]
    found = {n for n in names for c in candidates if n.endswith(c)}
    return sorted(found) if found else ["q_proj", "v_proj"]

# ---------- I/O & logging ----------

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

@rank_zero_only
def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

@rank_zero_only
def save_run_config(cfg_obj, out_dir: str | Path) -> None:
    """Persist the exact config used for reproducibility."""
    from pydantic import BaseModel
    out = Path(out_dir)
    ensure_dir(out)
    if isinstance(cfg_obj, BaseModel):
        data = cfg_obj.model_dump()
    else:
        # fallback if someone passes a dict
        data = dict(cfg_obj)
    save_json(data, out / "config.resolved.json")

def resolve_path(base: str | Path, maybe_rel: str | Path | None) -> Optional[Path]:
    if maybe_rel is None:
        return None
    p = Path(maybe_rel)
    return p if p.is_absolute() else Path(base) / p

# ---------- Profiling & diagnostics ----------

@contextlib.contextmanager
def time_block(label: str):
    t0 = time.time()
    yield
    dt = time.time() - t0
    if is_rank_zero():
        print(f"[timing] {label}: {dt:.2f}s")

def cuda_mem() -> Dict[str, float]:
    """Current and max allocated GPU memory in GB (rank-local)."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "max_allocated_gb": 0.0}
    alloc = torch.cuda.memory_allocated() / (1024**3)
    max_alloc = torch.cuda.max_memory_allocated() / (1024**3)
    return {"allocated_gb": round(alloc, 3), "max_allocated_gb": round(max_alloc, 3)}

def count_trainable_params(model) -> Tuple[int, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / total if total else 0.0
    return trainable, pct

# ---------- Safety checks ----------

def require_bitsandbytes_if_needed(qlora_enabled: bool) -> None:
    if qlora_enabled:
        try:
            import bitsandbytes  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "QLoRA requested but bitsandbytes is not installed. "
                "Install with: pip install bitsandbytes"
            ) from e

def maybe_torch_compile(model, enabled: bool = False):
    """Optional PyTorch 2.x compile gate."""
    if enabled and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[attr-defined]
        except Exception:
            # Fallback silently; compile can be finicky depending on ops
            pass
    return model
