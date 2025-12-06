from __future__ import annotations

import os
import json
import time
import functools
import contextlib
import random
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)

# ---------- Repro & distributed ----------


def set_seed_all(seed: int = 42, deterministic: bool = False) -> None:
    """
    Seed Python, NumPy and Torch RNGs. Optionally enable deterministic algorithms
    (when supported) for better reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Best-effort deterministic setup; ignore if unavailable.
            try:
                torch.use_deterministic_algorithms(True)  # type: ignore[attr-defined]
            except Exception:
                pass
            try:
                import torch.backends.cudnn as cudnn  # type: ignore

                cudnn.deterministic = True
                cudnn.benchmark = False
            except Exception:
                pass


def dist_info() -> dict:
    """Return world_size/rank/local_rank even if torch.distributed isn't initialized."""
    info = {
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "rank": int(os.environ.get("RANK", "0")),
        "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
    }
    try:
        import torch.distributed as dist  # type: ignore

        if dist.is_available() and dist.is_initialized():
            info["world_size"] = int(dist.get_world_size())
            info["rank"] = int(dist.get_rank())
            # Prefer env LOCAL_RANK if present; fallback to rank.
            info["local_rank"] = int(os.environ.get("LOCAL_RANK", str(info["rank"])))
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
    """True if bfloat16 training is supported on this GPU (Ampere, major >= 8)."""
    if not torch.cuda.is_available():
        return False
    try:
        dev = torch.cuda.current_device()  # type: ignore[attr-defined]
    except Exception:
        dev = None
    try:
        major, _minor = torch.cuda.get_device_capability(dev)  # type: ignore[attr-defined]
        return int(major) >= 8  # Ampere+
    except Exception:
        return False


def default_dtype(prefer_bf16: bool = True) -> torch.dtype:
    if prefer_bf16 and bf16_supported():
        return torch.bfloat16
    return torch.float16 if torch.cuda.is_available() else torch.float32


def device_map_auto() -> str | dict:
    """Return a safe default device map string for HF .from_pretrained."""
    return "auto" if torch.cuda.is_available() else "cpu"


# ---------- Tokenizer / model small fixes ----------


def ensure_padding_token(tokenizer) -> None:
    """Set pad_token (and id) to eos_token if missing (common for causal LMs)."""
    if (
        getattr(tokenizer, "pad_token", None) is None
        and getattr(tokenizer, "eos_token", None) is not None
    ):
        tokenizer.pad_token = tokenizer.eos_token
        # Also set pad_token_id if available/needed.
        if (
            getattr(tokenizer, "pad_token_id", None) in (None, -1)
            and getattr(tokenizer, "eos_token_id", None) is not None
        ):
            tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"


def lora_default_targets(model) -> Iterable[str]:
    """
    Heuristic target modules for LoRA based on common LLMs.
    Falls back to ["q_proj", "v_proj"] if nothing recognized.
    """
    names = [n for n, _m in model.named_modules()]
    candidates = [
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "Wqkv",
        "W_pack",
        "query_key_value",
        "attn.c_proj",
        "attn.q_proj",
        "attn.v_proj",
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
    """Persist the exact config used for reproducibility to 'config.resolved.json'."""
    from pydantic import BaseModel  # lazy import

    out = Path(out_dir)
    ensure_dir(out)
    if isinstance(cfg_obj, BaseModel):
        data = cfg_obj.model_dump()
    else:
        # fallback if someone passes a dict-like
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
    try:
        yield
    finally:
        dt = time.time() - t0
        if is_rank_zero():
            log.info("[timing] %s: %.2fs", label, dt)


def cuda_mem() -> Dict[str, float]:
    """Current and max allocated GPU memory in GB (rank-local)."""
    if not torch.cuda.is_available():
        return {"allocated_gb": 0.0, "max_allocated_gb": 0.0}
    try:
        alloc = float(torch.cuda.memory_allocated()) / (1024**3)  # type: ignore[attr-defined]
        max_alloc = float(torch.cuda.max_memory_allocated()) / (1024**3)  # type: ignore[attr-defined]
    except Exception:
        return {"allocated_gb": 0.0, "max_allocated_gb": 0.0}
    return {"allocated_gb": round(alloc, 3), "max_allocated_gb": round(max_alloc, 3)}


def count_trainable_params(model) -> Tuple[int, float]:
    total = int(sum(int(p.numel()) for p in model.parameters()))
    trainable = int(
        sum(
            int(p.numel())
            for p in model.parameters()
            if getattr(p, "requires_grad", False)
        )
    )
    pct = 100.0 * trainable / total if total else 0.0
    return trainable, pct
