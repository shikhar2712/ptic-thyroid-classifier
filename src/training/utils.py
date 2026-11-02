"""Reproducibility, device and precision selection, config merging."""
from __future__ import annotations

import copy
import os
import random
from typing import Any

import numpy as np
import torch
import yaml

from src.common import PROJECT_ROOT


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def select_device(prefer: str = "cuda") -> torch.device:
    return torch.device("cuda" if prefer == "cuda" and torch.cuda.is_available() else "cpu")


def select_amp_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    """'bf16' -> bfloat16 autocast when the GPU supports it; 'fp32' -> no autocast. Falls back to fp32 otherwise."""
    if precision == "bf16" and device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return None


def deep_update(base: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in new.items():
        out[k] = deep_update(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_train_config(name: str, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = yaml.safe_load(open(PROJECT_ROOT / "configs" / "train_default.yaml"))
    if name != "train_default.yaml":
        cfg = deep_update(cfg, yaml.safe_load(open(PROJECT_ROOT / "configs" / name)) or {})
    return deep_update(cfg, overrides or {})


def rng_state() -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s: dict[str, Any]) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
