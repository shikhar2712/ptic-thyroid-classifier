"""Shared helpers: project root, config loading, logging, hashing, environment snapshot."""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(name: str = "data.yaml", root: Path = PROJECT_ROOT) -> dict[str, Any]:
    with open(root / "configs" / name) as fh:
        return yaml.safe_load(fh)


def resolve(cfg_path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    p = Path(cfg_path)
    return p if p.is_absolute() else root / p


def get_logger(name: str, log_file: Path | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def file_hash(path: Path, algo: str = "md5", chunk: int = 8 << 20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def bytes_hash(data: bytes, algo: str = "sha256") -> str:
    return hashlib.new(algo, data).hexdigest()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
    tmp.replace(path)


def env_snapshot() -> dict[str, Any]:
    """Package/GPU versions for reproducibility. Never records environment variables."""
    import numpy
    import pandas
    import PIL
    import sklearn

    snap: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pillow": PIL.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import torch
        import torchvision

        snap.update(torch=torch.__version__, torchvision=torchvision.__version__,
                    torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                    cuda_available=torch.cuda.is_available())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            snap.update(gpu=p.name, gpu_mem_gib=round(p.total_memory / 2**30, 2))
    except Exception as exc:  # torch is optional for the pure-data steps
        snap["torch_error"] = repr(exc)
    return snap


def limit_cpu_threads(default: int = 8) -> int:
    """Cap torch's CPU threads (default 8; override with PTIC_TORCH_THREADS). With the default 128 threads per process, several
    concurrent jobs oversubscribe the CPU and tiny tensor copies slow down ~50x (measured: 900 ms vs 19 ms per training batch).
    Affects speed only: random draws and GPU computations are unchanged.

    Prefer starting the process with OMP_NUM_THREADS=8 (tests/conftest and scripts/run_demo_server.sh do): in this PyTorch build
    (2.11.0a0 NVIDIA, GNU OpenMP + MKL), changing the count at run time with torch.set_num_threads can crash later CPU element-wise
    ops (segfault in 9 of 20 processes of a 300-op repro; 0 of 20 with OMP_NUM_THREADS=8). GPU work is not affected, which is why the
    earlier runs completed. So the run-time call is made only when the count is not already right."""
    import os

    import torch

    n = int(os.environ.get("PTIC_TORCH_THREADS", default))
    if torch.get_num_threads() != n:
        torch.set_num_threads(n)
    return n
