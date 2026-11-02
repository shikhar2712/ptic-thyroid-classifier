"""Makes `src` importable when pytest is run from the project root (also under `python -I -m pytest`).

Also fixes the CPU thread count before torch is imported: changing it later at run time can crash CPU ops in this PyTorch build
(see src.common.limit_cpu_threads)."""
import os
import sys
from pathlib import Path

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, os.environ.get("PTIC_TORCH_THREADS", "8"))
assert "torch" not in sys.modules, "torch was imported before conftest.py could set the thread count"

sys.path.insert(0, str(Path(__file__).resolve().parent))
