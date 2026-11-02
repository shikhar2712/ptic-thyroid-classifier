"""Command-line entry point for training ONE fold.

A real run is gated behind --confirm because full training requires the project owner's approval
(see README / reports). Without --confirm this prints the plan and exits.

  python -I src/training/train.py --config train_default.yaml --fold 0 --set model.name=small_cnn --confirm
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from src.common import PROJECT_ROOT, get_logger, load_config, resolve  # noqa: E402
from src.data.dataset import ImageStore  # noqa: E402
from src.training.trainer import fit  # noqa: E402
from src.training.utils import load_train_config  # noqa: E402


def _parse_overrides(items: list[str]) -> dict:
    out: dict = {}
    for it in items:
        key, _, val = it.partition("=")
        cur = out
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = yaml.safe_load(val)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="train_default.yaml")
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides, e.g. optim.epochs=5")
    ap.add_argument("--confirm", action="store_true", help="actually start training (requires owner approval)")
    args = ap.parse_args()
    ov = _parse_overrides(args.set)
    if args.fold is not None:
        ov["fold"] = args.fold
    cfg = load_train_config(args.config, ov)
    print(yaml.safe_dump(cfg, sort_keys=False))
    if not args.confirm:
        print("DRY RUN: nothing started. Re-run with --confirm once training has been approved.")
        return 0
    dcfg = load_config("data.yaml")
    man = pd.read_csv(resolve(dcfg["paths"]["metadata"]) / "manifest.csv", keep_default_na=False, na_values=[])
    folds = pd.read_csv(PROJECT_ROOT / cfg["split_file"])
    man = man[man.image_id.isin(folds.image_id)].reset_index(drop=True)
    store = ImageStore.from_manifest(man, resolve(dcfg["paths"]["raw_images"]))
    run_dir = PROJECT_ROOT / cfg["output"]["root"] / cfg["output"]["run_name"] / f"fold{cfg['fold']}"
    log = get_logger(f"train.{cfg['output']['run_name']}.fold{cfg['fold']}", PROJECT_ROOT / cfg["output"]["log_root"] / f"{cfg['output']['run_name']}_fold{cfg['fold']}.log")
    print(fit(cfg, store, man, folds, run_dir, log))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
