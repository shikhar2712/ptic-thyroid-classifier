"""Run one configuration over several seeds x folds in a single process (the image store is decoded once).

Only the 'train' and 'val' partitions are used (see trainer.fit); no outer-test image is scored here.
Finished runs write done.json and are skipped on re-invocation; interrupted runs resume from last.pt.
Gated behind --confirm (training needs the owner's approval).

  python -I src/training/run_experiment.py --config e1_residual.yaml --seeds 0 1 2 --confirm
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import pandas as pd  # noqa: E402

from src.common import limit_cpu_threads, PROJECT_ROOT, get_logger, load_config, resolve, write_json  # noqa: E402
from src.data.dataset import ImageStore  # noqa: E402
from src.training.trainer import fit  # noqa: E402
from src.training.utils import load_train_config  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="e1_residual.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--confirm", action="store_true")
    args = ap.parse_args()
    limit_cpu_threads()
    base = load_train_config(args.config)
    plan = [(s, f) for s in args.seeds for f in args.folds]
    print(f"config={args.config} model={base['model']['name']} runs={len(plan)} (seeds {args.seeds} x folds {args.folds}) epochs<={base['optim']['epochs']}")
    if not args.confirm:
        print("DRY RUN: nothing started. Re-run with --confirm.")
        return 0
    dcfg = load_config("data.yaml")
    man = pd.read_csv(resolve(dcfg["paths"]["metadata"]) / "manifest.csv", keep_default_na=False, na_values=[])
    folds = pd.read_csv(PROJECT_ROOT / base["split_file"])
    man = man[man.image_id.isin(folds.image_id)].reset_index(drop=True)
    log = get_logger(f"exp.{base['output']['run_name']}", PROJECT_ROOT / "logs" / f"{base['output']['run_name']}_runner.log")
    t0 = time.time()
    store = ImageStore.from_manifest(man, resolve(dcfg["paths"]["raw_images"]))
    log.info("store ready: %d images in %.0fs", len(store), time.time() - t0)
    for seed, fold in plan:
        cfg = copy.deepcopy(base)
        cfg["seed"], cfg["fold"] = seed, fold
        name = f"{base['output']['run_name']}_s{seed}"
        run_dir = PROJECT_ROOT / base["output"]["root"] / name / f"fold{fold}"
        if (run_dir / "done.json").exists():
            log.info("skip %s fold %d (done.json present)", name, fold)
            continue
        t1 = time.time()
        rlog = get_logger(f"train.{name}.fold{fold}", PROJECT_ROOT / "logs" / f"{name}_fold{fold}.log")
        summary = fit(cfg, store, man, folds, run_dir, rlog)
        write_json(run_dir / "done.json", {"summary": summary, "seconds": round(time.time() - t1, 1)})
        log.info("DONE %s fold %d in %.0fs | best val AUROC %.4f at epoch %s", name, fold, time.time() - t1,
                 summary.get("best_val_auroc", float("nan")), summary.get("best_epoch"))
    log.info("all runs finished in %.1f min", (time.time() - t0) / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
