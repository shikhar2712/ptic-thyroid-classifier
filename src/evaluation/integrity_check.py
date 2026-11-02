"""Post-hoc integrity check of a finished experiment: did every run train and validate on exactly the patients the fold file says,
with nothing from the fold's test patients and no train/val overlap? Reads only run_info.json files and the fold table.

  python -I src/evaluation/integrity_check.py --run e1_residual --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import pandas as pd  # noqa: E402

from src.common import PROJECT_ROOT, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e1_residual")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tag", default="e1", help="output: reports/<tag>_integrity_check.json")
    a = ap.parse_args()
    folds = pd.read_csv(PROJECT_ROOT / "data/splits/folds_v1.csv")
    out, problems = [], []
    for s in a.seeds:
        for f in range(5):
            info = json.load(open(PROJECT_ROOT / "checkpoints" / f"{a.run}_s{s}" / f"fold{f}" / "run_info.json"))
            tr, va = set(info["train_patient_ids"]), set(info["val_patient_ids"])
            role = folds.groupby("patient_id")[f"role_fold{f}"].first()
            exp = {r: set(role.index[role == r]) for r in ("train", "val", "test")}
            checks = {"train_matches_fold_file": tr == exp["train"], "val_matches_fold_file": va == exp["val"],
                      "train_val_disjoint": not (tr & va), "no_test_patient_in_train": not (tr & exp["test"]),
                      "no_test_patient_in_val": not (va & exp["test"]),
                      "images_match": info["partition_sizes"]["train_images"] == int((folds[f"role_fold{f}"] == "train").sum())
                                      and info["partition_sizes"]["val_images"] == int((folds[f"role_fold{f}"] == "val").sum()),
                      "config_fold_and_seed": info["config"]["fold"] == f and info["config"]["seed"] == s}
            out.append({"seed": s, "fold": f, **checks})
            problems += [(s, f, k) for k, v in checks.items() if not v]
    write_json(PROJECT_ROOT / "reports" / f"{a.tag}_integrity_check.json", {"runs_checked": len(out), "problems": problems, "detail": out})
    print(f"runs checked: {len(out)} | problems: {problems if problems else 'none'}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
