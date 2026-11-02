"""Diagnostic: how well do colour / contrast statistics alone separate the classes on unseen patients?

Scores ONLY each fold's validation partition (fit on that fold's training partition). The outer test folds are not
read. This is a data diagnostic, not an experimental result for the final report.
Run:  python -I src/evaluation/shortcut_check.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.common import PROJECT_ROOT, load_config, resolve, write_json  # noqa: E402
from src.evaluation.metrics import binary_metrics  # noqa: E402
from src.models import colour_baseline  # noqa: E402

UNIVARIATE = colour_baseline.FEATURES + ["lap_var"]


def main() -> int:
    cfg = load_config("data.yaml")
    man = pd.read_csv(resolve(cfg["paths"]["metadata"]) / "manifest.csv", keep_default_na=False, na_values=[]).set_index("image_id")
    folds = pd.read_csv(PROJECT_ROOT / "data/splits/folds_v1.csv")
    rows, uni = [], []
    for f in range(5):
        role = folds.set_index("image_id")[f"role_fold{f}"]
        tr, va = man.loc[role.index[role == "train"]], man.loc[role.index[role == "val"]]   # 'test' is never indexed
        for name, feats in colour_baseline.BASELINES.items():
            m = binary_metrics(va.label.to_numpy(), colour_baseline.fit_predict(tr, va, features=feats))
            rows.append({"baseline": name, "fold": f, "val_patients": int(va.patient_id.nunique()), "val_images": len(va),
                         "auroc": round(m["auroc"], 4), "balanced_accuracy": round(m["balanced_accuracy"], 4),
                         "sensitivity": round(m["sensitivity"], 4), "specificity": round(m["specificity"], 4)})
        for ft in UNIVARIATE:   # direction-free univariate AUROC: max(a, 1-a)
            a = roc_auc_score(va.label, va[ft])
            uni.append({"fold": f, "feature": ft, "auroc_direction_free": round(max(a, 1 - a), 3)})
    uni_df = pd.DataFrame(uni).pivot(index="feature", columns="fold", values="auroc_direction_free")
    uni_df["mean"] = uni_df.mean(axis=1).round(3)
    uni_df.columns = [f"fold{c}" if isinstance(c, (int, np.integer)) else c for c in uni_df.columns]  # JSON keys must sort as strings
    out = {"scope": "VALIDATION partitions only (outer test folds not read); fit on each fold's training partition",
           "per_fold": rows,
           "summary": {n: {"auroc_mean": round(float(np.mean([r["auroc"] for r in rows if r["baseline"] == n])), 3),
                           "auroc_min": min(r["auroc"] for r in rows if r["baseline"] == n),
                           "auroc_max": max(r["auroc"] for r in rows if r["baseline"] == n)} for n in colour_baseline.BASELINES},
           "features": colour_baseline.BASELINES, "univariate_auroc_val": uni_df.sort_values("mean", ascending=False).to_dict("index")}
    write_json(PROJECT_ROOT / "reports" / "shortcut_check_validation_only.json", out)
    print(pd.DataFrame(rows).pivot(index="fold", columns="baseline", values="auroc").to_string()); print("\n", json.dumps(out["summary"], indent=1))
    print("\nunivariate (direction-free) AUROC on validation partitions:"); print(uni_df.sort_values("mean", ascending=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
