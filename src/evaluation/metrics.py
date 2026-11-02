"""Binary classification metrics and patient-cluster bootstrap. Positive class = PTC-like (label 1)."""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """y in {0,1}; p = predicted probability of class 1. AUROC/AUPRC are NaN if only one class is present."""
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    tp, fp = int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum())
    tn, fn = int(((pred == 0) & (y == 0)).sum()), int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: float(a / b) if b else float("nan")  # noqa: E731
    sens, spec, prec = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)
    both = len(np.unique(y)) == 2
    return {
        "n": int(len(y)), "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "sensitivity": sens, "specificity": spec, "precision": prec,
        "f1": div(2 * tp, 2 * tp + fp + fn), "accuracy": div(tp + tn, len(y)),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
        "auprc": float(average_precision_score(y, p)) if both else float("nan"),
        "brier": float(np.mean((p - y) ** 2)), "ece": expected_calibration_error(y, p),
        "threshold": float(threshold),
    }


def cluster_bootstrap(y: np.ndarray, p: np.ndarray, groups: np.ndarray, metric: Callable[[np.ndarray, np.ndarray], float],
                      n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> dict[str, float]:
    """Percentile CI from resampling *groups* (patients) with replacement, not images.

    Replicates for which `metric` is undefined (e.g. a single class for AUROC) are skipped and counted.
    """
    y, p, groups = np.asarray(y), np.asarray(p), np.asarray(groups)
    uniq, inv = np.unique(groups, return_inverse=True)
    members = [np.where(inv == g)[0] for g in range(len(uniq))]
    rng = np.random.default_rng(seed)
    vals, skipped = [], 0
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([members[g] for g in pick])
        v = metric(y[idx], p[idx])
        if np.isfinite(v):
            vals.append(v)
        else:
            skipped += 1
    if not vals:
        return {"lo": float("nan"), "hi": float("nan"), "n_valid": 0, "n_skipped": skipped}
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return {"lo": float(lo), "hi": float(hi), "n_valid": len(vals), "n_skipped": skipped}
