"""Apply the step-2 replacement rule declared in reports/07_robustness.md section 1.3 (before any stress-test result existed).

E3 replaces E1 as the deployment candidate only if ALL hold (validation partitions, stress-test conventions):
  1. clean: mean per-fold AUROC >= 0.99 and clean balanced accuracy at 0.5 >= E1's clean value - 0.01
  2. held-out robustness: mean balanced accuracy over the 13 held-out conditions >= E1's + 0.03
  3. no held-out condition worse than E1 by more than 0.03 balanced accuracy
Trained-kind conditions are reported separately for information only (they improve by construction inside the training range).

  python -I src/evaluation/robustness_decision.py --a e1 --b e3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import pandas as pd  # noqa: E402

from src.common import PROJECT_ROOT, write_json  # noqa: E402

CLEAN_AUROC_MIN, CLEAN_MARGIN, HELD_GAIN_MIN, HELD_WORST_MAX = 0.99, 0.01, 0.03, 0.03


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--a", default="e1"); ap.add_argument("--b", default="e3")
    a = ap.parse_args()
    A = pd.DataFrame(json.load(open(PROJECT_ROOT / "reports" / f"07_stress_{a.a}.json"))["conditions"]).set_index("condition")
    B = pd.DataFrame(json.load(open(PROJECT_ROOT / "reports" / f"07_stress_{a.b}.json"))["conditions"]).set_index("condition")
    assert list(A.index) == list(B.index) and (A.status == B.status).all(), "both stress tests must use the same declared conditions"
    held = A.index[A.status == "held_out"]
    assert len(held) == 13
    d = (B.balanced_accuracy - A.balanced_accuracy)
    r1 = bool(B.loc["clean", "mean_fold_auroc"] >= CLEAN_AUROC_MIN and B.loc["clean", "balanced_accuracy"] >= A.loc["clean", "balanced_accuracy"] - CLEAN_MARGIN)
    r2 = bool(B.loc[held, "balanced_accuracy"].mean() >= A.loc[held, "balanced_accuracy"].mean() + HELD_GAIN_MIN)
    r3 = bool((d[held] >= -HELD_WORST_MAX).all())
    out = {
        "rule_source": "reports/07_robustness.md section 1.3 (declared before any stress-test result existed)",
        "reference": a.a, "candidate": a.b,
        "rule_1_clean_kept": {"pass": r1, f"{a.b}_clean_mean_fold_auroc": round(float(B.loc['clean', 'mean_fold_auroc']), 4),
                              f"{a.a}_clean_bal_acc": round(float(A.loc['clean', 'balanced_accuracy']), 4), f"{a.b}_clean_bal_acc": round(float(B.loc['clean', 'balanced_accuracy']), 4)},
        "rule_2_held_out_gain": {"pass": r2, f"{a.a}_held_out_mean_bal_acc": round(float(A.loc[held, 'balanced_accuracy'].mean()), 4),
                                 f"{a.b}_held_out_mean_bal_acc": round(float(B.loc[held, 'balanced_accuracy'].mean()), 4), "required_gain": HELD_GAIN_MIN},
        "rule_3_no_held_out_condition_worse": {"pass": r3, "worst_change": round(float(d[held].min()), 4), "worst_condition": str(d[held].idxmin())},
        "DECISION": f"{a.b} replaces {a.a}" if (r1 and r2 and r3) else f"{a.a} stays",
        "per_condition": pd.DataFrame({"status": A.status, f"{a.a}_bal_acc": A.balanced_accuracy, f"{a.b}_bal_acc": B.balanced_accuracy, "change": d,
                                       f"{a.a}_confident_wrong": A.confident_wrong_rate, f"{a.b}_confident_wrong": B.confident_wrong_rate}).round(4).reset_index().to_dict("records"),
        "trained_kinds_info_only": {st: round(float(d[A.status == st].mean()), 4) for st in ("trained_in", "trained_beyond")},
    }
    write_json(PROJECT_ROOT / "reports" / f"07_decision_{a.b}_vs_{a.a}.json", out)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(out["per_condition"]).to_string(index=False))
    print(json.dumps({k: out[k] for k in ("rule_1_clean_kept", "rule_2_held_out_gain", "rule_3_no_held_out_condition_worse", "trained_kinds_info_only")}, indent=1))
    print("DECISION:", out["DECISION"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
