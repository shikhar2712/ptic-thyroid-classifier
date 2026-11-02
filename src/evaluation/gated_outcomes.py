"""System-level view: gate decisions (classifier-independent) + a model's stress-test predictions -> what the demo app would do.

Per condition: share of inputs answered / refused by the gate / "uncertain" (0.2 < P < 0.8), and wrong / confidently wrong answers per
input (wrong = answer contradicts the label; confidently wrong = wrong with P < 0.1 or P > 0.9). Validation partitions only.

  python -I src/evaluation/gated_outcomes.py --gate v3 --model-tag e1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import PROJECT_ROOT, write_json  # noqa: E402

BAND = (0.2, 0.8)


def outcomes(gate: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    ens = preds.groupby(["condition", "fold", "image_id", "label"], as_index=False).p.mean()
    J = gate.merge(ens, on=["condition", "fold", "image_id", "label"], how="inner")
    rows = []
    for name, g in J.groupby("condition", sort=False):
        y, p, acc = g.label.to_numpy(), g.p.to_numpy(), g.accepted.to_numpy().astype(bool)
        band = (p > BAND[0]) & (p < BAND[1])
        answered = acc & ~band
        wrong = (p >= 0.5).astype(int) != y
        conf_wrong = ((y == 1) & (p < 0.1)) | ((y == 0) & (p > 0.9))
        rows.append({"condition": name, "n": int(len(g)),
                     "no_gate_wrong_per_input": float(wrong.mean()), "no_gate_confidently_wrong_per_input": float(conf_wrong.mean()),
                     "answered": float(answered.mean()), "refused_by_gate": float((~acc).mean()), "uncertain": float((acc & band).mean()),
                     "wrong_answers_per_input": float((wrong & answered).mean()), "confidently_wrong_per_input": float((conf_wrong & answered).mean()),
                     "error_rate_among_answered": float(wrong[answered].mean()) if answered.any() else float("nan")})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--gate", default="v3"); ap.add_argument("--model-tag", default="e1")
    a = ap.parse_args()
    gate = pd.read_csv(PROJECT_ROOT / "reports" / f"07_gate_decisions_{a.gate}.csv.gz")
    preds = pd.read_csv(PROJECT_ROOT / "reports" / f"07_stress_{a.model_tag}_predictions.csv.gz")
    O = outcomes(gate, preds)
    write_json(PROJECT_ROOT / "reports" / f"07_system_{a.model_tag}_gate_{a.gate}.json",
               {"scope": "validation partitions only; gate decisions joined with stress-test predictions", "answer_band": list(BAND),
                "per_condition": O.round(4).to_dict("records"),
                "summary_all_stress_conditions": {"no_gate_confidently_wrong_mean": round(float(O.no_gate_confidently_wrong_per_input.mean()), 4),
                                                  "gated_confidently_wrong_mean": round(float(O.confidently_wrong_per_input.mean()), 4),
                                                  "gated_wrong_mean": round(float(O.wrong_answers_per_input.mean()), 4),
                                                  "answered_mean": round(float(O.answered.mean()), 4)}})
    pd.set_option("display.width", 220)
    print(O.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
