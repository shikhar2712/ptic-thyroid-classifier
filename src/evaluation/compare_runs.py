"""Paired per-fold comparison of two finished experiments' VALIDATION results, and E2's declared reading rules.

Inputs are the `<tag>_validation_results.json` files written by oof_validation.py (validation partitions only).
The rules below were declared in reports/05_e2_lowres.md before E2 was launched; do not edit them after results exist.

  python -I src/evaluation/compare_runs.py --a e1 --b e2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import PROJECT_ROOT, write_json  # noqa: E402

# Declared before launch (reports/05_e2_lowres.md). drop = (a) - (b) per fold, AUROC seed-averaged, epoch-29 checkpoints.
R1_MIN_B = 0.99        # R1 "separable at ~1 um/px": mean per-fold AUROC of (b) >= 0.99 ...
R1_MAX_DROP = 0.01     # ... and mean drop <= 0.01
R2_MIN_DROP = 0.02     # R2 "the change matters": mean drop >= 0.02 ...
R2_FOLD_MARGIN = 0.005 # ... and drop >= 0.005 (not a ceiling tie) ...
R2_MIN_FOLDS = 4       # ... in at least 4 of 5 folds. Otherwise: INCONCLUSIVE.


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="e1", help="reference run tag")
    ap.add_argument("--b", default="e2", help="comparison run tag")
    args = ap.parse_args()
    A = json.load(open(PROJECT_ROOT / "reports" / f"{args.a}_validation_results.json"))
    B = json.load(open(PROJECT_ROOT / "reports" / f"{args.b}_validation_results.json"))
    for X, t in ((A, args.a), (B, args.b)):
        assert X["checkpoint"] == "last", f"{t}: rules are declared for epoch-29 (last) checkpoints"
    pa, pb = pd.DataFrame(A["per_fold"]).set_index("fold"), pd.DataFrame(B["per_fold"]).set_index("fold")
    base_cols = ["B0_colour", "B1_texture", "B2_colour_texture"]
    baselines_identical = bool(np.allclose(pa[base_cols].to_numpy(), pb[base_cols].to_numpy(), atol=1e-9))   # sanity: baselines do not depend on the CNN run
    rows_identical = bool((pa[["val_patients", "n", "n_ptc"]] == pb[["val_patients", "n", "n_ptc"]]).all().all())
    T = pd.DataFrame({f"{args.a}_cnn": pa.cnn, f"{args.b}_cnn": pb.cnn, "B2": pa.B2_colour_texture,
                      f"{args.b}_seed_min": pb.cnn_seed_min, f"{args.b}_seed_max": pb.cnn_seed_max})
    T["drop"] = T[f"{args.a}_cnn"] - T[f"{args.b}_cnn"]
    T[f"{args.b}_minus_B2"] = T[f"{args.b}_cnn"] - T.B2
    mean_b, mean_drop = float(T[f"{args.b}_cnn"].mean()), float(T["drop"].mean())
    folds_dropped = int((T["drop"] >= R2_FOLD_MARGIN).sum())
    r1 = mean_b >= R1_MIN_B and mean_drop <= R1_MAX_DROP
    r2 = mean_drop >= R2_MIN_DROP and folds_dropped >= R2_MIN_FOLDS
    outcome = "R1" if r1 else ("R2" if r2 else "INCONCLUSIVE")
    out = {
        "scope": "validation partitions only; epoch-29 checkpoints; 3 seeds averaged per fold; outer test folds not scored",
        "reference": args.a, "comparison": args.b,
        "sanity": {"baselines_identical_across_runs": baselines_identical, "validation_rows_identical": rows_identical},
        "per_fold": T.round(4).reset_index().to_dict("records"),
        "declared_rules": {"R1": f"mean {args.b} AUROC >= {R1_MIN_B} and mean drop <= {R1_MAX_DROP}",
                           "R2": f"mean drop >= {R2_MIN_DROP} and drop >= {R2_FOLD_MARGIN} in >= {R2_MIN_FOLDS} of 5 folds",
                           "else": "INCONCLUSIVE"},
        "observed": {f"mean_{args.b}_auroc": round(mean_b, 4), "mean_drop": round(mean_drop, 4), "folds_with_drop_ge_margin": folds_dropped,
                     "R1_holds": r1, "R2_holds": r2},
        "DECLARED_OUTCOME": outcome,
        "secondary": {
            "texture_strata_auroc_cnn": {args.a: {b["bin"]: b["auroc"]["cnn"] for b in A["lap_var_quintile_strata"]["bins"]},
                                         args.b: {b["bin"]: b["auroc"]["cnn"] for b in B["lap_var_quintile_strata"]["bins"]}},
            "pooled_within_fold_ranks_cnn": {args.a: A["pooled_auroc_within_fold_ranks"]["cnn"], args.b: B["pooled_auroc_within_fold_ranks"]["cnn"]},
            "mixed_patients": {args.a: A["mixed_patients_individual"], args.b: B["mixed_patients_individual"]},
            "threshold_0.5_cnn": {args.a: A["threshold_0.5_metrics_seed_ensemble"]["cnn_ens"], args.b: B["threshold_0.5_metrics_seed_ensemble"]["cnn_ens"]},
            "single_class_patients_cnn": {args.a: {k: v for k, v in A["single_class_patient_level_at_0.5"]["cnn"].items()},
                                          args.b: {k: v for k, v in B["single_class_patient_level_at_0.5"]["cnn"].items()}},
            f"{args.b}_E1_style_gate_vs_B2": B.get("secondary_E1_gate_rules_applied_vs_B2"),
            "epoch0_val_auroc_median": {args.a: float(pd.DataFrame(A["training_dynamics"]).epoch0_val_auroc.median()),
                                        args.b: float(pd.DataFrame(B["training_dynamics"]).epoch0_val_auroc.median())},
        },
    }
    write_json(PROJECT_ROOT / "reports" / f"{args.b}_vs_{args.a}.json", out)

    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "font.size": 10,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = T.index.to_numpy()
    for col, lab, c, mk, ls, lw, dx in ((f"{args.a}_cnn", "E1 residual CNN @ 0.26 um/px", "#2a78d6", "o", "-", 2.2, -0.08),
                                        (f"{args.b}_cnn", "E2 same CNN @ 1.04 um/px", "#1baf7a", "s", "--", 2.0, 0.08),
                                        ("B2", "B2 colour + texture baseline", "#4a4945", "^", ":", 1.5, 0.0)):
        ax.plot(x + dx, T[col], marker=mk, ls=ls, lw=lw, color=c, ms=7, mec=SURF, mew=1.2, label=lab)   # small x-offsets: E1 and E2 coincide
    ax.text(0.99, 0.5, "E1 and E2 overlap at ~1.000 in every fold\n(points offset sideways so both are visible)", transform=ax.transAxes,
            ha="right", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([f"fold {f}" for f in x])
    ax.set_ylim(min(0.85, float(T[[f"{args.a}_cnn", f"{args.b}_cnn", "B2"]].min().min()) - 0.03), 1.01)   # never clip a low E2 value
    ax.set_ylabel("validation AUROC within the fold\n(epoch 29, mean of 3 seeds)")
    ax.set_title(f"Pixel size 0.26 vs 1.04 um/px, same 66.6 um tiles. Declared outcome: {outcome}")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left"); fig.tight_layout()
    fig.savefig(PROJECT_ROOT / "reports" / "figures" / f"{args.b}_vs_{args.a}_per_fold.png", dpi=140); plt.close(fig)

    print(T.round(4).to_string()); print(json.dumps(out["observed"]), "| sanity:", out["sanity"]); print("DECLARED OUTCOME:", outcome)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
