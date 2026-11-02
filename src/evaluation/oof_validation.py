"""Pooled out-of-fold VALIDATION evaluation of a finished experiment, against the shortcut baselines.

Scope guarantee (precise): the image store holds only images that are validation images in SOME fold (18 patients,
665 images); the 6 patients that are never in any validation set (1, 14, 15, 17, 19, 23) are never decoded. Note that
an image can be validation in one fold and test in another, so the store alone does not exclude every test image.
What does: each fold's model is scored ONLY on that fold's own validation rows, and the loop below asserts it.

Caveat that goes with every number: each fold's `best.pt` was chosen by validation AUROC (early stopping), so CNN
validation scores are optimistically biased, while the baselines (fit on training data only, never tuned) are not.

  python -I src/evaluation/oof_validation.py --run e1_residual --seeds 0 1 2
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
import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from src.common import limit_cpu_threads, PROJECT_ROOT, load_config, resolve, write_json  # noqa: E402
from src.data.dataset import AugConfig, GPUPreprocess, ImageStore  # noqa: E402
from src.evaluation.metrics import binary_metrics  # noqa: E402
from src.inference.predict import predict_store  # noqa: E402
from src.models import colour_baseline  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.training.trainer import fold_rows  # noqa: E402
from src.training.utils import select_amp_dtype, select_device  # noqa: E402

MIXED = [12, 13, 21]          # mixed patients that occur in some validation set (patient 1 is never validation)
BASELINES = list(colour_baseline.BASELINES)


def auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def load_run(ck_path: Path, device: torch.device):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = build_model(cfg["model"])
    model.load_state_dict(ck["model"])
    model = model.to(device).to(memory_format=torch.channels_last).eval()
    d = cfg["data"]
    pre = GPUPreprocess(ck["channel_mean"], ck["channel_std"], d["crop"], d["scale"], AugConfig(**cfg["aug"])).to(device)
    return model, pre, cfg, ck.get("epoch", ck.get("state", {}).get("epoch"))


def cluster_boot(df: pd.DataFrame, stat, n_boot: int = 2000, seed: int = 0) -> dict:
    """Resample PATIENTS with replacement; `stat(frame) -> dict[str, float]`. Returns percentile CIs per key."""
    uniq = df.patient_id.unique()
    members = {p: np.where(df.patient_id.values == p)[0] for p in uniq}
    rng = np.random.default_rng(seed)
    out: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = np.concatenate([members[p] for p in rng.choice(uniq, len(uniq))])
        for k, v in stat(df.iloc[idx]).items():
            if np.isfinite(v):
                out.setdefault(k, []).append(v)
    return {k: {"lo": float(np.quantile(v, .025)), "hi": float(np.quantile(v, .975)), "n_valid": len(v)} for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e1_residual")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tag", default="e1")
    ap.add_argument("--checkpoint", choices=["last", "best"], default="last", help="last = epoch-29 (PRIMARY, no selection on validation); best = diagnostic")
    args = ap.parse_args()
    limit_cpu_threads()
    dcfg = load_config("data.yaml")
    man = pd.read_csv(resolve(dcfg["paths"]["metadata"]) / "manifest.csv", keep_default_na=False, na_values=[])
    folds = pd.read_csv(PROJECT_ROOT / "data/splits/folds_v1.csv")
    device, K = select_device("cuda"), 5
    amp = select_amp_dtype("bf16", device)

    val_ids = sorted(set().union(*[set(folds.loc[folds[f"role_fold{f}"] == "val", "image_id"]) for f in range(K)]))
    man_val = man[man.image_id.isin(val_ids)].reset_index(drop=True)       # <- the ONLY images this script can load
    store = ImageStore.from_manifest(man_val, resolve(dcfg["paths"]["raw_images"]))
    role_of = folds.set_index("image_id")
    never_val = sorted(set(man.patient_id) - set(man_val.patient_id))
    man_i = man.set_index("image_id")

    rows = []
    for f in range(K):
        va = fold_rows(store, folds, f, "val")
        ids = [store.image_ids[r] for r in va]
        assert (role_of.loc[ids, f"role_fold{f}"] == "val").all(), "scoring rows must all be this fold's validation rows"
        assert not (role_of.loc[ids, "test_fold"] == f).any(), "a scored row is a test row of this fold"
        base = {"fold": f, "image_id": ids, "patient_id": store.patient_ids[va], "label": store.labels[va],
                "lap_var": man_i.loc[ids, "lap_var"].to_numpy()}
        tr = man_i.loc[folds.loc[folds[f"role_fold{f}"] == "train", "image_id"]]
        vdf = man_i.loc[ids]
        for b, feats in colour_baseline.BASELINES.items():
            base[b] = colour_baseline.fit_predict(tr, vdf, features=feats)
        for s in args.seeds:
            model, pre, cfg, ep = load_run(PROJECT_ROOT / "checkpoints" / f"{args.run}_s{s}" / f"fold{f}" / f"{args.checkpoint}.pt", device)
            assert cfg["fold"] == f and cfg["seed"] == s
            if args.checkpoint == "last":
                assert ep == cfg["optim"]["epochs"] - 1, f"seed {s} fold {f}: last.pt is epoch {ep}, run did not finish"
            _, p = predict_store(model, store, va, pre, cfg["data"]["eval_tile"], cfg["data"]["eval_batch_images"], device, amp)
            base[f"cnn_s{s}"] = p
        rows.append(pd.DataFrame(base))
    T = pd.concat(rows, ignore_index=True)
    seed_cols = [f"cnn_s{s}" for s in args.seeds]
    T["cnn_ens"] = T[seed_cols].mean(axis=1)

    y = T.label.to_numpy()
    model_cols = seed_cols + ["cnn_ens"] + BASELINES
    for c in model_cols:                       # within-fold percentile ranks: removes cross-fold calibration differences
        T[f"r_{c}"] = T.groupby("fold")[c].rank(pct=True)
    rcols = [f"r_{c}" for c in seed_cols]

    def cnn_auc(frame: pd.DataFrame, ranked: bool = True) -> float:      # seed-averaged AUROC
        return float(np.mean([auc(frame.label.to_numpy(), frame[("r_" if ranked else "") + c].to_numpy()) for c in seed_cols]))

    def base_auc(frame: pd.DataFrame, b: str, ranked: bool = True) -> float:
        return auc(frame.label.to_numpy(), frame[("r_" if ranked else "") + b].to_numpy())

    R: dict = {"scope": "VALIDATION partitions only: each fold's model scored only on that fold's own validation rows (asserted); "
                        f"{len(never_val)} patients never in a validation set were not decoded",
               "patients_never_in_a_validation_set": never_val, "checkpoint": args.checkpoint,
               "caveat": ("checkpoint=last: epoch-29 weights, no selection on validation. checkpoint=best: chosen by validation AUROC, optimistically biased"
                          if args.checkpoint == "last" else "best.pt was chosen by validation AUROC: CNN numbers are optimistically biased (diagnostic only)"),
               "rows": int(len(T)), "distinct_patients": int(T.patient_id.nunique()), "distinct_images": int(T.image_id.nunique()),
               "patients_in_two_folds": sorted(int(p) for p, n in T.groupby("patient_id").fold.nunique().items() if n > 1)}

    # ---- PRIMARY: per-fold AUROC (within-fold ranking, so no calibration artefact) and paired differences ----
    pf = []
    for f, g in T.groupby("fold"):
        yy = g.label.to_numpy()
        pf.append({"fold": int(f), "val_patients": int(g.patient_id.nunique()), "n": int(len(g)), "n_ptc": int(yy.sum()),
                   "cnn": cnn_auc(g, False), **{bl: base_auc(g, bl, False) for bl in BASELINES},
                   "cnn_seed_min": float(min(auc(yy, g[c].to_numpy()) for c in seed_cols)), "cnn_seed_max": float(max(auc(yy, g[c].to_numpy()) for c in seed_cols))})
    PF = pd.DataFrame(pf)
    PF["cnn_minus_B2"], PF["cnn_minus_B1"] = PF.cnn - PF.B2_colour_texture, PF.cnn - PF.B1_texture
    R["per_fold"] = PF.round(4).to_dict("records")
    R["paired_per_fold_summary"] = {"cnn_minus_B2": {"mean": round(float(PF.cnn_minus_B2.mean()), 4), "folds_cnn_wins": int((PF.cnn_minus_B2 > 0).sum()), "of": K},
                                    "cnn_minus_B1": {"mean": round(float(PF.cnn_minus_B1.mean()), 4), "folds_cnn_wins": int((PF.cnn_minus_B1 > 0).sum()), "of": K},
                                    "cnn_mean_auroc": round(float(PF.cnn.mean()), 4), **{f"{bl}_mean_auroc": round(float(PF[bl].mean()), 4) for bl in BASELINES}}

    # ---- stratified by label-free texture quintile (edges from ALL 1,006 images, no labels involved); within-fold ranks ----
    edges = np.quantile(man.lap_var, [.2, .4, .6, .8])
    T["lap_bin"] = np.digitize(T.lap_var, edges)
    strata = []
    for bq, g in T.groupby("lap_bin"):
        yy = g.label.to_numpy()
        rec = {"bin": int(bq) + 1, "lap_var_range": [round(float(np.min(g.lap_var)), 1), round(float(np.max(g.lap_var)), 1)],
               "n": int(len(g)), "n_ptc": int(yy.sum()), "n_non": int((1 - yy).sum()), "n_patients": int(g.patient_id.nunique())}
        ok = min(rec["n_ptc"], rec["n_non"]) >= 10
        rec["auroc"] = {"cnn": round(cnn_auc(g), 4) if ok else None, **{bl: round(base_auc(g, bl), 4) if ok else None for bl in BASELINES}}
        strata.append(rec)
    R["lap_var_quintile_strata"] = {"edges_from_all_images": [round(float(e), 1) for e in edges], "bins": strata,
                                    "note": "within-fold rank scores; AUROC only where both classes have >= 10 images; bins 1 and 5 are almost single-class"}

    # ---- pooled AUROC on within-fold ranks (secondary) + raw-probability pooled (calibration-sensitive, secondary) ----
    def pooled(frame: pd.DataFrame) -> dict[str, float]:
        d = {"cnn": cnn_auc(frame), **{bl: base_auc(frame, bl) for bl in BASELINES}}
        d["cnn_minus_B1"], d["cnn_minus_B2"] = d["cnn"] - d["B1_texture"], d["cnn"] - d["B2_colour_texture"]
        return d
    point, ci = pooled(T), cluster_boot(T, pooled)
    R["pooled_auroc_within_fold_ranks"] = {k: {"point": round(point[k], 4), **{a: round(v, 4) for a, v in ci[k].items() if a != "n_valid"}} for k in point}
    R["pooled_auroc_raw_probabilities"] = {"cnn": round(cnn_auc(T, False), 4), **{bl: round(base_auc(T, bl, False), 4) for bl in BASELINES},
                                           "note": "mixes fold models' calibration; shown only to quantify that artefact"}
    R["threshold_0.5_metrics_seed_ensemble"] = {c: {k: round(v, 4) for k, v in binary_metrics(y, T[c].to_numpy()).items() if k in ("sensitivity", "specificity", "balanced_accuracy", "accuracy", "ece")}
                                                 for c in ["cnn_ens"] + BASELINES}

    # ---- mixed patients, individually, NO bootstrap (2-3 patients make a patient bootstrap degenerate) ----
    mixed = {}
    for pid in MIXED:
        g = T[T.patient_id == pid]
        yy = g.label.to_numpy()
        mixed[str(pid)] = {"fold": int(g.fold.iloc[0]), "n": int(len(g)), "n_ptc": int(yy.sum()), "n_non": int((1 - yy).sum()),
                           "auroc": {"cnn": round(cnn_auc(g, False), 4), **{bl: round(base_auc(g, bl, False), 4) for bl in BASELINES}} if min(yy.sum(), (1 - yy).sum()) >= 10 else "not evaluable (<10 of one class)"}
    R["mixed_patients_individual"] = mixed

    # ---- single-class patients: mean predicted probability vs the patient's class ----
    pat = T.groupby(["patient_id", "fold"]).agg(label_mean=("label", "mean"), n=("label", "size"), cnn=("cnn_ens", "mean"),
                                                 **{bl: (bl, "mean") for bl in BASELINES}).reset_index()
    pat = pat[(pat.label_mean == 0) | (pat.label_mean == 1)]
    R["single_class_patient_level_at_0.5"] = {m: {"correct": int(((pat[m] >= .5) == (pat.label_mean == 1)).sum()), "of": int(len(pat))} for m in ["cnn"] + BASELINES}
    R["single_class_patient_level_at_0.5"]["table"] = pat.round(3).to_dict("records")

    # ---- THE PRE-DECLARED GATE (Report 3 section 14; written before any E1 result existed) ----
    MARGIN = 0.02
    mid = [b_ for b_ in strata if b_["bin"] in (2, 3, 4)]
    bin_diff = {b_["bin"]: (None if b_["auroc"]["cnn"] is None else round(b_["auroc"]["cnn"] - b_["auroc"]["B2_colour_texture"], 4)) for b_ in mid}
    gate = {"a1_mean_per_fold_cnn_minus_B2_ge_0.02": {"value": R["paired_per_fold_summary"]["cnn_minus_B2"]["mean"], "pass": bool(R["paired_per_fold_summary"]["cnn_minus_B2"]["mean"] >= MARGIN)},
            "a2_cnn_beats_B2_in_at_least_4_of_5_folds": {"value": R["paired_per_fold_summary"]["cnn_minus_B2"]["folds_cnn_wins"], "pass": bool(R["paired_per_fold_summary"]["cnn_minus_B2"]["folds_cnn_wins"] >= 4)},
            "b_cnn_minus_B2_ge_0_in_each_texture_bin_Q2_Q3_Q4": {"value": bin_diff, "pass": bool(all(v is not None and v >= 0 for v in bin_diff.values()))}}
    gate["OVERALL"] = "PASS" if all(v["pass"] for v in gate.values()) else "FAIL"
    gate["informational_same_rules_vs_B1"] = {"mean_diff": R["paired_per_fold_summary"]["cnn_minus_B1"]["mean"], "folds_cnn_wins": R["paired_per_fold_summary"]["cnn_minus_B1"]["folds_cnn_wins"]}
    # E1's declared gate. For any other experiment it is NOT the declared rule, so it is stored under a secondary label.
    gate_key = "gate" if args.tag == "e1" else "secondary_E1_gate_rules_applied_vs_B2"
    R[gate_key] = gate

    # ---- training dynamics ----
    dyn = []
    for s in args.seeds:
        for f in range(K):
            d = PROJECT_ROOT / "checkpoints" / f"{args.run}_s{s}" / f"fold{f}"
            m = pd.read_json(d / "metrics.jsonl", lines=True)
            dyn.append({"seed": s, "fold": f, "epochs_run": int(len(m)), "epoch0_val_auroc": round(float(m.val_auroc.iloc[0]), 4), "best_epoch": int(m.val_auroc.idxmax()),
                        "best_val_auroc": round(float(m.val_auroc.max()), 4), "last_epoch_val_auroc": round(float(m.val_auroc.iloc[-1]), 4),
                        "mean_last5_val_auroc": round(float(m.val_auroc.tail(5).mean()), 4), "first_train_loss": round(float(m.train_loss.iloc[0]), 4),
                        "final_train_loss": round(float(m.train_loss.iloc[-1]), 4), "final_val_loss": round(float(m.val_loss.iloc[-1]), 4),
                        "peak_vram_gib": float(m.peak_vram_gib.max()), "minutes": round(float(m.seconds.sum()) / 60, 2)})
    R["training_dynamics"] = dyn
    write_json(PROJECT_ROOT / "reports" / f"{args.tag}_validation_results.json", R)
    T.drop(columns=[c for c in T.columns if c.startswith("r_")]).to_csv(PROJECT_ROOT / "reports" / f"{args.tag}_oof_validation_predictions.csv", index=False)

    # ---- figure: per-fold paired AUROC | texture-matched bins | pooled on within-fold ranks ----
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "font.size": 10,
                         "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    um_px = 0.26 / cfg["data"]["scale"]
    style = {"cnn": (f"Residual CNN @ {um_px:.2f} um/px, epoch 29 (3 seeds, mean)", "#2a78d6", "o", "-", 2.2), "B1_texture": ("B1 texture only", "#eb6834", "D", "-", 1.6),
             "B0_colour": ("B0 colour", "#7a7975", "s", "--", 1.2), "B2_colour_texture": ("B2 colour + texture", "#4a4945", "^", ":", 1.5)}
    fig, (a0, a2, a1) = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1.15, 1.25, 1]})
    for k, (lab, col, mk, ls, lw) in style.items():
        a0.plot(PF.fold, PF[k], marker=mk, ls=ls, lw=lw, color=col, ms=6.5, mec=SURF, mew=1.2, label=lab)
    a0.set_xticks(range(K)); a0.set_xticklabels([f"fold {f}\n{int(r.val_patients)} pts" for f, r in zip(PF.fold, PF.itertuples())], fontsize=8.5)
    a0.set_ylim(0.4, 1.02); a0.set_ylabel("validation AUROC (within the fold)"); a0.set_title("PRIMARY: per-fold AUROC"); a0.legend(frameon=False, fontsize=8, loc="lower left")
    xs = np.arange(1, 6)
    for k, (lab, col, mk, ls, lw) in style.items():
        ys = [np.nan if b_["auroc"]["cnn" if k == "cnn" else k] is None else b_["auroc"]["cnn" if k == "cnn" else k] for b_ in strata]
        a2.plot(xs, ys, marker=mk, ls=ls, lw=lw, color=col, ms=6.5, mec=SURF, mew=1.2)
    a2.set_xticks(xs); a2.set_xticklabels([f"Q{b_['bin']}\n{b_['n_ptc']} PTC | {b_['n_non']} non" for b_ in strata], fontsize=8.2)
    a2.set_ylim(0.3, 1.02); a2.axhline(0.5, color=INK2, lw=.8, ls=":"); a2.text(4.95, .515, "chance", fontsize=8, color=INK2, ha="right")
    a2.axvspan(1.5, 4.5, color="#f0efec", zorder=0); a2.text(3, 0.325, "gate bins Q2-Q4", ha="center", fontsize=8, color=INK2)
    a2.set_xlabel("texture quintile (label-free; Q1 smoothest)", fontsize=9); a2.set_title("Within texture-matched groups")
    for i, (k, (lab, col, mk, ls, lw)) in enumerate(style.items()):
        kk = "cnn" if k == "cnn" else k
        pt, lo, hi = (R["pooled_auroc_within_fold_ranks"][kk][x] for x in ("point", "lo", "hi"))
        a1.plot([lo, hi], [i, i], color=col, lw=2); a1.plot(pt, i, mk, color=col, ms=8, mec=SURF, mew=1.5)
        a1.text(max(hi, pt) + .006, i, f"{pt:.3f}", va="center", fontsize=8.5, color=INK2)
    a1.set_yticks(range(4)); a1.set_yticklabels(["CNN", "B1 texture", "B0 colour", "B2 colour+texture"], fontsize=9); a1.invert_yaxis()
    a1.set_xlim(0.55, 1.06); a1.grid(axis="y", visible=False); a1.set_xlabel("AUROC, 95% CI resampling patients", fontsize=9)
    a1.set_title("Pooled over folds (within-fold ranks)")
    fig.tight_layout(); out = PROJECT_ROOT / "reports" / "figures" / f"{args.tag}_validation_auroc.png"; fig.savefig(out, dpi=140); plt.close(fig)

    print(("GATE:" if gate_key == "gate" else "SECONDARY (E1 gate rules applied to this run; NOT its declared rule):"), json.dumps(R[gate_key], indent=1))
    print("\nper-fold AUROC:"); print(PF.round(4).to_string(index=False))
    print("\npaired summary:", json.dumps(R["paired_per_fold_summary"]))
    print("\nstrata:"); [print(" ", b_) for b_ in strata]
    print("\npooled (within-fold ranks):", json.dumps(R["pooled_auroc_within_fold_ranks"])); print("pooled raw prob:", json.dumps(R["pooled_auroc_raw_probabilities"]))
    print("\nmixed patients individually:", json.dumps(mixed))
    print("\nsingle-class patient-level @0.5:", {k: v for k, v in R["single_class_patient_level_at_0.5"].items() if k != "table"})
    print("\nthreshold 0.5 (seed ensemble):", json.dumps(R["threshold_0.5_metrics_seed_ensemble"]))
    print("\ndynamics:"); print(pd.DataFrame(dyn).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
