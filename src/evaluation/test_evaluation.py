"""ONE-TIME final evaluation on the outer TEST folds (protocol: reports/06_final_test_evaluation.md, declared before running).

Scores the frozen E1 and E2 epoch-29 checkpoints (3 seeds each) and the B0/B1/B2 baselines (fit on each fold's training partition)
on each fold's test patients. Every image is a test image in exactly one fold, so the pooled result covers all 1,006 images and all
24 patients exactly once.

Guards: refuses to run if any output already exists (the test folds are opened once) or if any frozen file (30 checkpoints, fold
table, manifest) differs from reports/provenance/test/FROZEN_SHA256SUMS. Nothing computed here is used to choose a model or threshold.

  python -I src/evaluation/test_evaluation.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import beta  # noqa: E402

from src.common import PROJECT_ROOT, load_config, resolve, write_json  # noqa: E402
from src.data.dataset import ImageStore  # noqa: E402
from src.evaluation.metrics import binary_metrics  # noqa: E402
from src.evaluation.oof_validation import auc, cluster_boot, load_run  # noqa: E402
from src.inference.predict import predict_store  # noqa: E402
from src.models import colour_baseline  # noqa: E402
from src.training.trainer import fold_rows  # noqa: E402
from src.training.utils import select_amp_dtype, select_device  # noqa: E402

RUNS = {"E1": "e1_residual", "E2": "e2_residual_lowres"}
SEEDS, K, THRESHOLD = [0, 1, 2], 5, 0.5
BASELINES = list(colour_baseline.BASELINES)
OUT_JSON = PROJECT_ROOT / "reports" / "06_test_results.json"
OUT_CSV = PROJECT_ROOT / "reports" / "06_test_predictions.csv"
OUT_FIG = PROJECT_ROOT / "reports" / "figures" / "06_test_per_fold_auroc.png"
FROZEN = PROJECT_ROOT / "reports" / "provenance" / "test" / "FROZEN_SHA256SUMS"
NEVER_VALIDATION = [1, 14, 15, 17, 19, 23]
MIXED = [1, 12, 13, 21]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        while b := fh.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def cp_interval(k: int, n: int, a: float = 0.05) -> list[float]:
    """Exact (Clopper-Pearson) two-sided 95% interval for a proportion k/n."""
    lo = 0.0 if k == 0 else float(beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a / 2, k + 1, n - k))
    return [round(lo, 4), round(hi, 4)]


def main() -> int:
    for p in (OUT_JSON, OUT_CSV):
        if p.exists():
            print(f"REFUSING: {p.name} already exists. The test folds are evaluated once; a repeat must be a deliberate, "
                  "documented second look (move the old outputs aside and record why).")
            return 2
    # ---- frozen-state check -------------------------------------------------------------------------------------------
    frozen = [l.split(None, 1) for l in FROZEN.read_text().splitlines() if l.strip()]
    mismatch = [rel.strip() for h, rel in frozen if sha256(PROJECT_ROOT / rel.strip()) != h]
    if mismatch:
        print("REFUSING: frozen files changed since the protocol was declared:", mismatch)
        return 3
    print(f"frozen state verified: {len(frozen)} files match FROZEN_SHA256SUMS")

    dcfg = load_config("data.yaml")
    man = pd.read_csv(resolve(dcfg["paths"]["metadata"]) / "manifest.csv", keep_default_na=False, na_values=[])
    folds = pd.read_csv(PROJECT_ROOT / "data/splits/folds_v1.csv")
    man = man[man.image_id.isin(folds.image_id)].reset_index(drop=True)
    role_of, man_i = folds.set_index("image_id"), man.set_index("image_id")
    device = select_device("cuda")
    amp = select_amp_dtype("bf16", device)
    store = ImageStore.from_manifest(man, resolve(dcfg["paths"]["raw_images"]))
    t0 = time.time()

    parts = []
    for f in range(K):
        te = fold_rows(store, folds, f, "test")
        ids = [store.image_ids[r] for r in te]
        assert (role_of.loc[ids, f"role_fold{f}"] == "test").all() and (role_of.loc[ids, "test_fold"] == f).all()
        D = pd.DataFrame({"fold": f, "image_id": ids, "patient_id": store.patient_ids[te], "label": store.labels[te],
                          "wsi_key": man_i.loc[ids, "wsi_key"].to_numpy(), "lap_var": man_i.loc[ids, "lap_var"].to_numpy()})
        tr = man_i.loc[folds.loc[folds[f"role_fold{f}"] == "train", "image_id"]]          # baselines: same data the CNNs trained on
        for b, feats in colour_baseline.BASELINES.items():
            D[b] = colour_baseline.fit_predict(tr, man_i.loc[ids], features=feats)
        for name, run in RUNS.items():
            for s in SEEDS:
                model, pre, cfg, ep = load_run(PROJECT_ROOT / "checkpoints" / f"{run}_s{s}" / f"fold{f}" / "last.pt", device)
                assert cfg["fold"] == f and cfg["seed"] == s and ep == cfg["optim"]["epochs"] - 1
                _, p = predict_store(model, store, te, pre, cfg["data"]["eval_tile"], cfg["data"]["eval_batch_images"], device, amp)
                D[f"{name}_s{s}"] = p
                del model
            D[f"{name}_ens"] = D[[f"{name}_s{s}" for s in SEEDS]].mean(axis=1)
        parts.append(D)
    T = pd.concat(parts, ignore_index=True)
    assert len(T) == len(man) == T.image_id.nunique() == 1006, "every image must be scored exactly once"
    assert T.patient_id.nunique() == 24 and set(T.image_id) == set(man.image_id)

    models = {"E1": "E1_ens", "E2": "E2_ens", **{b: b for b in BASELINES}}
    y = T.label.to_numpy()
    R: dict = {"scope": "ONE-TIME outer-test evaluation; all 1,006 images / 24 patients, each scored once by the models of the fold in which it is a test image",
               "threshold": THRESHOLD, "seconds": round(time.time() - t0, 1),
               "frozen_files_verified": len(frozen)}

    # ---- PRIMARY: per-fold AUROC (point estimates) ---------------------------------------------------------------------
    pf = []
    for f, g in T.groupby("fold"):
        yy = g.label.to_numpy()
        rec = {"fold": int(f), "test_patients": sorted(int(p) for p in g.patient_id.unique()), "n": int(len(g)), "n_ptc": int(yy.sum())}
        for name in RUNS:
            per_seed = [auc(yy, g[f"{name}_s{s}"].to_numpy()) for s in SEEDS]
            rec[name], rec[f"{name}_seed_min"], rec[f"{name}_seed_max"] = float(np.mean(per_seed)), float(min(per_seed)), float(max(per_seed))
        for b in BASELINES:
            rec[b] = auc(yy, g[b].to_numpy())
        pf.append(rec)
    PF = pd.DataFrame(pf)
    R["per_fold_auroc"] = PF.round(4).to_dict("records")
    R["per_fold_auroc_mean"] = {m: round(float(PF[m].mean()), 4) for m in list(RUNS) + BASELINES}

    # ---- pooled threshold metrics at 0.5 (seed ensembles for the CNNs), confusion matrices, per-seed spread -------------
    thr = {}
    for m, col in models.items():
        bm = binary_metrics(y, T[col].to_numpy(), THRESHOLD)
        thr[m] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in bm.items()}
    for name in RUNS:
        seeds = [binary_metrics(y, T[f"{name}_s{s}"].to_numpy(), THRESHOLD) for s in SEEDS]
        thr[name]["per_seed_errors"] = [int(d["fp"] + d["fn"]) for d in seeds]
        thr[name]["per_seed_sensitivity"] = [round(d["sensitivity"], 4) for d in seeds]
        thr[name]["per_seed_specificity"] = [round(d["specificity"], 4) for d in seeds]
    R["pooled_threshold_0.5"] = thr
    R["pooled_auroc_raw_probabilities"] = {m: round(auc(y, T[c].to_numpy()), 4) for m, c in models.items()}

    # ---- uncertainty: patient bootstrap only where there are errors; exact patient-level bounds otherwise --------------
    def boot_stats(frame: pd.DataFrame) -> dict[str, float]:
        out = {}
        for m, c in models.items():
            d = binary_metrics(frame.label.to_numpy(), frame[c].to_numpy(), THRESHOLD)
            out[f"{m}_sensitivity"], out[f"{m}_specificity"], out[f"{m}_balanced_accuracy"] = d["sensitivity"], d["specificity"], d["balanced_accuracy"]
        return out
    ci = cluster_boot(T, boot_stats, n_boot=2000, seed=0)
    R["patient_bootstrap_ci95"] = {}
    for m in models:
        has_fn, has_fp = thr[m]["fn"] > 0, thr[m]["fp"] > 0
        R["patient_bootstrap_ci95"][m] = {
            "sensitivity": [round(ci[f"{m}_sensitivity"]["lo"], 4), round(ci[f"{m}_sensitivity"]["hi"], 4)] if has_fn else "no false negatives: see patient-level bound",
            "specificity": [round(ci[f"{m}_specificity"]["lo"], 4), round(ci[f"{m}_specificity"]["hi"], 4)] if has_fp else "no false positives: see patient-level bound",
            "balanced_accuracy": [round(ci[f"{m}_balanced_accuracy"]["lo"], 4), round(ci[f"{m}_balanced_accuracy"]["hi"], 4)] if (has_fn or has_fp) else "no errors: see patient-level bound"}

    # ---- patient level ------------------------------------------------------------------------------------------------
    pl = {}
    for m, c in models.items():
        T["_wrong"] = ((T[c] >= THRESHOLD).astype(int) != T.label).astype(int)
        per_pat = T.groupby("patient_id").agg(n=("label", "size"), errors=("_wrong", "sum"), label_mean=("label", "mean"), mean_p=(c, "mean"))
        n_err_pat = int((per_pat.errors > 0).sum())
        single = per_pat[(per_pat.label_mean == 0) | (per_pat.label_mean == 1)]
        wrong_single = int(((single.mean_p >= THRESHOLD).astype(int) != single.label_mean).sum())
        pl[m] = {"patients_with_any_image_error": n_err_pat, "of_patients": 24,
                 "ci95_proportion_patients_with_any_error": cp_interval(n_err_pat, 24),
                 "single_class_patients_misclassified_by_mean_score": wrong_single, "of_single_class_patients": int(len(single)),
                 "ci95_proportion_single_class_patients_misclassified": cp_interval(wrong_single, int(len(single)))}
    T.drop(columns="_wrong", inplace=True)
    R["patient_level"] = pl

    # ---- calibration (ECE, Brier) are inside pooled_threshold_0.5; errors listed ---------------------------------------
    errs = {}
    for m in ("E1", "E2"):
        c = models[m]
        e = T[(T[c] >= THRESHOLD).astype(int) != T.label]
        errs[m] = e[["fold", "image_id", "patient_id", "label", c] + [f"{m}_s{s}" for s in SEEDS] + ["B2_colour_texture"]].round(4).to_dict("records")
    R["cnn_errors_at_0.5"] = errs

    # ---- mixed patients (within-patient AUROC; slide caveat) and never-validation patients -------------------------
    mixed = {}
    for pid in MIXED:
        g = T[T.patient_id == pid]
        yy = g.label.to_numpy()
        ok = min(yy.sum(), (1 - yy).sum()) >= 5
        mixed[str(pid)] = {"fold": int(g.fold.iloc[0]), "n_ptc": int(yy.sum()), "n_non": int((1 - yy).sum()),
                           "slides": sorted(g.wsi_key.unique().tolist()),
                           "auroc": {m: round(auc(yy, g[c].to_numpy()), 4) for m, c in models.items()} if ok else "not evaluable (<5 images of one class)"}
    R["mixed_patients_individual"] = mixed
    nv = []
    for pid in NEVER_VALIDATION:
        g = T[T.patient_id == pid]
        nv.append({"patient_id": pid, "fold": int(g.fold.iloc[0]), "n": int(len(g)), "n_ptc": int(g.label.sum()),
                   **{f"{m}_mean_p_ptc": round(float(g[c].mean()), 4) for m, c in (("E1", "E1_ens"), ("E2", "E2_ens"))},
                   **{f"{m}_errors_at_0.5": int(((g[c] >= THRESHOLD).astype(int) != g.label).sum()) for m, c in (("E1", "E1_ens"), ("E2", "E2_ens"))}})
    R["never_in_a_validation_set"] = nv

    # ---- texture strata, corrected version: inside each (fold, bin) cell with >= 5 images per class --------------------
    edges = np.quantile(man.lap_var, [.2, .4, .6, .8])
    T["lap_bin"] = np.digitize(T.lap_var, edges) + 1
    cells = []
    for (f, b), g in T.groupby(["fold", "lap_bin"]):
        yy = g.label.to_numpy()
        if min(yy.sum(), (1 - yy).sum()) < 5:
            continue
        cells.append({"fold": int(f), "bin": int(b), "n_ptc": int(yy.sum()), "n_non": int((1 - yy).sum()),
                      **{m: round(float(np.mean([auc(yy, g[f"{m}_s{s}"].to_numpy()) for s in SEEDS])), 4) for m in RUNS},
                      **{bl: round(auc(yy, g[bl].to_numpy()), 4) for bl in BASELINES}})
    R["texture_strata_within_fold_cells"] = cells

    # ---- figure ---------------------------------------------------------------------------------------------------------
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "font.size": 10,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = PF.fold.to_numpy()
    for col, lab, c, mk, ls, lw, dx in (("E1", "E1 CNN @ 0.26 um/px", "#2a78d6", "o", "-", 2.2, -0.08), ("E2", "E2 CNN @ 1.04 um/px", "#1baf7a", "s", "--", 2.0, 0.08),
                                        ("B1_texture", "B1 texture only", "#eb6834", "D", "-", 1.5, 0), ("B0_colour", "B0 colour", "#7a7975", "s", "--", 1.2, 0),
                                        ("B2_colour_texture", "B2 colour + texture", "#4a4945", "^", ":", 1.5, 0)):
        ax.plot(x + dx, PF[col], marker=mk, ls=ls, lw=lw, color=c, ms=6.5, mec=SURF, mew=1.2, label=lab)
    ax.set_xticks(x); ax.set_xticklabels([f"fold {int(f)}\n{len(p)} test pts" for f, p in zip(PF.fold, PF.test_patients)], fontsize=8.5)
    ax.set_ylim(min(0.5, float(PF[list(RUNS) + BASELINES].min().min()) - 0.03), 1.02)
    ax.set_ylabel("TEST AUROC within the fold"); ax.set_title("Final exam (outer test folds, scored once): per-fold AUROC")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left"); fig.tight_layout(); fig.savefig(OUT_FIG, dpi=140); plt.close(fig)

    T.to_csv(OUT_CSV, index=False)          # outputs are written last: the guard only blocks a run whose results were produced
    write_json(OUT_JSON, R)

    print(PF.round(4).to_string(index=False))
    print("\nper-fold mean:", R["per_fold_auroc_mean"])
    for m in models:
        t = thr[m]
        print(f"{m:18s} @0.5  TP {t['tp']:3d} FN {t['fn']:3d} FP {t['fp']:3d} TN {t['tn']:3d} | sens {t['sensitivity']:.4f} spec {t['specificity']:.4f} "
              f"prec {t['precision']:.4f} F1 {t['f1']:.4f} acc {t['accuracy']:.4f} bal {t['balanced_accuracy']:.4f} ECE {t['ece']:.4f} Brier {t['brier']:.4f}")
    print("\npatient level:", json.dumps(pl))
    print("\nCNN errors:", json.dumps(errs))
    print("\nmixed:", json.dumps(mixed)); print("\nnever-validation patients:", json.dumps(nv)); print("\ntexture cells:", json.dumps(cells))
    print("\nbootstrap CIs:", json.dumps(R["patient_bootstrap_ci95"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
