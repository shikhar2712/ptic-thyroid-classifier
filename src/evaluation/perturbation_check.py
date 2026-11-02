"""Inference-only diagnostic: what does a trained model rely on? Remove colour and/or fine detail from the VALIDATION images and
re-score. No training; outer-test folds are not touched (each fold's model is scored only on its own fold's validation rows).

Reading it: if AUROC stays ~1.0 after nuclear-scale detail is removed, the decision does not depend on nuclear morphology. If AUROC
collapses, that is ambiguous (loss of information OR distribution shift, since the models were trained on unperturbed images).

  python -I src/evaluation/perturbation_check.py --run e1_residual --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.common import PROJECT_ROOT, load_config, resolve, write_json  # noqa: E402
from src.data.dataset import ImageStore  # noqa: E402
from src.evaluation.oof_validation import auc, load_run  # noqa: E402
from src.inference.predict import predict_uint8  # noqa: E402
from src.training.trainer import fold_rows  # noqa: E402
from src.training.utils import select_amp_dtype, select_device  # noqa: E402

# name -> (grayscale?, downsample-then-upsample factor). Pixel size 0.26 um: factor 8 ~ 2.1 um/px, nuclei (6-10 um) ~ 3-5 px.
PERTURBATIONS = {"identity": (False, 1), "grayscale": (True, 1), "blur_x2": (False, 2), "blur_x4": (False, 4),
                 "blur_x8": (False, 8), "blur_x16": (False, 16), "gray_blur_x8": (True, 8)}


def perturb(x_u8: torch.Tensor, gray: bool, k: int) -> torch.Tensor:
    x = x_u8.float()
    if gray:
        x = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2]).unsqueeze(1).expand(-1, 3, -1, -1)
    if k > 1:
        h, w = x.shape[-2:]
        x = F.interpolate(F.interpolate(x, size=(h // k, w // k), mode="area"), size=(h, w), mode="bilinear", align_corners=False)
    return x.round().clamp(0, 255).to(torch.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e1_residual")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tag", default="e1")
    a = ap.parse_args()
    dcfg = load_config("data.yaml")
    man = pd.read_csv(resolve(dcfg["paths"]["metadata"]) / "manifest.csv", keep_default_na=False, na_values=[])
    folds = pd.read_csv(PROJECT_ROOT / "data/splits/folds_v1.csv")
    role_of = folds.set_index("image_id")
    K, device = 5, select_device("cuda")
    amp = select_amp_dtype("bf16", device)
    val_ids = sorted(set().union(*[set(folds.loc[folds[f"role_fold{f}"] == "val", "image_id"]) for f in range(K)]))
    store = ImageStore.from_manifest(man[man.image_id.isin(val_ids)].reset_index(drop=True), resolve(dcfg["paths"]["raw_images"]))

    rec = []   # one row per (perturbation, fold, seed)
    scores: dict = {}
    for f in range(K):
        va = fold_rows(store, folds, f, "val")
        ids = [store.image_ids[r] for r in va]
        assert (role_of.loc[ids, f"role_fold{f}"] == "val").all() and not (role_of.loc[ids, "test_fold"] == f).any()
        y = store.labels[va]
        for s in a.seeds:
            model, pre, cfg, ep = load_run(PROJECT_ROOT / "checkpoints" / f"{a.run}_s{s}" / f"fold{f}" / "last.pt", device)
            assert cfg["fold"] == f and cfg["seed"] == s and ep == cfg["optim"]["epochs"] - 1
            for name, (gray, k) in PERTURBATIONS.items():
                ps = []
                with torch.no_grad():
                    for i in range(0, len(va), 8):
                        x = store.images[torch.as_tensor(va[i:i + 8])].permute(0, 3, 1, 2).to(device)
                        ps.append(torch.softmax(predict_uint8(model.eval(), perturb(x, gray, k), pre, cfg["data"]["eval_tile"], amp), 1)[:, 1].cpu())
                p = torch.cat(ps).numpy()
                scores[(name, f, s)] = (y, p, store.patient_ids[va])
                rec.append({"perturbation": name, "fold": f, "seed": s, "auroc": auc(y, p), "mean_p_ptc_on_ptc": float(p[y == 1].mean()),
                            "mean_p_ptc_on_non": float(p[y == 0].mean()), "balanced_acc_at_0.5": float(0.5 * ((p[y == 1] >= .5).mean() + (p[y == 0] < .5).mean()))})
            del model
    R = pd.DataFrame(rec)
    per_fold = R.groupby(["perturbation", "fold"]).auroc.mean().unstack()                      # seed-averaged per-fold AUROC
    summ = R.groupby("perturbation").agg(mean_fold_auroc=("auroc", "mean"), min_fold_auroc=("auroc", "min"),
                                          bal_acc_at_0_5=("balanced_acc_at_0.5", "mean"), mean_p_on_ptc=("mean_p_ptc_on_ptc", "mean"),
                                          mean_p_on_non=("mean_p_ptc_on_non", "mean")).reindex(PERTURBATIONS)
    # mixed patients 12 and 13 individually (within-patient AUROC), seed-averaged
    mixed = {}
    for name in PERTURBATIONS:
        mixed[name] = {}
        for pid in (12, 13):
            vals = []
            for (nm, f, s), (y, p, pat) in scores.items():
                if nm == name and (pat == pid).any():
                    vals.append(auc(y[pat == pid], p[pat == pid]))
            mixed[name][str(pid)] = round(float(np.mean(vals)), 4)
    base = pd.read_json(PROJECT_ROOT / "reports" / f"{a.tag}_validation_results.json", typ="series")
    ref = {"B0_colour": base["paired_per_fold_summary"]["B0_colour_mean_auroc"], "B1_texture": base["paired_per_fold_summary"]["B1_texture_mean_auroc"],
           "B2_colour_texture": base["paired_per_fold_summary"]["B2_colour_texture_mean_auroc"]}
    write_json(PROJECT_ROOT / "reports" / f"{a.tag}_perturbation_check.json",
               {"scope": "inference-only, VALIDATION partitions, epoch-29 checkpoints, 3 seeds; no outer-test image scored",
                "pixel_size_um": 0.26, "perturbations": {k: {"grayscale": v[0], "blur_factor": v[1], "effective_um_per_px": round(0.26 * v[1], 2)} for k, v in PERTURBATIONS.items()},
                "per_fold_auroc_seed_mean": per_fold.round(4).to_dict("index"), "summary": summ.round(4).to_dict("index"),
                "mixed_patients_within_patient_auroc": mixed, "baseline_mean_fold_auroc_for_reference": ref,
                "caveat": "models were trained on unperturbed images: a drop can mean lost information or distribution shift"})

    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "font.size": 10, "axes.titleweight": "bold", "axes.titlelocation": "left"})
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    xs = np.arange(len(PERTURBATIONS))
    for f in range(K):
        ax.plot(xs, per_fold.loc[list(PERTURBATIONS), f], color="#9ec5f4", lw=1, marker="o", ms=3.5, zorder=2)
    ax.plot(xs, summ.mean_fold_auroc, color="#2a78d6", lw=2.4, marker="o", ms=7, mec=SURF, mew=1.4, label="Residual CNN, mean of 5 folds (thin: each fold)", zorder=3)
    for nm, col, ls in (("B1_texture", "#eb6834", "-"), ("B2_colour_texture", "#4a4945", ":")):
        ax.axhline(ref[nm], color=col, ls=ls, lw=1.4, label=f"{nm.split('_')[0]} baseline on clean images (mean of folds)")
    ax.axhline(0.5, color=INK2, lw=.8, ls=":"); ax.text(len(PERTURBATIONS) - 1.05, 0.515, "chance", fontsize=8, color=INK2, ha="right")
    ax.set_xticks(xs); ax.set_xticklabels([f"{k}\n{0.26 * v[1]:.2f} um/px" + ("\n+gray" if v[0] and k != "grayscale" else "") for k, v in PERTURBATIONS.items()], fontsize=8.2)
    ax.set_ylim(0.0, 1.02); ax.set_ylabel("validation AUROC within the fold"); ax.set_title("What the trained CNN needs: remove colour / remove fine detail at inference")
    ax.legend(frameon=False, fontsize=8.5, loc="lower left"); fig.tight_layout(); fig.savefig(PROJECT_ROOT / "reports" / "figures" / f"{a.tag}_perturbation_check.png", dpi=140); plt.close(fig)

    print(per_fold.reindex(list(PERTURBATIONS)).round(4).to_string()); print()
    print(summ.round(4).to_string()); print("\nmixed patients (within-patient AUROC):", mixed); print("\nbaseline reference (clean):", ref)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
