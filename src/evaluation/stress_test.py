"""Step 1: simulated outside-world stress test (inference only; VALIDATION partitions only; outer test folds untouched).

Each fold's frozen epoch-29 models (3 seeds) score that fold's own validation rows under image changes that differ between
scanners/labs: stain strength, colour/white balance, gamma, hue, focus blur, JPEG re-compression, sensor noise and magnification
(true resize: tissue appears larger/smaller). Every condition carries a STATUS fixed before any result existed
(reports/07_robustness.md section 1): "held_out" families are never used for training in step 2; "trained_beyond" / "trained_in"
are families step 2 may train on, evaluated beyond / inside the training range. Only held-out results measure robustness.

  python -I src/evaluation/stress_test.py --run e1_residual --tag e1
"""
from __future__ import annotations

import argparse
import io
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402

from src.common import limit_cpu_threads, PROJECT_ROOT, load_config, resolve, write_json  # noqa: E402
from src.data.dataset import _HED_ROWS, ImageStore  # noqa: E402
from src.evaluation.metrics import binary_metrics  # noqa: E402
from src.evaluation.oof_validation import auc, load_run  # noqa: E402
from src.inference.predict import predict_uint8  # noqa: E402
from src.training.trainer import fold_rows  # noqa: E402
from src.training.utils import select_amp_dtype, select_device  # noqa: E402

# (name, family, kind, parameter, status). Status is relative to the step-2 augmentation set declared in reports/07_robustness.md.
CONDITIONS = [
    ("clean", "clean", "none", None, "reference"),
    ("stain_H_x0.7", "stain", "stain", (0, 0.7), "trained_beyond"), ("stain_H_x1.3", "stain", "stain", (0, 1.3), "trained_beyond"),
    ("stain_E_x0.7", "stain", "stain", (1, 0.7), "trained_beyond"), ("stain_E_x1.3", "stain", "stain", (1, 1.3), "trained_beyond"),
    ("stain_pale_x0.7", "stain", "stain", (None, 0.7), "trained_beyond"), ("stain_dark_x1.4", "stain", "stain", (None, 1.4), "trained_beyond"),
    ("brightness_x0.8", "brightness/contrast", "brightness", 0.8, "trained_beyond"), ("brightness_x1.2", "brightness/contrast", "brightness", 1.2, "trained_beyond"),
    ("contrast_x0.75", "brightness/contrast", "contrast", 0.75, "trained_beyond"), ("contrast_x1.25", "brightness/contrast", "contrast", 1.25, "trained_beyond"),
    ("saturation_x0.6", "saturation", "saturation", 0.6, "trained_beyond"), ("saturation_x1.4", "saturation", "saturation", 1.4, "trained_beyond"),
    ("blur_sigma1", "focus blur", "blur", 1.0, "trained_in"), ("blur_sigma2", "focus blur", "blur", 2.0, "trained_beyond"),
    ("blur_sigma3", "focus blur", "blur", 3.0, "trained_beyond"),
    ("scale_x0.95", "magnification", "scale", 0.95, "trained_in"), ("scale_x1.05", "magnification", "scale", 1.05, "trained_in"),
    ("scale_x0.85", "magnification", "scale", 0.85, "trained_beyond"), ("scale_x1.2", "magnification", "scale", 1.2, "trained_beyond"),
    ("scale_x0.7", "magnification", "scale", 0.7, "trained_beyond"), ("scale_x1.5", "magnification", "scale", 1.5, "trained_beyond"),
    ("scale_x0.5_(20x_as_40x)", "magnification", "scale", 0.5, "trained_beyond"),
    ("jpeg_q90", "JPEG", "jpeg", 90, "held_out"), ("jpeg_q70", "JPEG", "jpeg", 70, "held_out"),
    ("jpeg_q50", "JPEG", "jpeg", 50, "held_out"), ("jpeg_q30", "JPEG", "jpeg", 30, "held_out"),
    ("cast_warm", "white balance", "cast", (1.06, 1.0, 0.94), "held_out"), ("cast_cool", "white balance", "cast", (0.94, 1.0, 1.06), "held_out"),
    ("gamma_0.8", "gamma", "gamma", 0.8, "held_out"), ("gamma_1.25", "gamma", "gamma", 1.25, "held_out"),
    ("hue_+10deg", "hue", "hue", 10.0, "held_out"), ("hue_-10deg", "hue", "hue", -10.0, "held_out"),
    ("grayscale", "grayscale", "gray", None, "held_out"),
    ("noise_sigma4", "sensor noise", "noise", 4.0, "held_out"), ("noise_sigma8", "sensor noise", "noise", 8.0, "held_out"),
]
_HED = torch.from_numpy(_HED_ROWS)
_HED_INV = torch.from_numpy(np.linalg.inv(_HED_ROWS).astype(np.float32))
_LUM = torch.tensor([0.299, 0.587, 0.114])
_YIQ = torch.tensor([[0.299, 0.587, 0.114], [0.596, -0.274, -0.322], [0.211, -0.523, 0.312]])


def _gauss_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = int(math.ceil(3 * sigma))
    t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    k = torch.exp(-t ** 2 / (2 * sigma ** 2)); k = k / k.sum()
    c = x.shape[1]
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def perturb(x_u8: torch.Tensor, kind: str, param, seed: int = 0) -> torch.Tensor:
    """uint8 [B,3,H,W] on device -> uint8 (8-bit quantised, as a real image file would be). Deterministic."""
    if kind == "none":
        return x_u8
    if kind == "jpeg":
        out = []
        for im in x_u8.permute(0, 2, 3, 1).cpu().numpy():
            buf = io.BytesIO(); Image.fromarray(im).save(buf, format="JPEG", quality=int(param)); buf.seek(0)
            out.append(torch.from_numpy(np.asarray(Image.open(buf).convert("RGB")).copy()))
        return torch.stack(out).permute(0, 3, 1, 2).to(x_u8.device)
    x = x_u8.float() / 255.0
    dev = x.device
    if kind == "stain":
        ch, f = param
        od = -torch.log(x.clamp(1 / 255, 1.0))
        conc = torch.einsum("bchw,cd->bdhw", od, _HED_INV.to(dev))
        if ch is None:
            conc = conc * f
        else:
            conc[:, ch] = conc[:, ch] * f
        x = torch.exp(-torch.einsum("bdhw,dc->bchw", conc, _HED.to(dev)).clamp_min(0))
    elif kind == "brightness":
        x = x * param
    elif kind == "contrast":
        mu = x.mean(dim=(1, 2, 3), keepdim=True); x = (x - mu) * param + mu
    elif kind == "saturation":
        lum = torch.einsum("bchw,c->bhw", x, _LUM.to(dev)).unsqueeze(1); x = lum + param * (x - lum)
    elif kind == "gamma":
        x = x.clamp(0, 1) ** param
    elif kind == "cast":
        x = x * torch.tensor(param, device=dev).view(1, 3, 1, 1)
    elif kind == "hue":
        th = math.radians(param)
        rot = torch.tensor([[1, 0, 0], [0, math.cos(th), -math.sin(th)], [0, math.sin(th), math.cos(th)]], dtype=torch.float32)
        m = (torch.linalg.inv(_YIQ) @ rot @ _YIQ).to(dev)
        x = torch.einsum("bchw,dc->bdhw", x, m)
    elif kind == "gray":
        x = torch.einsum("bchw,c->bhw", x, _LUM.to(dev)).unsqueeze(1).expand(-1, 3, -1, -1)
    elif kind == "noise":
        g = torch.Generator(device=dev).manual_seed(seed)
        x = x + torch.randn(x.shape, device=dev, generator=g) * (param / 255.0)
    elif kind == "blur":
        x = _gauss_blur(x, float(param))
    elif kind == "scale":
        h, w = x.shape[-2:]
        x = F.interpolate(x, size=(round(h * param), round(w * param)), mode="area" if param < 1 else "bilinear",
                          **({} if param < 1 else {"align_corners": False}))
    else:
        raise ValueError(kind)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e1_residual"); ap.add_argument("--tag", default="e1")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()
    limit_cpu_threads()
    dcfg = load_config("data.yaml")
    man = pd.read_csv(resolve(dcfg["paths"]["metadata"]) / "manifest.csv", keep_default_na=False, na_values=[])
    folds = pd.read_csv(PROJECT_ROOT / "data/splits/folds_v1.csv")
    role_of = folds.set_index("image_id")
    K, device = 5, select_device("cuda")
    amp = select_amp_dtype("bf16", device)
    val_ids = sorted(set().union(*[set(folds.loc[folds[f"role_fold{f}"] == "val", "image_id"]) for f in range(K)]))
    store = ImageStore.from_manifest(man[man.image_id.isin(val_ids)].reset_index(drop=True), resolve(dcfg["paths"]["raw_images"]))
    t0 = time.time()
    recs = []
    for f in range(K):
        va = fold_rows(store, folds, f, "val")
        ids = [store.image_ids[r] for r in va]
        assert (role_of.loc[ids, f"role_fold{f}"] == "val").all() and not (role_of.loc[ids, "test_fold"] == f).any()
        for s in a.seeds:
            model, pre, cfg, ep = load_run(PROJECT_ROOT / "checkpoints" / f"{a.run}_s{s}" / f"fold{f}" / "last.pt", device)
            assert cfg["fold"] == f and cfg["seed"] == s and ep == cfg["optim"]["epochs"] - 1
            for name, fam, kind, param, status in CONDITIONS:
                ps = []
                with torch.no_grad():
                    for i in range(0, len(va), a.batch):
                        x = store.images[torch.as_tensor(va[i:i + a.batch])].permute(0, 3, 1, 2).to(device)
                        xp = perturb(x, kind, param, seed=1000 * f + i)
                        ps.append(torch.softmax(predict_uint8(model, xp, pre, cfg["data"]["eval_tile"], amp), 1)[:, 1].float().cpu())
                p = torch.cat(ps).numpy()
                recs.append(pd.DataFrame({"condition": name, "fold": f, "seed": s, "image_id": ids, "patient_id": store.patient_ids[va],
                                          "label": store.labels[va], "p": p}))
            del model
        print(f"fold {f} done ({time.time() - t0:.0f}s)", flush=True)
    P = pd.concat(recs, ignore_index=True)
    meta = {n: {"family": fam, "kind": kind, "param": (list(prm) if isinstance(prm, tuple) else prm), "status": st} for n, fam, kind, prm, st in CONDITIONS}

    # per-fold AUROC (seed mean) + pooled decisions of the per-fold seed ensemble
    ens = P.groupby(["condition", "fold", "image_id", "patient_id", "label"], as_index=False).p.mean()
    clean = ens[ens.condition == "clean"].set_index(["fold", "image_id"]).p
    rows = []
    for name, *_ in CONDITIONS:
        g, e = P[P.condition == name], ens[ens.condition == name]
        fold_auc = [np.mean([auc(h.label.to_numpy(), h.p.to_numpy()) for _, h in gs.groupby("seed")]) for _, gs in g.groupby("fold")]
        y, p = e.label.to_numpy(), e.p.to_numpy()
        m = binary_metrics(y, p, 0.5)
        flips = float(((e.set_index(["fold", "image_id"]).p >= .5) != (clean.loc[e.set_index(["fold", "image_id"]).index] >= .5)).mean())
        rows.append({"condition": name, **meta[name], "mean_fold_auroc": float(np.mean(fold_auc)), "min_fold_auroc": float(np.min(fold_auc)),
                     "sensitivity": m["sensitivity"], "specificity": m["specificity"], "balanced_accuracy": m["balanced_accuracy"],
                     "confident_wrong_rate": float((((y == 1) & (p < .1)) | ((y == 0) & (p > .9))).mean()),
                     "decision_flip_rate_vs_clean": flips, "mean_p_on_ptc": float(p[y == 1].mean()), "mean_p_on_non": float(p[y == 0].mean())})
    S = pd.DataFrame(rows)
    out_json = PROJECT_ROOT / "reports" / f"07_stress_{a.tag}.json"
    write_json(out_json, {"scope": "inference only; validation partitions only; frozen epoch-29 models; 3 seeds (per-fold seed ensemble for decisions)",
                          "run": a.run, "threshold": 0.5, "seconds": round(time.time() - t0, 1), "conditions": S.round(4).to_dict("records")})
    P.to_csv(PROJECT_ROOT / "reports" / f"07_stress_{a.tag}_predictions.csv.gz", index=False)

    # figure: balanced accuracy and confident-wrong rate per condition, coloured by status
    SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
    col = {"reference": INK, "held_out": "#2a78d6", "trained_beyond": "#eb6834", "trained_in": "#9a9993"}
    lab = {"reference": "clean (reference)", "held_out": "held-out kind (never used for training)",
           "trained_beyond": "kind step 2 may train on, tested beyond the training range", "trained_in": "kind step 2 may train on, inside the training range"}
    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": .6, "axes.axisbelow": True, "font.size": 9,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 10), sharey=True, gridspec_kw={"width_ratios": [1.25, 1]})
    yy = np.arange(len(S))[::-1]
    for ax, metric, xlim, title in ((a1, "balanced_accuracy", (0.4, 1.02), "Balanced accuracy at the 0.5 cut-off"),
                                    (a2, "confident_wrong_rate", (0, max(0.05, float(S.confident_wrong_rate.max()) * 1.15)), "Confidently wrong (wrong and score <0.1 or >0.9)")):
        for st in col:
            m_ = S.status == st
            ax.scatter(S.loc[m_, metric], yy[m_.to_numpy()], color=col[st], s=34, zorder=3, label=lab[st], edgecolor=SURF, linewidth=1)
        ax.axvline(float(S.loc[S.condition == "clean", metric].iloc[0]), color=INK2, lw=.8, ls=":")
        ax.set_xlim(*xlim); ax.set_title(title); ax.grid(axis="y", visible=False)
        for y_, v in zip(yy, S[metric]):
            ax.text(v + (xlim[1] - xlim[0]) * 0.012, y_, f"{v:.2f}" if metric == "balanced_accuracy" else f"{v:.1%}", va="center", fontsize=7.5, color=INK2)
    a1.set_yticks(yy); a1.set_yticklabels(S.condition, fontsize=8.5)
    a1.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, -0.13), ncol=2)
    fig.suptitle(f"Stress test, {a.run} (validation images, inference only)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(); fig.savefig(PROJECT_ROOT / "reports" / "figures" / f"07_stress_{a.tag}.png", dpi=130); plt.close(fig)
    pd.set_option("display.width", 220)
    print(S[["condition", "status", "mean_fold_auroc", "sensitivity", "specificity", "balanced_accuracy", "confident_wrong_rate", "decision_flip_rate_vs_clean", "mean_p_on_ptc", "mean_p_on_non"]].round(3).to_string(index=False))
    print(f"\ntotal {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
