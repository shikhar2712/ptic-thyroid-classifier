"""Step 3: calibrate and evaluate the input gate by patient-grouped cross-fitting (VALIDATION partitions only), then fit the
deployment gate. Classifier-independent; joins the stress-test predictions only to report what the gate prevents.

Per outer fold f: thresholds come from the fold's TRAINING patients, the gate is fitted on the training partition, and acceptance is
measured on the fold's VALIDATION partition (never the same images).
Gate threshold rules (v2 declared before v2 was evaluated; v3 changes only HOW sharpness is measured - half resolution - not
the focus rule; v1 rules are in reports/07_robustness.md section 3):
  appearance distance  <= 99.5th percentile of out-of-fold training distances (inner 4-fold patient split)
  colour present       saturation >= 0.5 x training 0.5th percentile
  H&E-like colours     he_hue_share >= 0.8 x training 0.5th percentile
  white balance        glass colour (rb, g) inside the training 0.5-99.5% range widened by 50% of its width on each side
  noise                glass noise <= 1.25 x training 99.5th percentile
  focus                midway between training images blurred by sigma 1 px (1st percentile) and sigma 2 px (99th percentile)
  answer band          P >= 0.8 "PTC-like", P <= 0.2 "not PTC-like", otherwise "uncertain"

  python -I src/inference/calibrate_gate.py --stress-tag e1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

from src.common import limit_cpu_threads, PROJECT_ROOT, load_config, resolve, write_json  # noqa: E402
from src.data.dataset import _HED_ROWS, ImageStore  # noqa: E402
from src.evaluation.stress_test import CONDITIONS, _gauss_blur, perturb  # noqa: E402
from src.inference.gate import (GateParams, GenericFeatures, evaluate_checks, fit_gate, gate_distances, image_stats,  # noqa: E402
                                mahalanobis, read_image)

Q_DIST, BAND = 0.995, (0.2, 0.8)
K = 5


# ------------------------------------------------------------------------------------------ extra negatives (gate only)
def dab_remap(x_u8: torch.Tensor) -> torch.Tensor:
    """Move the eosin signal into the DAB (brown) channel: an IHC-like brown + blue image."""
    x = x_u8.float() / 255
    hed = torch.from_numpy(_HED_ROWS).to(x.device); inv = torch.linalg.inv(hed)
    conc = torch.einsum("bchw,cd->bdhw", -torch.log(x.clamp(1 / 255, 1)), inv)
    conc[:, 2], conc[:, 1] = conc[:, 1] * 1.5, 0
    return (torch.exp(-torch.einsum("bdhw,dc->bchw", conc, hed).clamp_min(0)).clamp(0, 1) * 255).round().to(torch.uint8)


def mostly_glass(x_u8: torch.Tensor) -> torch.Tensor:
    y = x_u8.clone(); w = y.shape[-1]
    y[..., : int(w * 0.8)] = torch.tensor([238, 236, 241], dtype=torch.uint8, device=y.device).view(1, 3, 1, 1)
    return y


def screenshot(x_u8: torch.Tensor) -> torch.Tensor:
    """Tissue shown at 60% inside a viewer-like frame: dark toolbar, grey side panel with text-like stripes, black margin."""
    B, _, H, W = x_u8.shape
    y = torch.full_like(x_u8, 30)
    tis = F.interpolate(x_u8.float(), scale_factor=0.6, mode="area").round().to(torch.uint8)
    th, tw = tis.shape[-2:]
    top, left = int(H * 0.12), int(W * 0.22)
    y[..., top:top + th, left:left + tw] = tis[..., : H - top, : W - left]
    y[..., :top, :] = 60
    y[..., top:, :left] = 210
    for k in range(top + 20, H - 20, 28):
        y[..., k:k + 6, 20:left - 30] = 90
    return y


EXTRA = [("negative_DAB_brown_stain", dab_remap), ("negative_heavy_blur_sigma5", lambda x: (lambda f: (f.clamp(0, 1) * 255).round().to(torch.uint8))(_gauss_blur(x.float() / 255, 5.0))),
         ("negative_mostly_glass", mostly_glass), ("negative_viewer_screenshot", screenshot)]


def stats_feats(gf: GenericFeatures, x_u8: torch.Tensor) -> tuple[list[dict], np.ndarray]:
    st = [image_stats(xi.permute(1, 2, 0), device=x_u8.device) for xi in x_u8]
    return st, gf.batch(x_u8)


def thresholds_from(feat_X: np.ndarray, st: pd.DataFrame, focus_b1: np.ndarray, focus_b2: np.ndarray, groups: np.ndarray) -> dict:
    """All v2 thresholds from TRAINING images only (see module docstring)."""
    of = np.zeros(len(feat_X))
    for tr, ho in GroupKFold(n_splits=4).split(feat_X, groups=groups):
        p = fit_gate(feat_X[tr])
        of[ho] = mahalanobis(feat_X[ho], np.array(p.feat_mean), np.array(p.feat_prec))

    def widened(v: pd.Series) -> list[float]:
        v = v.dropna(); lo, hi = float(v.quantile(0.005)), float(v.quantile(0.995)); w = hi - lo
        return [lo - 0.5 * w, hi + 0.5 * w]
    b1, b2 = float(np.quantile(focus_b1, 0.01)), float(np.quantile(focus_b2, 0.99))
    return {"appearance_distance_max": float(np.quantile(of, Q_DIST)),
            "saturation_min": 0.5 * float(st.saturation_mean.quantile(0.005)),
            "he_hue_share_min": 0.8 * float(st.he_hue_share.dropna().quantile(0.005)),
            "white_point_rb_range": widened(st.white_point_rb), "white_point_g_range": widened(st.white_point_g),
            "glass_noise_max": 1.25 * float(st.glass_noise.dropna().quantile(0.995)),
            "focus_ratio_min": (b1 + b2) / 2 if b1 > b2 else b1, "focus_sigma1_q01": b1, "focus_sigma2_q99": b2}


def decide(p: GateParams, st: dict, feat: np.ndarray, mpp: float = 0.26, shape: tuple[int, int] | None = None) -> tuple[bool, list[str]]:
    checks = evaluate_checks(p, mpp, shape, st, gate_distances(p, feat))
    failed = [k for k, v in checks.items() if not v["passed"]]
    return (not failed), failed


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--stress-tag", default="e1"); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--version", default="v3")
    a = ap.parse_args()
    limit_cpu_threads()
    t0 = time.time()
    dcfg = load_config("data.yaml")
    man = pd.read_csv(resolve(dcfg["paths"]["metadata"]) / "manifest.csv", keep_default_na=False, na_values=[])
    folds = pd.read_csv(PROJECT_ROOT / "data/splits/folds_v1.csv")
    man = man[man.image_id.isin(folds.image_id)].reset_index(drop=True)
    store = ImageStore.from_manifest(man, resolve(dcfg["paths"]["raw_images"]))
    dev = torch.device("cuda")
    gf = GenericFeatures(dev)

    # ---- clean generic features (unchanged since v1: reuse cache) + v2 statistics + focus ratios after sigma 1/2 blur ---------
    cache = PROJECT_ROOT / "data" / "processed" / "gate_clean_features.npz"
    cache2 = PROJECT_ROOT / "data" / "processed" / f"gate_clean_stats_{a.version}.npz"
    if cache.exists():
        FE = np.load(cache, allow_pickle=True)["feats"]
    else:
        fes = []
        for i in range(0, len(store), a.batch):
            fes.append(gf.batch(store.images[i:i + a.batch].permute(0, 3, 1, 2).to(dev)))
        FE = np.concatenate(fes); cache.parent.mkdir(parents=True, exist_ok=True); np.savez(cache, feats=FE)
    if cache2.exists():
        z = np.load(cache2, allow_pickle=True); ST, FB1, FB2 = pd.DataFrame(list(z["stats"])), z["fb1"], z["fb2"]
    else:
        sts, fb1, fb2 = [], [], []
        for i in range(len(store)):
            x = store.images[i].to(dev)
            sts.append(image_stats(x, dev))
            xb = x.permute(2, 0, 1)[None]
            fb1.append(image_stats(perturb(xb, "blur", 1.0)[0].permute(1, 2, 0), dev)["focus_ratio"])
            fb2.append(image_stats(perturb(xb, "blur", 2.0)[0].permute(1, 2, 0), dev)["focus_ratio"])
        ST, FB1, FB2 = pd.DataFrame(sts), np.array(fb1), np.array(fb2)
        np.savez(cache2, stats=np.array(sts, dtype=object), fb1=FB1, fb2=FB2)
    print(f"clean features + v2 statistics ready ({time.time() - t0:.0f}s)", flush=True)

    rows, fold_gates = [], {}
    role_of = folds.set_index("image_id")
    for f in range(K):
        tr = store.rows(folds.loc[folds[f"role_fold{f}"] == "train", "image_id"].tolist())
        va = store.rows(folds.loc[folds[f"role_fold{f}"] == "val", "image_id"].tolist())
        p = fit_gate(FE[tr]); p.thresholds = thresholds_from(FE[tr], ST.iloc[tr].reset_index(drop=True), FB1[tr], FB2[tr], store.patient_ids[tr])
        fold_gates[f] = p
        # clean validation + every stress condition + extra negatives, all built from this fold's validation images
        conds = [(n, kind, prm) for n, fam, kind, prm, st in CONDITIONS] + [(n, "extra", fn) for n, fn in EXTRA]
        for name, kind, prm in conds:
            for i in range(0, len(va), a.batch):
                rr = va[i:i + a.batch]
                x = store.images[torch.as_tensor(rr)].permute(0, 3, 1, 2).to(dev)
                xp = prm(x) if kind == "extra" else perturb(x, kind, prm, seed=1000 * f + i)
                st_, fe_ = stats_feats(gf, xp)
                for j, r in enumerate(rr):
                    ok, failed = decide(p, st_[j], fe_[j], 0.26, tuple(xp.shape[-2:]))
                    rows.append({"condition": name, "fold": f, "image_id": store.image_ids[r], "patient_id": int(store.patient_ids[r]),
                                 "label": int(store.labels[r]), "accepted": ok, "failed_checks": ";".join(failed)})
        print(f"fold {f} gate evaluated ({time.time() - t0:.0f}s)", flush=True)
    G = pd.DataFrame(rows)
    assert all((role_of.loc[g.image_id, f"role_fold{int(f_)}"] == "val").all() for f_, g in G.groupby("fold"))

    # ---- unrelated images and input-contract negatives (each fold gate; declared mpp 0.26 = worst case: user claims 40x) ----
    ext = []
    files = sorted(set(Path("/workspace/tutorials").rglob("*.png")) | set(Path("/workspace/tutorials").rglob("*.jpg")) | set(Path("/opt/pytorch/vision").rglob("*.jpg")))
    files = [f_ for f_ in files if f_.stat().st_size > 20_000]
    for fp in files:
        arr, info = read_image(fp)
        if arr is None or min(arr.shape[:2]) < 64:
            continue
        x = torch.from_numpy(arr).permute(2, 0, 1)[None].to(dev)
        st_, fe_ = stats_feats(gf, x)
        for f, p in fold_gates.items():
            ok, failed = decide(p, st_[0], fe_[0], 0.26, tuple(arr.shape[:2]))
            ext.append({"input": str(fp.relative_to(fp.parents[2])), "fold_gate": f, "accepted": ok, "failed_checks": ";".join(failed)})
    E = pd.DataFrame(ext)
    p0 = fold_gates[0]
    contract = {
        "no_scale_given": evaluate_checks(p0, None, None, None, None)["scale_known"]["passed"],
        "declared_10x_(1.04um/px)": evaluate_checks(p0, 1.04, None, None, None)["magnification_sufficient"]["passed"],
        "declared_20x_(0.52um/px)": evaluate_checks(p0, 0.52, None, None, None)["magnification_sufficient"]["passed"],
        "tiny_200x200_crop_at_40x": evaluate_checks(p0, 0.26, (200, 200), None, None)["size_sufficient"]["passed"],
    }

    # ---- join with the classifier's stress predictions: what does the gate (+ uncertain band) prevent? ----------------
    P = pd.read_csv(PROJECT_ROOT / "reports" / f"07_stress_{a.stress_tag}_predictions.csv.gz")
    ens = P.groupby(["condition", "fold", "image_id", "label"], as_index=False).p.mean()
    J = G.merge(ens, on=["condition", "fold", "image_id", "label"], how="left")
    summ = []
    status = {n: st for n, fam, kind, prm, st in CONDITIONS}
    for name, g in J.groupby("condition", sort=False):
        y, pr, acc = g.label.to_numpy(), g.p.to_numpy(), g.accepted.to_numpy()
        rec = {"condition": name, "status": status.get(name, "gate_negative"), "n": int(len(g)),
               "accepted_all": float(acc.mean()), "accepted_ptc": float(acc[y == 1].mean()), "accepted_non": float(acc[y == 0].mean()),
               "top_reasons": g.loc[~g.accepted, "failed_checks"].str.split(";").explode().value_counts().head(3).to_dict()}
        if not np.isnan(pr).all():
            wrong = (pr >= .5).astype(int) != y
            conf_wrong = ((y == 1) & (pr < .1)) | ((y == 0) & (pr > .9))
            band = (pr > BAND[0]) & (pr < BAND[1])
            answered = acc & ~band
            rec.update({"no_gate_error_rate": float(wrong.mean()), "no_gate_confident_wrong": float(conf_wrong.mean()),
                        "gated_answered_share": float(answered.mean()), "gated_uncertain_share": float((acc & band).mean()),
                        "gated_error_rate_among_answered": float(wrong[answered].mean()) if answered.any() else None,
                        "gated_wrong_answers_per_input": float((wrong & answered).mean()),
                        "gated_confident_wrong_per_input": float((conf_wrong & acc).mean())})
        summ.append(rec)
    S = pd.DataFrame(summ)
    ext_summary = {"inputs": int(E.input.nunique()), "rejected_share_mean_over_fold_gates": float(1 - E.accepted.mean()),
                   "accepted_examples": E[E.accepted].input.value_counts().head(10).to_dict(),
                   "top_reasons": E.loc[~E.accepted, "failed_checks"].str.split(";").explode().value_counts().head(5).to_dict()}

    # ---- deployment gate: fitted on all 1,006 images; thresholds from 5-fold patient-grouped out-of-fold distances -------
    pdep = fit_gate(FE); pdep.thresholds = thresholds_from(FE, ST, FB1, FB2, store.patient_ids)
    pdep.meta = {"gate_version": int(a.version.lstrip("v")), "fitted_on": "all 1006 PTIC images (24 patients) at 0.26 um/px", "generic_model": "torchvision resnet18 IMAGENET1K_V1",
                 "rules": "see src/inference/calibrate_gate.py docstring", "answer_band": list(BAND), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out_gate = PROJECT_ROOT / "artifacts" / "deploy" / "gate_params.json"; pdep.save(out_gate)

    write_json(PROJECT_ROOT / "reports" / f"07_gate_results_{a.version}.json", {
        "scope": "cross-fitted on validation partitions (thresholds from training partitions); classifier joined only to measure effects",
        "declared": {"rules": "see calibrate_gate.py docstring (v2)", "answer_band": list(BAND)},
        "fold_thresholds": {f: p.thresholds for f, p in fold_gates.items()}, "deployment_thresholds": pdep.thresholds, "limits": pdep.limits,
        "per_condition": S.round(4).to_dict("records"), "unrelated_images": ext_summary, "input_contract_checks_pass": contract,
        "seconds": round(time.time() - t0, 1)})
    G.to_csv(PROJECT_ROOT / "reports" / f"07_gate_decisions_{a.version}.csv.gz", index=False)
    pd.set_option("display.width", 250)
    print(S.drop(columns=["top_reasons"]).round(3).to_string(index=False))
    print("\nunrelated images:", ext_summary); print("input contract (False = refused):", contract)
    print("deployment thresholds:", pdep.thresholds, "| saved", out_gate.relative_to(PROJECT_ROOT), f"| {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
