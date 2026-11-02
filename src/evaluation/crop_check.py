"""Does the gate behave on realistic DEMO inputs it was never validated on? (VALIDATION partitions only; inference only.)

Variants built from each fold's validation images, judged by that fold's gate (fitted on the fold's training patients):
  glassfree512       the 512x512 window with the least glass (bright, neutral pixels) - a crop inside dense tissue
  glassfree512_warm  the same window with a 6% warm colour cast (the model's biggest weakness)
  random512 / random768   random crops (fewer tiles than the full 28-tile patches the thresholds were set on)
  declared20x        the whole image downsampled x2 (a 20x scan), declared as 20x -> rescaled x2 by the predictor
Per variant and class: gate acceptance and refusal reasons, plus E1 and E3 accuracy on the accepted inputs.

  python -I src/evaluation/crop_check.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.common import PROJECT_ROOT, limit_cpu_threads, load_config, resolve, write_json  # noqa: E402
from src.data.dataset import ImageStore  # noqa: E402
from src.evaluation.oof_validation import load_run  # noqa: E402
from src.evaluation.stress_test import perturb  # noqa: E402
from src.inference.calibrate_gate import thresholds_from  # noqa: E402
from src.inference.gate import GenericFeatures, evaluate_checks, fit_gate, gate_distances, image_stats, rescale_to_model  # noqa: E402
from src.inference.predict import predict_uint8  # noqa: E402

BAND = (0.2, 0.8)


def glassfree_window(x: torch.Tensor, side: int = 512, stride: int = 128) -> tuple[int, int, float]:
    """x uint8 [H,W,3] -> top-left of the side x side window with the smallest glass fraction, and that fraction."""
    f = x.float(); mn, mx = f.min(2).values, f.max(2).values
    glass = ((mn > 215) & ((mx - mn) < 20)).float()[None, None]
    frac = F.avg_pool2d(glass, side, stride=stride)[0, 0]
    k = int(torch.argmin(frac)); r, c = divmod(k, frac.shape[1])
    return r * stride, c * stride, float(frac.view(-1)[k])


def main() -> int:
    limit_cpu_threads()
    t0 = time.time()
    dcfg = load_config("data.yaml")
    man = pd.read_csv(resolve(dcfg["paths"]["metadata"]) / "manifest.csv", keep_default_na=False, na_values=[])
    folds = pd.read_csv(PROJECT_ROOT / "data/splits/folds_v1.csv")
    man = man[man.image_id.isin(folds.image_id)].reset_index(drop=True)
    store = ImageStore.from_manifest(man, resolve(dcfg["paths"]["raw_images"]))
    dev = torch.device("cuda"); amp = torch.bfloat16
    gf = GenericFeatures(dev)
    FE = np.load(PROJECT_ROOT / "data/processed/gate_clean_features.npz", allow_pickle=True)["feats"]
    z = np.load(PROJECT_ROOT / "data/processed/gate_clean_stats_v3.npz", allow_pickle=True)
    ST, FB1, FB2 = pd.DataFrame(list(z["stats"])), z["fb1"], z["fb2"]
    rows = []
    for f in range(5):
        role = folds.set_index("image_id").loc[man.image_id, f"role_fold{f}"].to_numpy()
        tr, va = np.where(role == "train")[0], np.where(role == "val")[0]
        p = fit_gate(FE[tr]); p.thresholds = thresholds_from(FE[tr], ST.iloc[tr].reset_index(drop=True), FB1[tr], FB2[tr], man.patient_id.to_numpy()[tr])
        models = {name: [load_run(PROJECT_ROOT / "checkpoints" / f"{run}_s{s}" / f"fold{f}" / "last.pt", dev)[:2] for s in range(3)]
                  for name, run in (("E1", "e1_residual"), ("E3", "e3_robust"))}
        rng = np.random.default_rng(f)
        for r in va:
            x = store.images[int(r)].to(dev)                                     # [H,W,3] uint8, 0.26 um/px
            H, W = x.shape[:2]
            gy, gx, gfrac = glassfree_window(x)
            win = x[gy:gy + 512, gx:gx + 512]
            y5, x5 = int(rng.integers(0, H - 512 + 1)), int(rng.integers(0, W - 512 + 1))
            y7, x7 = int(rng.integers(0, H - 768 + 1)), int(rng.integers(0, W - 768 + 1))
            half = perturb(x.permute(2, 0, 1)[None], "scale", 0.5)[0].permute(1, 2, 0).cpu().numpy()   # a 20x scan of the same tissue
            variants = {"glassfree512": (win, 0.26), "glassfree512_warm": (perturb(win.permute(2, 0, 1)[None], "cast", (1.06, 1.0, 0.94))[0].permute(1, 2, 0), 0.26),
                        "random512": (x[y5:y5 + 512, x5:x5 + 512], 0.26), "random768": (x[y7:y7 + 768, x7:x7 + 768], 0.26),
                        "declared20x": (torch.from_numpy(rescale_to_model(half, 0.52)).to(dev), 0.52)}
            for vname, (img, mpp) in variants.items():
                st = image_stats(img, dev)
                d = gate_distances(p, gf(img.cpu().numpy()))
                checks = evaluate_checks(p, mpp, tuple(img.shape[:2]), st, d)
                failed = [k for k, v in checks.items() if not v["passed"]]
                rec = {"variant": vname, "fold": f, "image_id": store.image_ids[int(r)], "label": int(store.labels[int(r)]),
                       "accepted": not failed, "failed_checks": ";".join(failed), "window_glass_fraction": gfrac if vname.startswith("glassfree") else None,
                       "white_balance_measured": checks.get("white_balance_neutral", {}).get("value") is not None}
                xt = img.permute(2, 0, 1)[None].contiguous()
                for name, members in models.items():
                    with torch.no_grad():
                        ps = [float(torch.softmax(predict_uint8(m, xt, pre, 256, amp), 1)[0, 1]) for m, pre in members]
                    rec[f"p_{name}"] = float(np.mean(ps))
                rows.append(rec)
        print(f"fold {f} done ({time.time() - t0:.0f}s)", flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(PROJECT_ROOT / "reports" / "07_crop_check.csv.gz", index=False)
    summ = []
    for (v, lab), g in D.groupby(["variant", "label"]):
        acc = g.accepted.to_numpy()
        rec = {"variant": v, "class": "PTC-like" if lab == 1 else "Non-PTC-like", "n": int(len(g)), "accepted": float(acc.mean()),
               "white_balance_measured": float(g.white_balance_measured.mean()),
               "top_refusal_reasons": g.loc[~g.accepted, "failed_checks"].str.split(";").explode().value_counts().head(3).to_dict()}
        for name in ("E1", "E3"):
            pr = g[f"p_{name}"].to_numpy()
            right = (pr >= .5).astype(int) == lab
            answered = acc & ~((pr > BAND[0]) & (pr < BAND[1]))
            conf_wrong = (pr < .1) if lab == 1 else (pr > .9)
            rec[f"{name}_accuracy_no_gate"] = float(right.mean())
            rec[f"{name}_accuracy_on_answered"] = float(right[answered].mean()) if answered.any() else None
            rec[f"{name}_confidently_wrong_per_input_with_gate"] = float((conf_wrong & answered).mean())
        summ.append(rec)
    S = pd.DataFrame(summ)
    write_json(PROJECT_ROOT / "reports" / "07_crop_check.json", {"scope": "validation partitions, fold gates fitted on training patients, inference only",
                                                                "per_variant_class": S.round(4).to_dict("records"), "seconds": round(time.time() - t0, 1)})
    pd.set_option("display.width", 250)
    print(S.drop(columns=["top_refusal_reasons"]).round(3).to_string(index=False))
    for r in summ:
        if r["top_refusal_reasons"]:
            print(f"  {r['variant']:18s} {r['class']:13s} refusals: {r['top_refusal_reasons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
