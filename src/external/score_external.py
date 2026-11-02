"""Score the extracted external regions ONCE with the deployed system (rules: reports/08_external_test.md section 1).

Primary: E3 ensemble + safety checks v3 (the deployed system), each region given with its true pixel size. Information only:
E1 behind the same checks, and both models' scores without the checks. Slides are the unit: 95% intervals resample slides.
No accuracy or AUROC across the two sources is computed (each class comes from a different programme: tissue and source are mixed).

  .venv/bin/python -I src/external/score_external.py                 # reports/08_external_*; refuses to run if they exist
  .venv/bin/python -I src/external/score_external.py --manifest <m> --out-prefix <scratch>/dry   # dry run on other images
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):          # before torch starts: run-time changes can crash CPU ops (src.common)
    os.environ.setdefault(_var, os.environ.get("PTIC_TORCH_THREADS", "8"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.common import PROJECT_ROOT, limit_cpu_threads, write_json  # noqa: E402
from src.inference.gate import MODEL_MPP, rescale_to_model  # noqa: E402
from src.inference.predict import predict_uint8  # noqa: E402
from src.inference.predictor import PTICPredictor  # noqa: E402

RESULTS = ("PTC-like", "not PTC-like", "uncertain")
N_BOOT = 10_000


@torch.no_grad()
def score_without_checks(pred: PTICPredictor, arr: np.ndarray, mpp: float) -> float:
    """The ensemble's score with the safety checks skipped (information only: what the model alone would say)."""
    x = torch.from_numpy(np.ascontiguousarray(rescale_to_model(arr, mpp, MODEL_MPP))).permute(2, 0, 1)[None].to(pred.device)
    return float(np.mean([float(torch.softmax(predict_uint8(net, x, pre, tile, pred.amp), 1)[0, 1]) for net, pre, tile in pred.members]))


def slide_bootstrap(g: pd.DataFrame, num: str, den: str, seed: int = 0) -> tuple[float, float, float]:
    """Pooled share sum(num)/sum(den) over slides, with a 95% percentile interval from resampling slides."""
    per = g.groupby("slide_id")[[num, den]].sum()
    n, d = per[num].to_numpy(float), per[den].to_numpy(float)
    est = n.sum() / d.sum() if d.sum() else float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per), (N_BOOT, len(per)))
    with np.errstate(invalid="ignore", divide="ignore"):
        bs = n[idx].sum(1) / d[idx].sum(1)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]) if np.isfinite(bs).any() else (float("nan"), float("nan"))
    return round(float(est), 4), round(float(lo), 4), round(float(hi), 4)


def summarise(R: pd.DataFrame, model: str) -> list[dict]:
    out = []
    for src, g in R[R.model == model].groupby("source"):
        g = g.assign(one=1, accepted=(g.status == "ok").astype(int), **{f"is_{r}": (g.answer == r).astype(int) for r in RESULTS})
        rec = {"source": src, "model": model, "slides": int(g.slide_id.nunique()), "regions": int(len(g)),
               "accepted_by_checks": slide_bootstrap(g, "accepted", "one")}
        for r in RESULTS:
            rec[f"{r}_among_accepted"] = slide_bootstrap(g, f"is_{r}", "accepted")
            rec[f"{r}_among_all"] = slide_bootstrap(g, f"is_{r}", "one")
        refused = g[g.status != "ok"]
        fc = refused.failed_checks.fillna("").str.split(";")
        rec["counts"] = {**{r: int(g[f"is_{r}"].sum()) for r in RESULTS}, "refused": int(len(refused))}
        rec["refusal_share_failing_check"] = (fc.explode().value_counts() / len(refused)).round(4).to_dict() if len(refused) else {}
        rec["refusal_share_failing_only_this_check"] = ((fc[fc.str.len() == 1].str[0].value_counts() / len(refused)).round(4).to_dict()
                                                         if len(refused) else {})
        rec["slides_with_any_accepted_region"] = int((g.groupby("slide_id").accepted.sum() > 0).sum())
        rec["small_denominator"] = bool(g.accepted.sum() < 20 or rec["slides_with_any_accepted_region"] < 5)   # Report 8 section 4: counts only
        g = g.assign(ns_ptc=(g.p_without_checks >= 0.8).astype(int), ns_neg=(g.p_without_checks <= 0.2).astype(int))
        rec["without_checks_info_only"] = {"score_ge_0.8": slide_bootstrap(g, "ns_ptc", "one"), "score_le_0.2": slide_bootstrap(g, "ns_neg", "one"),
                                           "median_score": round(float(g.p_without_checks.median()), 4)}
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(PROJECT_ROOT / "data" / "external" / "regions" / "manifest.csv"))
    ap.add_argument("--out-prefix", default=str(PROJECT_ROOT / "reports" / "08_external"))
    a = ap.parse_args()
    limit_cpu_threads()
    outs = [Path(a.out_prefix + s) for s in ("_regions.csv.gz", "_summary.json", "_slides.csv")]
    for p in outs:
        if p.exists():
            print(f"REFUSING: {p} already exists. The external test is scored once (Report 8, one-shot rule).")
            return 1
    t0 = time.time()
    M = pd.read_csv(a.manifest)
    root = Path(a.manifest).parent
    deployed = (PROJECT_ROOT / "artifacts" / "deploy" / "model" / "CURRENT").read_text().strip()
    assert deployed == "E3_fold_ensemble", f"deployed model is {deployed}, protocol says E3"
    preds = {"E3": PTICPredictor(model_name="E3_fold_ensemble"), "E1": PTICPredictor(model_name="E1_fold_ensemble")}
    rows = []
    for i, r in enumerate(M.itertuples(index=False)):
        assert abs(r.mpp_x / r.mpp_y - 1) < 0.01, "non-square pixels"
        path = root / r.file
        for name, pred in preds.items():
            res = pred.predict(path, mpp=float(r.mpp_x))
            if res["status"] == "ok":                                # same computation, verified identical in the dry run
                p_nc = res["probability_ptc_like"]
            else:
                p_nc = round(score_without_checks(pred, np.array(Image.open(path).convert("RGB")), float(r.mpp_x)), 4)
            rows.append({"source": r.source, "slide_id": r.slide_id, "region": r.region, "model": name, "status": res["status"],
                         "answer": res["answer"], "p": res["probability_ptc_like"],
                         "ensemble_std": (res["ensemble"] or {}).get("std"),
                         "failed_checks": ";".join(k for k, c in res["checks"].items() if not c["passed"]) or None,
                         "refusal_reasons": " | ".join(res["refusal_reasons"]) or None,
                         "p_without_checks": p_nc,
                         **{f"check_{k}": json.dumps(c["value"]) for k, c in res["checks"].items()}})
        if (i + 1) % 50 == 0:
            print(f"{i + 1}/{len(M)} regions ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows)
    S = R[R.model == "E3"].groupby(["source", "slide_id"]).agg(
        regions=("status", "size"), accepted=("status", lambda s: int((s == "ok").sum())),
        ptc_like=("answer", lambda s: int((s == "PTC-like").sum())), not_ptc_like=("answer", lambda s: int((s == "not PTC-like").sum())),
        uncertain=("answer", lambda s: int((s == "uncertain").sum())), median_score_without_checks=("p_without_checks", "median")).reset_index()
    summary = {"protocol": "reports/08_external_test.md section 1 (declared before extraction and scoring)",
               "primary": "E3 + safety checks v3 (deployed)", "info_only": ["E1 + safety checks v3", "scores without checks"],
               "answer_band": list(preds["E3"].band), "versions": {k: p.versions for k, p in preds.items()},
               "intervals": f"95% percentile intervals, {N_BOOT} resamples of slides (seed 0); each value is [estimate, low, high]",
               "E3": summarise(R, "E3"), "E1_info_only": summarise(R, "E1"), "seconds": round(time.time() - t0, 1)}
    R.to_csv(outs[0], index=False)                                  # outputs written last
    write_json(outs[1], summary)
    S.to_csv(outs[2], index=False)
    print(json.dumps(summary["E3"], indent=1))
    print(f"done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
