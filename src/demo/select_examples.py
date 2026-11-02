"""Choose the web page's sample images (rule fixed in this docstring before any heat-map number was looked at).

The page shows only images the model answers correctly. This is a selection for display. It says nothing about accuracy: the models' error rates are in Reports 6 and 8, and the images they get wrong
(patient 1, other hospitals, simulated problems) are deliberately not on the page.

Pool         PTIC images of every patient except patient 1; the dataset label is the truth the answer is compared with.
Scoring      Exactly what the page does for a PTIC file: the 3 models that never saw the image's patient (fair mode), 40x, all
             safety checks. Run on the CPU (the GPU is shared); the page's GPU score differs by about 0.01 at most.
Keep         status ok, the answer equals the label, and P >= 0.95 (PTC-like) or P <= 0.05 (not PTC-like), so a small score
             difference between GPU and CPU cannot change the answer.
Heat map     The page's own: live_maps -> evidence() on the fixed colour scale of the classic figures (never rescaled per image).
             "Some" = evidence >= 0.40 of that scale (jet: cyan-green and above), "strong" = >= 0.75 (orange to red).
What the page shows (final, 2026-10-29): the six PTIC images it showed before that day (3 PTC-like and 3 not PTC-like, from 6 patients,
             drawn at random with seed 2026), re-scored here with the rule above (`--earlier-six`). Their maps are moderate (see
             selection.json, "earlier_six").
The trial     The default run first scored up to 10 images per PTC-like patient (100 images) and showed the five patients whose best
             image had the largest share of its map at "some" or above (five images, patients 11, 24, 13, 25 and 18; two more not
             PTC-like images were added). The owner found those maps too dense and asked for the earlier page back, so that pick is
             kept in selection.json ("dense_trial", "candidates") and is not shown.

  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES= .venv/bin/python -I src/demo/select_examples.py [--earlier-six]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import PROJECT_ROOT, limit_cpu_threads  # noqa: E402
from src.explain.gradcam_classic import classic_scale, evidence, live_maps  # noqa: E402
from src.inference.predictor import PTICPredictor  # noqa: E402

OUT = PROJECT_ROOT / "artifacts" / "demo_pack" / "selection.json"
EARLIER_PACK = PROJECT_ROOT / "backups" / "demo_before_two_images_2026-10-29" / "demo_pack" / "examples.json"
SEED, PER_PATIENT, MARGIN, SOME, STRONG = 2026, 10, 0.95, 0.40, 0.75
N_PTC, N_NON = 5, 5


def earlier_non_ptc() -> list[str]:
    """The three not-PTC-like images that the earlier page showed (random draw, seed 2026), read from the backed-up pack."""
    p = PROJECT_ROOT / "backups" / "demo_before_two_images_2026-10-29" / "demo_pack" / "examples.json"
    ex = json.loads(p.read_text())["examples"]
    return [e["file"].removeprefix("data/raw/images/") for e in ex if e["id"] in ("ptic_4", "ptic_5", "ptic_6")]


def score(pred, scale: float, row: pd.Series) -> dict:
    path = PROJECT_ROOT / "data" / "raw" / "images" / row.image_id
    fold = int(row.test_fold)
    res = pred.predict(path, magnification="40x", member_folds={fold})
    out = {"image_id": row.image_id, "patient_id": int(row.patient_id), "label": "PTC-like" if row.label == 1 else "not PTC-like",
           "test_fold": fold, "status": res["status"], "answer": res["answer"], "p": res["probability_ptc_like"]}
    if res["status"] != "ok":
        out["refusal"] = res["refusal_reasons"]
        return out
    out["confidence"] = round(out["p"] if res["answer"] == "PTC-like" else 1 - out["p"], 4)
    out["correct"] = res["answer"] == out["label"] and out["confidence"] >= MARGIN
    if out["correct"] and row.label == 1:                 # the heat map matters for PTC-like images only
        _, ens, _, _, _ = live_maps(pred, path.read_bytes(), None, "40x", {fold})
        u = evidence(ens) / scale
        out.update(some=round(float((u >= SOME).mean()), 4), strong=round(float((u >= STRONG).mean()), 4), mean_u=round(float(u.mean()), 4))
    return out


def earlier_six() -> int:
    """Show the earlier page's six PTIC images again (the owner found the trial's picks too dense, 2026-10-29): score them with the
    same rule, keep the trial's numbers in selection.json and point `ptc_like` / `not_ptc_like` at the six."""
    limit_cpu_threads()
    sel = json.loads(OUT.read_text())
    man = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "manifest.csv", keep_default_na=False, na_values=[])
    pred = PTICPredictor(device="cpu")
    scale = classic_scale()
    assert scale is not None
    six = [e["file"].removeprefix("data/raw/images/") for e in json.loads(EARLIER_PACK.read_text())["examples"] if e["group"] == "PTIC image"]
    rows = [score(pred, scale, man[man.image_id == iid].iloc[0]) for iid in six]
    assert all(r.get("correct") for r in rows), "every shown image must be answered correctly"
    if "dense_trial" not in sel:
        sel["dense_trial"] = {"ptc_like": sel["ptc_like"], "not_ptc_like": sel["not_ptc_like"],
                              "status": "rejected by the owner on 2026-10-29: the maps were too dense; not shown"}
    sel["ptc_like"] = [r["image_id"] for r in rows if r["label"] == "PTC-like"]
    sel["not_ptc_like"] = [r["image_id"] for r in rows if r["label"] != "PTC-like"]
    sel["earlier_six"] = rows
    OUT.write_text(json.dumps(sel, indent=1))
    for r in rows:
        print(f"patient {r['patient_id']:>2} {r['label']:13s} {r['image_id']:28s} {r['answer']:12s} p={r['p']} some={r.get('some')} strong={r.get('strong')}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--earlier-six", action="store_true", help="show the earlier page's six PTIC images (the final choice) instead of the dense trial")
    if ap.parse_args().earlier_six:
        return earlier_six()
    limit_cpu_threads()
    man = pd.read_csv(PROJECT_ROOT / "data" / "metadata" / "manifest.csv", keep_default_na=False, na_values=[])
    man = man[man.patient_id.astype(int) != 1]
    rng = np.random.default_rng(SEED)
    pred = PTICPredictor(device="cpu")
    scale = classic_scale()
    assert scale is not None
    rows, t0 = [], time.time()

    # PTC-like candidates
    ptc = man[man.label == 1]
    for pid in sorted(set(ptc.patient_id.astype(int))):
        ids = sorted(ptc[ptc.patient_id.astype(int) == pid].image_id)
        for iid in rng.choice(ids, min(PER_PATIENT, len(ids)), replace=False):
            rows.append(score(pred, scale, man[man.image_id == iid].iloc[0]))
            print(f"[{time.time() - t0:5.0f}s] patient {pid:2d} {rows[-1]['image_id']:28s} {rows[-1]['answer']} p={rows[-1]['p']} "
                  f"some={rows[-1].get('some')} strong={rows[-1].get('strong')}", flush=True)

    # not PTC-like: the earlier three, then two more from other patients
    keep = earlier_non_ptc()
    for iid in keep:
        rows.append(score(pred, scale, man[man.image_id == iid].iloc[0]))
    used = {int(man[man.image_id == i].patient_id.iloc[0]) for i in keep}
    non = man[man.label == 0]
    pats = [int(p) for p in rng.permutation(sorted(set(non.patient_id.astype(int)) - used))]
    extra = []
    for pid in pats:
        if len(extra) >= N_NON - len(keep):
            break
        ids = sorted(non[non.patient_id.astype(int) == pid].image_id)
        for iid in rng.choice(ids, min(3, len(ids)), replace=False):
            r = score(pred, scale, man[man.image_id == iid].iloc[0])
            rows.append(r)
            print(f"[{time.time() - t0:5.0f}s] non-PTC patient {pid:2d} {r['image_id']:28s} {r['answer']} p={r['p']}", flush=True)
            if r.get("correct"):
                extra.append(iid)
                break

    df = pd.DataFrame(rows)
    good = df[df.correct == True]                                                   # noqa: E712
    best = good[good.label == "PTC-like"].sort_values(["some", "strong"], ascending=False).groupby("patient_id", sort=False).head(1)
    ptc_pick = list(best.head(N_PTC).image_id)
    non_pick = [i for i in keep if df[df.image_id == i].correct.eq(True).iloc[0]] + extra
    sel = {"rule": __doc__.split("\n\n")[0:2], "seed": SEED, "per_patient": PER_PATIENT, "margin": MARGIN, "some_at": SOME, "strong_at": STRONG,
           "colour_scale": scale, "device": "cpu", "ptc_like": ptc_pick, "not_ptc_like": non_pick[:N_NON],
           "candidates": json.loads(df.to_json(orient="records")),
           "counts": {"ptc_candidates": int((df.label == "PTC-like").sum()), "ptc_correct_with_margin": int(((df.label == "PTC-like") & (df.correct == True)).sum()),  # noqa: E712
                      "non_candidates": int((df.label == "not PTC-like").sum()), "non_correct_with_margin": int(((df.label == "not PTC-like") & (df.correct == True)).sum())}}  # noqa: E712
    OUT.write_text(json.dumps(sel, indent=1))
    print("\nselected PTC-like:", ptc_pick, "\nselected not PTC-like:", sel["not_ptc_like"], "\ncounts:", sel["counts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
