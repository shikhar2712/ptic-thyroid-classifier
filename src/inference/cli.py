"""Command-line demo: one JSON result per image.

  python -I src/inference/cli.py path/to/patch.jpg --magnification 40x
  python -I src/inference/cli.py scan.tif                      # micrometres per pixel read from TIFF/SVS metadata if present
  python -I src/inference/cli.py a.jpg b.jpg --mpp 0.25 --out results.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

from src.inference.predictor import PTICPredictor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="PTIC research demo (not a medical device)")
    ap.add_argument("images", nargs="+")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--mpp", type=float, help="micrometres per pixel of the input image")
    g.add_argument("--magnification", help="scanner objective, e.g. 40x or 20x (40x is taken as 0.26 um/px)")
    ap.add_argument("--device", default=None); ap.add_argument("--model", default=None); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pred = PTICPredictor(model_name=a.model, device=a.device)
    lines = []
    for img in a.images:
        r = pred.predict(img, mpp=a.mpp, magnification=a.magnification)
        r["file"] = img
        lines.append(json.dumps(r))
        print(json.dumps(r, indent=1) if len(a.images) == 1 else json.dumps({k: r[k] for k in ("file", "status", "answer", "probability_ptc_like", "refusal_reasons")}))
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
