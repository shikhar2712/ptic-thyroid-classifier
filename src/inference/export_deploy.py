"""Package a trained run as a deployable ensemble (weights only + what each member needs to preprocess), with checksums.

Each (fold, seed) epoch-29 model becomes one ensemble member with its OWN normalisation statistics (from its training patients).
Writes artifacts/deploy/model/<name>/{member_f<k>_s<s>.pt, manifest.json} and points artifacts/deploy/model/CURRENT at it.

  python -I src/inference/export_deploy.py --run e1_residual --name E1_fold_ensemble
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import torch  # noqa: E402

from src.common import PROJECT_ROOT  # noqa: E402


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="e1_residual"); ap.add_argument("--name", default="E1_fold_ensemble")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--set-current", action="store_true", help="point artifacts/deploy/model/CURRENT at this export")
    a = ap.parse_args()
    out = PROJECT_ROOT / "artifacts" / "deploy" / "model" / a.name
    out.mkdir(parents=True, exist_ok=True)
    members = []
    for f in a.folds:
        for s in a.seeds:
            src = PROJECT_ROOT / "checkpoints" / f"{a.run}_s{s}" / f"fold{f}" / "last.pt"
            ck = torch.load(src, map_location="cpu", weights_only=False)
            cfg = ck["config"]
            assert cfg["fold"] == f and cfg["seed"] == s and ck["state"]["epoch"] == cfg["optim"]["epochs"] - 1, src
            dst = out / f"member_f{f}_s{s}.pt"
            torch.save({"model": ck["model"], "model_cfg": cfg["model"], "data_cfg": cfg["data"], "aug_cfg": cfg["aug"],
                        "channel_mean": ck["channel_mean"], "channel_std": ck["channel_std"]}, dst)
            members.append({"file": dst.name, "sha256": sha256(dst), "fold": f, "seed": s, "source_checkpoint": str(src.relative_to(PROJECT_ROOT)),
                            "source_sha256": sha256(src)})
    manifest = {"name": a.name, "run": a.run, "members": members, "n_members": len(members), "epoch": 29,
                "input_mpp": 0.26,                                   # the predictor always hands members images at 0.26 um/px
                "note": "Ensemble of the cross-validation fold models (each trained on ~60% of the 24 patients). There is no held-out test "
                        "of this ensemble itself: every PTIC image was in the training data of some members.",
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    if a.set_current:
        (out.parent / "CURRENT").write_text(a.name + "\n")
    print(f"exported {len(members)} members to {out.relative_to(PROJECT_ROOT)}" + (" (set as CURRENT)" if a.set_current else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
