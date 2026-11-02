"""Demo predictor: input checks (gate) -> rescale to 0.26 um/px -> ensemble of the fold models -> three-way answer, as JSON.

Research demo only. It never forces a yes/no answer: inputs that fail a check are refused with reasons, and scores in the middle
band are reported as "uncertain". The middle band does NOT protect against confidently wrong answers on unusual tissue (see Report 6).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.common import PROJECT_ROOT
from src.data.dataset import AugConfig, GPUPreprocess
from src.inference.gate import (MODEL_MPP, GateParams, GenericFeatures, evaluate_checks, gate_distances, image_stats, read_image,
                                rescale_to_model, resolve_scale)
from src.inference.predict import predict_uint8
from src.models.factory import build_model

DISCLAIMER = ("Research demo, not a medical device and not for diagnosis. 'PTC-like' / 'not PTC-like' describe the appearance of a "
              "tissue patch, learned from 1,006 hand-picked, clear-cut patches of 24 patients scanned at one hospital (Leica GT450, 40x). "
              "Performance on images from other hospitals, scanners or stains is unknown.")
SCHEMA_VERSION = "1.0"


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


class PTICPredictor:
    def __init__(self, deploy_dir: Path | None = None, model_name: str | None = None, device: str | None = None) -> None:
        self.deploy_dir = Path(deploy_dir or PROJECT_ROOT / "artifacts" / "deploy")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.amp = torch.bfloat16 if (self.device.type == "cuda" and torch.cuda.is_bf16_supported()) else None
        name = model_name or (self.deploy_dir / "model" / "CURRENT").read_text().strip()
        mdir = self.deploy_dir / "model" / name
        self.manifest = json.loads((mdir / "manifest.json").read_text())
        self.members = []
        for m in self.manifest["members"]:
            path = mdir / m["file"]
            if _sha(path) != m["sha256"][:16]:
                raise RuntimeError(f"checksum mismatch for {path}")
            ck = torch.load(path, map_location="cpu", weights_only=False)
            net = build_model(ck["model_cfg"]); net.load_state_dict(ck["model"])
            net = net.to(self.device).to(memory_format=torch.channels_last).eval()
            d = ck["data_cfg"]
            pre = GPUPreprocess(ck["channel_mean"], ck["channel_std"], d["crop"], d["scale"], AugConfig(**ck["aug_cfg"])).to(self.device)
            self.members.append((net, pre, d["eval_tile"]))
        self.member_folds = [int(m.get("fold", -1)) for m in self.manifest["members"]]   # the CV fold each member held out
        self.gate = GateParams.load(self.deploy_dir / "gate_params.json")
        self.band = tuple(self.gate.meta.get("answer_band", [0.2, 0.8]))
        self.features = GenericFeatures(self.device)
        here = Path(__file__).resolve().parent
        self.versions = {"schema": SCHEMA_VERSION, "model": name, "model_manifest_sha256": _sha(mdir / "manifest.json"),
                         "gate_params_sha256": _sha(self.deploy_dir / "gate_params.json"),
                         "code_sha256": {f: _sha(here / f) for f in ("predictor.py", "gate.py", "predict.py")}}

    @torch.no_grad()
    def predict(self, image: Any, mpp: float | None = None, magnification: str | None = None,
                member_folds: set[int] | None = None) -> dict[str, Any]:
        """member_folds: score with only the members whose held-out fold is in this set (e.g. the 3 models that never saw a known
        PTIC image's patient). Default: all members."""
        use = [i for i, f in enumerate(self.member_folds) if member_folds is None or f in member_folds]
        if not use:
            raise ValueError(f"no ensemble member held out folds {sorted(member_folds)}")
        out: dict[str, Any] = {"status": None, "answer": None, "probability_ptc_like": None, "ensemble": None, "checks": {},
                               "refusal_reasons": [], "input": {}, "versions": self.versions, "disclaimer": DISCLAIMER}
        arr, info = read_image(image)
        out["input"] = {k: info.get(k) for k in ("format", "mode", "width", "height", "metadata_mpp", "grayscale_file")}
        if arr is None:
            out.update(status="refused", refusal_reasons=[f"could not read image: {info.get('error')}"])
            return out
        src_mpp, how = resolve_scale(info, mpp, magnification)
        out["input"].update(source_mpp=src_mpp, mpp_source=how)
        checks = evaluate_checks(self.gate, src_mpp, None, None, None)
        if not all(c["passed"] for c in checks.values()):
            out.update(status="refused", checks=checks, refusal_reasons=[c["reason"] for c in checks.values() if not c["passed"]])
            return out
        scaled = rescale_to_model(arr, src_mpp, MODEL_MPP)
        stats = image_stats(scaled, self.device)
        dists = gate_distances(self.gate, self.features(scaled))
        checks = evaluate_checks(self.gate, src_mpp, scaled.shape[:2], stats, dists)
        out["checks"] = checks
        out["input"].update(rescaled_shape=list(scaled.shape[:2]), model_mpp=MODEL_MPP)
        failed = [c["reason"] for c in checks.values() if not c["passed"]]
        if failed:
            out.update(status="refused", refusal_reasons=failed)
            return out
        x = torch.from_numpy(np.ascontiguousarray(scaled)).permute(2, 0, 1)[None].to(self.device)
        probs = []
        for i in use:
            net, pre, tile = self.members[i]
            probs.append(float(torch.softmax(predict_uint8(net, x, pre, tile, self.amp), 1)[0, 1]))
        p = float(np.mean(probs))
        answer = "PTC-like" if p >= self.band[1] else ("not PTC-like" if p <= self.band[0] else "uncertain")
        out.update(status="ok", answer=answer, probability_ptc_like=round(p, 4),
                   ensemble={"members": len(probs), "mean": round(p, 4), "std": round(float(np.std(probs)), 4),
                             "min": round(float(np.min(probs)), 4), "max": round(float(np.max(probs)), 4),
                             "held_out_folds": sorted({self.member_folds[i] for i in use})},
                   answer_band=list(self.band))
        return out
