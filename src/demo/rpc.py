"""The web demo's API as plain functions, with no web framework: `handle("POST", "api/predict?magnification=40x", body)` answers like
the FastAPI service in src/demo/app.py does (same paths, same JSON). The Streamlit front end (deploy/streamlit) runs the unchanged page
of src/demo/static inside a component and routes the page's requests here, so a free Streamlit host can show the same interface.

Public by design: no patient number, image id or example title is ever returned. Uploads are scored in memory and dropped.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from PIL import Image

log = logging.getLogger("ptic.rpc")
MAX_MB = 25
MAX_PIXELS = 40_000_000                                           # whole slides are refused; PTIC patches are about 2 megapixels
UNREADABLE = "not a readable image file (use JPEG, PNG or TIFF)"
NO_MAP = {"refused": "No heat map: the image was refused, so the model gave no answer.",
          "unavailable": "Heat maps are not available on this server.",
          "failed": "The heat map could not be made for this image."}
JSON = "application/json"
EXAMPLE_FIELDS = ("id", "group", "note", "scale", "dataset_label")    # no title: example titles can carry patient numbers


class HTTPError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


@dataclass
class Backend:
    predictor: Any
    heatmap: Callable[..., dict[str, Any]] | None                  # (predictor, data, mpp, magnification, folds, result) -> heat-map dict
    known: dict[str, dict[str, Any]]                               # sha256 of a PTIC sample file -> {"test_fold", "dataset_label"}; no patient number
    examples: dict[str, dict[str, Any]]                            # id -> record (path, thumb, preview bytes, public fields)
    device: str = "cpu"
    lock: threading.Lock = field(default_factory=threading.Lock)   # one analysis at a time: it bounds memory on a small host


def load_backend(base: Path, device: str = "cpu") -> Backend:
    """Everything the demo needs from the (downloaded) private files under `base`; raises when they are not complete."""
    import os
    os.environ.setdefault("TORCH_HOME", str(base / "artifacts" / "torch_cache"))
    from src.common import limit_cpu_threads
    from src.explain.gradcam_classic import classic_scale, live_heatmap
    from src.inference.predictor import PTICPredictor
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    limit_cpu_threads()
    predictor = PTICPredictor(deploy_dir=base / "artifacts" / "deploy", device=device)
    scale = classic_scale(base / "reports" / "11_gradcam_classic.json")
    heatmap = (lambda pred, data, mpp, mag, folds, result: live_heatmap(pred, data, mpp, mag, folds, result, scale)) if scale is not None else None
    known: dict[str, dict[str, Any]] = {}
    with (base / "data" / "metadata" / "manifest.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            known[row["sha256"]] = {"test_fold": int(row["test_fold"]), "dataset_label": "PTC-like" if int(row["label"]) == 1 else "non-PTC-like"}
    pack = base / "artifacts" / "demo_pack"
    examples = {}
    for ex in json.loads((pack / "examples.json").read_text())["examples"]:
        path = base / ex["file"]
        if path.exists():
            examples[ex["id"]] = {**ex, "path": path, "thumb": (pack / f"{ex['id']}_thumb.jpg").read_bytes(),
                                  "preview": (pack / f"{ex['id']}_preview.jpg").read_bytes()}
    return Backend(predictor, heatmap, known, examples, device)


def score(b: Backend, data: bytes, mpp: float | None, magnification: str | None, explain: bool = True) -> dict[str, Any]:
    """One analysis, shaped like the web demo's API answer."""
    t0 = time.time()
    try:
        with Image.open(io.BytesIO(data)) as im:                  # header only: refuse whole slides before decoding
            w, h = im.size
    except Image.DecompressionBombError:
        raise HTTPError(413, "image has too many pixels; send a patch or a crop, not a whole slide")
    except Exception:
        w = h = 0                                                 # not an image: the predictor refuses it with a reason
    if w * h > MAX_PIXELS:
        raise HTTPError(413, f"image is {w} x {h} px ({w * h / 1e6:.0f} megapixels, limit {MAX_PIXELS / 1e6:.0f}); send a patch or a crop, not a whole slide")
    known = b.known.get(hashlib.sha256(data).hexdigest())
    folds = {known["test_fold"]} if known else None               # a PTIC sample is scored only by the models that never saw its patient
    with b.lock:
        result = b.predictor.predict(io.BytesIO(data), mpp=mpp, magnification=magnification, member_folds=folds)
        result["device"] = b.device
        if explain:
            if result["status"] != "ok" or b.heatmap is None:
                result["heatmap"] = {"made": False, "reason": NO_MAP["refused" if result["status"] != "ok" else "unavailable"]}
            else:
                try:
                    hm = b.heatmap(b.predictor, data, mpp, magnification, folds, result)
                except Exception as exc:                          # the answer stands without its heat map
                    log.exception("heat map failed: %s", type(exc).__name__)
                    hm = None
                result["heatmap"] = {"made": True, **hm} if hm else {"made": False, "reason": NO_MAP["failed"]}
    result["refusal_reasons"] = [UNREADABLE if r.startswith("could not read image") else r for r in result.get("refusal_reasons", [])]
    if known:
        result["known_ptic_image"] = {"dataset_label": known["dataset_label"], "test_fold": known["test_fold"],
                                      "scored_by": f"only the models that never saw this image's patient (cross-validation test fold {known['test_fold']})"}
    result["request"] = {"magnification": magnification, "mpp": mpp, "seconds": round(time.time() - t0, 2)}
    return result


def _scale(q: dict[str, str]) -> tuple[float | None, str | None]:
    from src.inference.gate import MAGNIFICATION_TO_MPP
    mag, mpp = q.get("magnification"), q.get("mpp")
    if (mag is None) == (mpp is None):
        raise HTTPError(422, "give exactly one of magnification (for example 40x or 20x) or mpp (micrometres per pixel)")
    if mag is not None:
        mag = mag.lower().replace(" ", "")
        mag = mag if mag.endswith("x") else mag + "x"
        if mag not in MAGNIFICATION_TO_MPP:
            raise HTTPError(422, f"unknown magnification; use one of {', '.join(MAGNIFICATION_TO_MPP)} or give mpp")
        return None, mag
    try:
        value = float(mpp)
    except ValueError:
        raise HTTPError(422, "mpp must be a positive number of micrometres per pixel")
    if not (math.isfinite(value) and value > 0):
        raise HTTPError(422, "mpp must be a positive number of micrometres per pixel")
    return value, None


def health(b: Backend) -> dict[str, Any]:
    from src.inference.gate import MAGNIFICATION_TO_MPP
    v = b.predictor.versions
    return {"status": "ok", "model": v["model"], "gate_params_sha256": v["gate_params_sha256"], "device": b.device, "token_required": False,
            "max_upload_mb": float(MAX_MB), "max_megapixels": MAX_PIXELS / 1e6, "magnifications": MAGNIFICATION_TO_MPP,
            "known_ptic_images": len(b.known), "examples": len(b.examples), "figures": 0, "heatmaps": b.heatmap is not None}


def handle(b: Backend, method: str, target: str, body: bytes | None = None) -> tuple[int, str, bytes]:
    """(HTTP status, content type, body) for one request of the page. Errors are {"detail": ...} like the FastAPI service's."""
    try:
        url = urlsplit(target)
        parts = url.path.strip("/").split("/")
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        explain = q.get("explain", "").lower() in ("1", "true", "yes")
        if parts[:1] != ["api"]:
            raise HTTPError(404, "not found")
        if method == "GET" and parts == ["api", "health"]:
            return 200, JSON, json.dumps(health(b)).encode()
        if method == "GET" and parts == ["api", "examples"]:
            return 200, JSON, json.dumps({"examples": [{k: ex[k] for k in EXAMPLE_FIELDS} for ex in b.examples.values()]}).encode()
        if len(parts) == 4 and parts[:2] == ["api", "examples"]:
            ex = b.examples.get(parts[2])
            if ex is None:
                raise HTTPError(404, "not found")
            if method == "GET" and parts[3] in ("thumb", "preview"):
                return 200, "image/jpeg", ex[parts[3]]
            if method == "POST" and parts[3] == "predict":
                result = score(b, ex["path"].read_bytes(), ex["scale"].get("mpp"), ex["scale"].get("magnification"), explain)
                result["example"] = {k: ex[k] for k in EXAMPLE_FIELDS}
                return 200, JSON, json.dumps(result).encode()
        if method == "POST" and parts == ["api", "predict"]:
            mpp, mag = _scale(q)
            if not body:
                raise HTTPError(422, "empty request: send the image file's bytes as the request body")
            if len(body) > MAX_MB * 2**20:
                raise HTTPError(413, f"file larger than {MAX_MB} MB")
            return 200, JSON, json.dumps(score(b, body, mpp, mag, explain)).encode()
        raise HTTPError(404, "not found")
    except HTTPError as exc:
        return exc.status, JSON, json.dumps({"detail": exc.detail}).encode()
    except Exception as exc:                                       # logged without image content
        log.exception("request failed: %s", type(exc).__name__)
        return 500, JSON, json.dumps({"detail": "internal error while scoring the image"}).encode()


def handle_batch(b: Backend, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The component's wire format: each item {id, method, path, body (base64 or None)} -> {id, status, type, body (base64)}."""
    out = []
    for it in items:
        data = base64.b64decode(it["body"]) if it.get("body") else None
        status, ctype, body = handle(b, str(it.get("method", "GET")).upper(), str(it.get("path", "")), data)
        out.append({"id": it["id"], "status": status, "type": ctype, "body": base64.b64encode(body).decode("ascii")})
    return out
