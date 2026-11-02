"""Web demo: an upload page and a JSON API around PTICPredictor. Internal research demo, not a medical device.

Start/stop with scripts/run_demo_server.sh (binds 127.0.0.1:8765 by default; reach it through an SSH tunnel, see README).
Uploads are read into memory, scored and dropped: nothing is written to disk and no image content is logged.

  POST api/predict?magnification=40x   body = the image file's raw bytes (no multipart form, so nothing is spooled to disk)
  POST api/predict?mpp=0.25            the same with an exact pixel size in micrometres
       ...&explain=1                   also a heat map of the scored image ("heatmap": two base64 JPEGs, made in memory; answered
                                       images only), drawn like the Report 11 classic figures and on their colour scale
  GET  api/examples                    the built-in examples (artifacts/demo_pack, made by src/demo/build_examples.py)
  GET  api/examples/<id>/thumb|preview small / medium JPEG of an example
  POST api/examples/<id>/predict       score an example with its known pixel size
  GET  api/figures/<id>                one of the three Report 11 heat-map figures (made once, offline; none is made for uploads)
  GET  api/health

Fair answers on PTIC images: every PTIC image was in the training data of 12 of the 15 ensemble members. When an upload (or example)
is byte-for-byte a PTIC file, it is scored only by the 3 members that never saw its patient (its cross-validation test fold), and
the response says so under "known_ptic_image". (The input checks were calibrated on all PTIC images, so they stay in-sample.)

Settings (environment):
  PTIC_DEMO_TOKEN         if set, the predict endpoints need the header "Authorization: Bearer <token>"
  PTIC_DEMO_MAX_MB        upload limit in MB (default 40; PTIC JPEGs are ~14 MB because of an embedded colour profile)
  PTIC_DEMO_MAX_MPIX      largest accepted image in megapixels (default 40; whole slides are refused)
  PTIC_DEMO_CORS_ORIGINS  comma-separated browser origins allowed to call the API (default: none, same origin only)
  PTIC_DEMO_MODEL         model folder under artifacts/deploy/model (default: the one named in CURRENT)
  PTIC_DEMO_DEVICE        cuda or cpu (default: the GPU if present; falls back to the CPU per request when the GPU is full)
  PTIC_DEMO_PUBLIC        1 = public hosting: no patient number, image id or example title leaves the server (default: off)
  PTIC_DEMO_FRAME_ANCESTORS  who may embed the page in a frame, as a CSP source list (default: 'none'; the Hugging Face Space sets
                          "https://huggingface.co https://*.hf.space" so the Space page can show it)
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import os
import re
import secrets
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import HTMLResponse, Response  # noqa: E402
from PIL import Image  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from src.common import PROJECT_ROOT, limit_cpu_threads  # noqa: E402
from src.explain.gradcam_classic import classic_scale, live_heatmap  # noqa: E402
from src.inference.gate import MAGNIFICATION_TO_MPP  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
EXAMPLES_FILE = PROJECT_ROOT / "artifacts" / "demo_pack" / "examples.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
FIGURES = {"gradcam_random": "11_gradcam_classic_random_examples.png", "gradcam_patient1": "11_gradcam_classic_patient1.png",
           "gradcam_outside": "11_gradcam_classic_outside.png"}         # the only report files the page can show (classic style, Report 11 §8)
UNREADABLE = "not a readable image file (use JPEG, PNG or TIFF)"
NO_MAP = {"refused": "No heat map: the image was refused, so the model gave no answer.",
          "unavailable": "Heat maps are not available on this server.",
          "failed": "The heat map could not be made for this image."}
PAGE_CSP = "default-src 'self'; img-src 'self' blob:; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
GPU_FAILURE_MARKERS = ("CUDA", "CUBLAS", "cuDNN", "NVML", "out of memory")
log = logging.getLogger("uvicorn.error")


class Settings:
    def __init__(self, env: dict[str, str] | None = None) -> None:
        env = os.environ if env is None else env
        self.token = env.get("PTIC_DEMO_TOKEN") or None
        self.max_bytes = int(float(env.get("PTIC_DEMO_MAX_MB", 40)) * 2**20)
        self.max_pixels = int(float(env.get("PTIC_DEMO_MAX_MPIX", 40)) * 1e6)
        self.cors = [o.strip() for o in env.get("PTIC_DEMO_CORS_ORIGINS", "").split(",") if o.strip()]
        self.model = env.get("PTIC_DEMO_MODEL") or None
        self.device = env.get("PTIC_DEMO_DEVICE") or None      # cuda / cpu; default: the GPU if there is one
        self.public = env.get("PTIC_DEMO_PUBLIC", "").strip().lower() in ("1", "true", "yes")
        self.frame_ancestors = (env.get("PTIC_DEMO_FRAME_ANCESTORS") or "'none'").strip()
        if not re.fullmatch(r"[A-Za-z0-9'*:/._ -]+", self.frame_ancestors):          # no ';' or newline: it goes into a header
            raise ValueError("PTIC_DEMO_FRAME_ANCESTORS may hold only a CSP source list such as: https://huggingface.co https://*.hf.space")


def load_predictor(model: str | None, device: str | None = None):
    from src.inference.predictor import PTICPredictor
    return PTICPredictor(model_name=model, device=device)


def load_known_images(manifest: Path = PROJECT_ROOT / "data" / "metadata" / "manifest.csv") -> dict[str, dict[str, Any]]:
    """SHA256 of each PTIC file -> patient, dataset label and test fold. Empty when the dataset is not on this machine."""
    import pandas as pd
    try:
        m = pd.read_csv(manifest, usecols=["image_id", "patient_id", "label", "sha256", "test_fold"], keep_default_na=False, na_values=[])
    except (OSError, ValueError):
        return {}
    return {r.sha256: {"image_id": r.image_id, "patient_id": int(r.patient_id), "test_fold": int(r.test_fold),
                       "dataset_label": "PTC-like" if int(r.label) == 1 else "non-PTC-like"} for r in m.itertuples()}


def load_examples(path: Path = EXAMPLES_FILE) -> dict[str, dict[str, Any]]:
    """Example id -> record with the image path and the thumbnail/preview bytes. Empty when the pack was not built."""
    if not path.exists():
        return {}
    out = {}
    for ex in json.loads(path.read_text())["examples"]:
        rec = dict(ex, path=PROJECT_ROOT / ex["file"])
        if not rec["path"].exists():
            continue
        for view in ("thumb", "preview"):
            rec[view] = (path.parent / f"{ex['id']}_{view}.jpg").read_bytes()
        out[ex["id"]] = rec
    return out


def load_figures(folder: Path = FIGURES_DIR) -> dict[str, bytes]:
    """Figure id -> full-size JPEG of a Report 11 heat-map figure (a third of the PNG's size). Empty unless all of them exist."""
    if not all((folder / name).exists() for name in FIGURES.values()):
        return {}
    out = {}
    for fid, name in FIGURES.items():
        with Image.open(folder / name) as im:
            buf = io.BytesIO()
            im.convert("RGB").save(buf, "JPEG", quality=90, subsampling=0)   # no chroma subsampling: colour blocks stay sharp
            out[fid] = buf.getvalue()
    return out


def default_explainer() -> Any:
    """explain(predictor, data, mpp, magnification, folds, result) -> heat-map dict or None; None when the classic figures (and so
    their colour scale) were not made."""
    scale = classic_scale()
    if scale is None:
        return None
    return lambda pred, data, mpp, magnification, folds, result: live_heatmap(pred, data, mpp, magnification, folds, result, scale)


def create_app(predictor: Any = None, settings: Settings | None = None, cpu_fallback: Any = None,
               known_images: dict[str, dict[str, Any]] | None = None, examples: dict[str, dict[str, Any]] | None = None,
               figures: dict[str, bytes] | None = None, explainer: Any = None) -> FastAPI:
    """cpu_fallback: factory for a CPU predictor, used when the shared GPU has no free memory (other jobs on the MIG slice).
    explainer: makes the heat map of a scored image (default: default_explainer(); False = none)."""
    s = settings or Settings()
    gpu_lock = threading.Lock()                                   # one prediction at a time
    make_cpu_predictor = cpu_fallback or (lambda: load_predictor(s.model, "cpu"))
    known = load_known_images() if known_images is None else known_images
    exs = load_examples() if examples is None else examples
    figs = load_figures() if figures is None else figures
    explain_fn = default_explainer() if explainer is None else (explainer or None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        limit_cpu_threads()
        previous = Image.MAX_IMAGE_PIXELS
        Image.MAX_IMAGE_PIXELS = s.max_pixels                    # Pillow refuses decompression bombs beyond 2x this
        if predictor is not None:
            app.state.predictor = predictor
        else:
            try:
                app.state.predictor = load_predictor(s.model, s.device)
            except RuntimeError as exc:                              # GPU full at start-up: run on the CPU
                if s.device is not None or not any(m in str(exc) for m in GPU_FAILURE_MARKERS):
                    raise
                log.warning("GPU busy at start-up (%s); loading the model on the CPU", str(exc).splitlines()[0][:120])
                app.state.predictor = make_cpu_predictor()
        app.state.cpu_predictor = None
        log.info("PTIC demo ready: model %s on %s, token %s, %d known PTIC images, %d examples, %d heat-map figures",
                 app.state.predictor.versions["model"], getattr(app.state.predictor, "device", "?"), "required" if s.token else "not set",
                 len(known), len(exs), len(figs))
        try:
            yield
        finally:
            Image.MAX_IMAGE_PIXELS = previous

    app = FastAPI(title="PTIC research demo", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    page_csp = PAGE_CSP.replace("frame-ancestors 'none'", f"frame-ancestors {s.frame_ancestors}")
    example_fields = ("id", "group", "note", "scale", "dataset_label") if s.public else ("id", "group", "title", "note", "scale", "dataset_label")
    if s.cors:
        app.add_middleware(CORSMiddleware, allow_origins=s.cors, allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])

    @app.middleware("http")
    async def common_headers(request: Request, call_next):
        resp = await call_next(request)
        resp.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"})
        return resp

    assets = {name: (STATIC / name).read_bytes() for name in ("index.html", "app.js", "theme.js", "style.css", "favicon.svg")}
    fonts = {p.name: p.read_bytes() for p in (STATIC / "fonts").glob("*.woff2")}   # Geist (OFL), served from here: the CSP allows no other host

    @app.get("/", response_class=HTMLResponse)
    def index() -> Response:
        return Response(assets["index.html"], media_type="text/html; charset=utf-8", headers={"Content-Security-Policy": page_csp})

    @app.get("/static/{name}")
    def static(name: str) -> Response:
        types = {"app.js": "text/javascript", "theme.js": "text/javascript", "style.css": "text/css", "favicon.svg": "image/svg+xml"}
        if name not in types:
            raise HTTPException(404)
        return Response(assets[name], media_type=types[name])

    @app.get("/static/fonts/{name}")
    def font(name: str) -> Response:
        if name not in fonts:
            raise HTTPException(404)
        return Response(fonts[name], media_type="font/woff2")

    @app.get("/api/health")
    def health(request: Request) -> dict[str, Any]:
        v = request.app.state.predictor.versions
        return {"status": "ok", "model": v["model"], "gate_params_sha256": v["gate_params_sha256"],
                "device": str(getattr(request.app.state.predictor, "device", "?")), "token_required": s.token is not None,
                "max_upload_mb": round(s.max_bytes / 2**20, 1), "max_megapixels": s.max_pixels / 1e6,
                "magnifications": MAGNIFICATION_TO_MPP, "known_ptic_images": len(known), "examples": len(exs), "figures": len(figs),
                "heatmaps": explain_fn is not None}

    def check_token(request: Request) -> None:
        if s.token is None:
            return
        given = request.headers.get("authorization", "").encode()
        if not secrets.compare_digest(given, f"Bearer {s.token}".encode()):
            raise HTTPException(401, "missing or wrong access code")

    def scored(pred: Any, data: bytes, mpp: float | None, magnification: str | None, folds: set[int] | None, explain: bool,
               device: str) -> dict[str, Any]:
        """Score with this predictor and, when asked, make the heat map with the same predictor (so a GPU failure in either moves
        both to the CPU)."""
        result = pred.predict(io.BytesIO(data), mpp=mpp, magnification=magnification, member_folds=folds)
        result["device"] = device
        if explain:
            if result["status"] != "ok" or explain_fn is None:
                result["heatmap"] = {"made": False, "reason": NO_MAP["refused" if result["status"] != "ok" else "unavailable"]}
                return result
            try:
                hm = explain_fn(pred, data, mpp, magnification, folds, result)
            except RuntimeError as exc:
                if device != "cpu" and device != "cpu (GPU busy)" and any(m in str(exc) for m in GPU_FAILURE_MARKERS):
                    raise
                log.exception("heat map failed: %s", type(exc).__name__)
                hm = None
            except Exception as exc:                               # the answer stands without its heat map
                log.exception("heat map failed: %s", type(exc).__name__)
                hm = None
            result["heatmap"] = {"made": True, **hm} if hm else {"made": False, "reason": NO_MAP["failed"]}
        return result

    def run_locked(data: bytes, mpp: float | None, magnification: str | None, folds: set[int] | None,
                   explain: bool = False) -> dict[str, Any]:
        with gpu_lock:
            pred = app.state.predictor
            try:
                return scored(pred, data, mpp, magnification, folds, explain, str(getattr(pred, "device", "?")))
            except RuntimeError as exc:
                if str(getattr(pred, "device", "")) == "cpu" or not any(m in str(exc) for m in GPU_FAILURE_MARKERS):
                    raise
                log.warning("GPU busy (%s); scoring on the CPU instead", str(exc).splitlines()[0][:120])
            if app.state.cpu_predictor is None:
                app.state.cpu_predictor = make_cpu_predictor()
            return scored(app.state.cpu_predictor, data, mpp, magnification, folds, explain, "cpu (GPU busy)")

    async def score(data: bytes, mpp: float | None, magnification: str | None, explain: bool = False) -> dict[str, Any]:
        k = known.get(hashlib.sha256(data).hexdigest())
        folds = {k["test_fold"]} if k else None
        t0 = time.time()
        try:
            result = await run_in_threadpool(run_locked, data, mpp, magnification, folds, explain)
        except Exception as exc:                                   # logged without image content
            log.exception("prediction failed: %s", type(exc).__name__)
            raise HTTPException(500, "internal error while scoring the image")
        result["refusal_reasons"] = [UNREADABLE if r.startswith("could not read image") else r for r in result.get("refusal_reasons", [])]
        if k and s.public:                                         # no patient number or image id leaves a public server
            result["known_ptic_image"] = {"dataset_label": k["dataset_label"], "test_fold": k["test_fold"],
                                          "scored_by": f"only the models that never saw this image's patient (cross-validation test fold {k['test_fold']})"}
        elif k:
            result["known_ptic_image"] = {**k, "scored_by": f"only the models that never saw patient {k['patient_id']} "
                                                             f"(cross-validation test fold {k['test_fold']})"}
        result["request"] = {"magnification": magnification, "mpp": mpp, "seconds": round(time.time() - t0, 2)}
        log.info("prediction: status=%s answer=%s known=%s heatmap=%s %.2fs", result["status"], result["answer"], bool(k),
                 (result.get("heatmap") or {}).get("made"), time.time() - t0)
        return result

    @app.post("/api/predict")
    async def predict(request: Request, magnification: str | None = None, mpp: float | None = None,
                      explain: bool = False) -> dict[str, Any]:
        check_token(request)
        if (magnification is None) == (mpp is None):
            raise HTTPException(422, "give exactly one of magnification (for example 40x or 20x) or mpp (micrometres per pixel)")
        if magnification is not None:
            magnification = magnification.lower().replace(" ", "")
            magnification = magnification if magnification.endswith("x") else magnification + "x"
            if magnification not in MAGNIFICATION_TO_MPP:
                raise HTTPException(422, f"unknown magnification; use one of {', '.join(MAGNIFICATION_TO_MPP)} or give mpp")
        if mpp is not None and not (math.isfinite(mpp) and mpp > 0):
            raise HTTPException(422, "mpp must be a positive number of micrometres per pixel")
        limit = f"{s.max_bytes / 2**20:.0f} MB"
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > s.max_bytes:
            raise HTTPException(413, f"file larger than {limit}")
        buf = bytearray()
        async for chunk in request.stream():                       # in memory only
            buf += chunk
            if len(buf) > s.max_bytes:
                raise HTTPException(413, f"file larger than {limit}")
        if not buf:
            raise HTTPException(422, "empty request: send the image file's bytes as the request body")
        data = bytes(buf); del buf
        try:
            with Image.open(io.BytesIO(data)) as im:               # header only: refuse whole slides before decoding
                w, h = im.size
        except Image.DecompressionBombError:
            raise HTTPException(413, "image has too many pixels; send a patch or a crop, not a whole slide")
        except Exception:
            w = h = 0                                              # not an image: the predictor refuses it with a reason
        if w * h > s.max_pixels:
            raise HTTPException(413, f"image is {w} x {h} px ({w * h / 1e6:.0f} megapixels, limit {s.max_pixels / 1e6:.0f}); "
                                     "send a patch or a crop, not a whole slide")
        try:
            return await score(data, mpp, magnification, explain)
        finally:
            del data

    @app.get("/api/examples")
    def list_examples() -> dict[str, Any]:
        return {"examples": [{k: ex[k] for k in example_fields} for ex in exs.values()]}

    @app.get("/api/examples/{ex_id}/{view}")
    def example_image(ex_id: str, view: str) -> Response:
        if ex_id not in exs or view not in ("thumb", "preview"):
            raise HTTPException(404)
        return Response(exs[ex_id][view], media_type="image/jpeg")

    @app.post("/api/examples/{ex_id}/predict")
    async def predict_example(ex_id: str, request: Request, explain: bool = False) -> dict[str, Any]:
        check_token(request)
        ex = exs.get(ex_id)
        if ex is None:
            raise HTTPException(404)
        result = await score(ex["path"].read_bytes(), ex["scale"].get("mpp"), ex["scale"].get("magnification"), explain)
        result["example"] = {k: ex[k] for k in example_fields}
        return result

    @app.get("/api/figures/{fig_id}")
    def figure(fig_id: str) -> Response:
        if fig_id not in figs:
            raise HTTPException(404)
        return Response(figs[fig_id], media_type="image/jpeg")

    return app


app = create_app()
