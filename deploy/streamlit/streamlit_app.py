"""Streamlit front end of the PTIC research demo, for Streamlit Community Cloud. Research use only, not a medical device.

    streamlit run deploy/streamlit/streamlit_app.py

The page scores one H&E thyroid patch with the 15-model ensemble behind the 11 input checks and draws a Grad-CAM heat map. The model
files are not in this repository: they are downloaded at the first start from a private Hugging Face repository with a read token, given
as the secret HF_TOKEN (and, if it is not the default, PTIC_ASSETS_REPO). Uploads are read into memory, scored and dropped.
"""
from __future__ import annotations

import base64
import csv
import hashlib
import html
import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Thread counts are fixed before PyTorch is imported (it is imported only when the model loads). Shared hosts report many cores that this
# process may not use, so the default is small.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PTIC_TORCH_THREADS", "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from src.demo import answers as A  # noqa: E402
from src.demo.assets import ensure_assets  # noqa: E402

log = logging.getLogger("ptic.streamlit")
DEFAULT_ASSETS_REPO = "shikharr27/ptic-private-assets"
MAX_PIXELS = 40_000_000                                           # whole slides are refused; PTIC patches are about 2 megapixels
UNREADABLE = "not a readable image file (use JPEG, PNG or TIFF)"
NO_MAP = {"refused": "No heat map: the image was refused, so the model gave no answer.",
          "unavailable": "Heat maps are not available on this server.",
          "failed": "The heat map could not be made for this image."}
DOT = {"ptc": "#da1e28", "neg": "#24a148", "unc": "#f1c21b", "ref": "#8d8d8d"}

CSS = """<style>
.verdict { display: flex; align-items: flex-start; gap: .7rem; font-weight: 300; line-height: 1.18; margin: .1rem 0 .5rem; }
.verdict .dot { flex: none; width: .95rem; height: .95rem; border-radius: 50%; margin-top: .95rem; }
.verdict.long .dot { margin-top: .6rem; }
.conf { font-size: 1.15rem; margin: 0 0 .6rem; }
.conf small { color: #525252; }
.sample-cap { font-size: .8rem; line-height: 1.35; color: #525252; margin: -.3rem 0 .5rem; }
</style>"""


def setting(name: str, default: str | None = None) -> str | None:
    """A Streamlit secret, else an environment variable, else the default. Never logged."""
    try:
        value = st.secrets.get(name)
    except Exception:                                             # no secrets file (a local run without one)
        value = None
    return str(value) if value else os.environ.get(name, default)


def assets_dir() -> Path:
    """Where the private files live: next to the code when that folder can be written, else a temporary folder."""
    chosen = os.environ.get("PTIC_ASSETS_DIR")
    if chosen:
        return Path(chosen)
    return ROOT if os.access(ROOT, os.W_OK) else Path(tempfile.gettempdir()) / "ptic_assets"


class AssetsMissing(RuntimeError):
    pass


@dataclass
class Backend:
    predictor: Any
    heatmap: Callable[..., dict[str, Any]] | None
    known: dict[str, dict[str, Any]]                               # sha256 of a PTIC sample file -> {"test_fold", "dataset_label"}; no patient number
    examples: list[dict[str, Any]]
    lock: threading.Lock                                           # one analysis at a time: like the web demo, and it bounds memory


@st.cache_resource(show_spinner="Loading the model (the first visit after a restart takes about a minute)…")
def load_backend() -> Backend:
    Image.MAX_IMAGE_PIXELS = MAX_PIXELS
    base = assets_dir()
    gone = ensure_assets(base, setting("PTIC_ASSETS_REPO", DEFAULT_ASSETS_REPO), setting("HF_TOKEN"))
    if gone:
        log.error("model files missing: %s", ", ".join(gone))
        raise AssetsMissing("The model files could not be loaded.")
    os.environ.setdefault("TORCH_HOME", str(base / "artifacts" / "torch_cache"))
    from src.common import limit_cpu_threads
    from src.explain.gradcam_classic import classic_scale, live_heatmap
    from src.inference.predictor import PTICPredictor
    limit_cpu_threads()
    predictor = PTICPredictor(deploy_dir=base / "artifacts" / "deploy", device="cpu")
    scale = classic_scale(base / "reports" / "11_gradcam_classic.json")
    heatmap = (lambda pred, data, mpp, mag, folds, result: live_heatmap(pred, data, mpp, mag, folds, result, scale)) if scale is not None else None
    known = {}
    with (base / "data" / "metadata" / "manifest.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            known[row["sha256"]] = {"test_fold": int(row["test_fold"]), "dataset_label": "PTC-like" if int(row["label"]) == 1 else "non-PTC-like"}
    pack = base / "artifacts" / "demo_pack"
    examples = []
    for ex in json.loads((pack / "examples.json").read_text())["examples"]:
        path = base / ex["file"]
        if path.exists():
            examples.append({**ex, "path": path, "thumb": (pack / f"{ex['id']}_thumb.jpg").read_bytes(), "preview": (pack / f"{ex['id']}_preview.jpg").read_bytes()})
    return Backend(predictor, heatmap, known, examples, threading.Lock())


def score(b: Backend, data: bytes, mpp: float | None, magnification: str | None) -> dict[str, Any]:
    """One analysis, shaped like the web demo's API answer (patient numbers and image ids never leave this function)."""
    t0 = time.time()
    try:
        with Image.open(io.BytesIO(data)) as im:                  # header only: refuse whole slides before decoding
            w, h = im.size
    except Image.DecompressionBombError:
        return {"status": "rejected", "message": "The image has too many pixels. Send a patch or a crop, not a whole slide."}
    except Exception:
        w = h = 0                                                 # not an image: the predictor refuses it with a reason
    if w * h > MAX_PIXELS:
        return {"status": "rejected", "message": f"The image is {w} x {h} px ({w * h / 1e6:.0f} megapixels, limit {MAX_PIXELS / 1e6:.0f}). "
                                                  "Send a patch or a crop, not a whole slide."}
    known = b.known.get(hashlib.sha256(data).hexdigest())
    folds = {known["test_fold"]} if known else None               # a PTIC sample is scored only by the models that never saw its patient
    with b.lock:
        result = b.predictor.predict(io.BytesIO(data), mpp=mpp, magnification=magnification, member_folds=folds)
        if result["status"] != "ok" or b.heatmap is None:
            result["heatmap"] = {"made": False, "reason": NO_MAP["refused" if result["status"] != "ok" else "unavailable"]}
        else:
            try:
                hm = b.heatmap(b.predictor, data, mpp, magnification, folds, result)
            except Exception as exc:                              # the answer stands without its heat map
                log.exception("heat map failed: %s", type(exc).__name__)
                hm = None
            result["heatmap"] = {"made": True, **hm} if hm else {"made": False, "reason": NO_MAP["failed"]}
    result["refusal_reasons"] = [UNREADABLE if r.startswith("could not read image") else r for r in result.get("refusal_reasons", [])]
    if known:
        result["known_ptic_image"] = {"dataset_label": known["dataset_label"]}
    result["seconds"] = round(time.time() - t0, 1)
    return result


@st.cache_data(max_entries=4, show_spinner=False)
def preview(data: bytes) -> bytes | None:
    """A screen-sized JPEG of an uploaded file, or None when it cannot be shown (it can still be analyzed)."""
    try:
        with Image.open(io.BytesIO(data)) as im:
            im.draft("RGB", (1400, 1400))
            rgb = im.convert("RGB")
        rgb.thumbnail((1200, 1200))
        buf = io.BytesIO()
        rgb.save(buf, "JPEG", quality=88)
        return buf.getvalue()
    except Exception:
        return None


def verdict_html(text: str, kind: str) -> str:
    return f'<div class="verdict{" long" if len(text) > 14 else ""}" style="font-size:{"2.5rem" if len(text) <= 14 else "1.9rem"}">' \
           f'<span class="dot" style="background:{DOT[kind]}"></span><span>{html.escape(text)}</span></div>'


def render_topics(res: dict[str, Any]) -> None:
    items = A.topics(res)
    if not items:
        return
    st.markdown("#### About this result")
    for _, question, blocks in items:
        with st.expander(question):
            for block in blocks:
                if isinstance(block, str):
                    st.write(block)
                else:
                    st.markdown("\n".join(f"- {x}" for x in block))


def render_result(res: dict[str, Any]) -> None:
    if res.get("status") == "rejected":
        st.markdown(verdict_html("Not analyzed", "ref"), unsafe_allow_html=True)
        st.warning(res["message"])
        return
    text, kind = A.headline(res)
    st.markdown(verdict_html(text, kind), unsafe_allow_html=True)
    if A.has_confidence(res):
        st.markdown(f'<p class="conf"><strong>{A.confidence(res)}</strong> <small>confidence</small></p>', unsafe_allow_html=True)
    if res["status"] != "ok":
        st.error("\n".join(f"- {r}" for r in res.get("refusal_reasons", [])) or "- the image could not be read")
    hm = res.get("heatmap") or {}
    if res["status"] == "ok":
        if hm.get("made"):
            a, b = st.columns(2)
            a.image(base64.b64decode(hm["scan_jpeg"]), caption="Selected image", width="stretch")
            b.image(base64.b64decode(hm["overlay_jpeg"]), caption="Grad-CAM", width="stretch")
            st.caption(A.GRADCAM_KEY)
        else:
            st.info(hm.get("reason") or "No heat map for this image.")
    render_topics(res)
    st.caption(f"Analyzed in {res.get('seconds', 0):.0f} s on shared CPU hardware.")


ABOUT = """
This research tool describes whether one H&E image of thyroid tissue *looks* like papillary thyroid carcinoma (PTC). It learned from
1,006 clear-cut patches of 24 patients, scanned at one hospital on one scanner (Leica GT450, 40×). **Not a medical device and not for
diagnosis:** it describes how a patch looks, not a patient's diagnosis, and it has not been validated on other hospitals, scanners or
stains. Such images are refused on purpose: in a test on 400 regions from 40 TCGA and GTEx slides, the input checks refused 389.

**Quality checks.** Before scoring, an image must pass 11 checks (pixel size known, magnification, size, tissue, dark areas, colour,
H&E colours, white balance, noise, focus, and a look similar to the training scans). Failing images are refused with the reasons.

**Fair scoring of the samples.** The six sample images come from the PTIC dataset and are a display selection of images the model
answers correctly: they show what the page does when it works, not how accurate it is. Every PTIC image was in the training data of 12
of the 15 models, so a sample is scored only by the 3 models that never saw its patient. Any other image, including a re-saved copy of
a sample, is scored by all 15.

**Heat maps.** Grad-CAM colours each part of an image by how much it pushed the model towards PTC: blue means none, yellow to red means
strong. It shows *where*, not *why*, and it cannot rule out the main worry: if the model learned the colour of PTIC's stain or scanner,
the map would still "look at the cells".

**Credits.** Sample images and training data: PTIC, Shabrina et al., Zenodo 10.5281/zenodo.18827019, CC BY-SA 4.0.
"""


def main() -> None:
    st.set_page_config(page_title="Thyroid patch check", page_icon=":material/biotech:", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    st.title("Thyroid patch check")
    st.caption("For research use only. Not a medical device and not for diagnosis. Images are checked in memory and never stored.")
    try:
        backend = load_backend()
    except AssetsMissing as exc:
        st.error(f"{exc} The app owner needs to add the secret HF_TOKEN in the app's settings.")
        st.stop()
    ss = st.session_state
    left, right = st.columns([5, 4], gap="large")

    with left:
        st.subheader("Image")
        mode = st.radio("Image source", ["Sample image", "Upload an image"], horizontal=True, label_visibility="collapsed")
        data: bytes | None = None
        magnification: str | None = None
        mpp: float | None = None
        key: str | None = None
        shown: bytes | None = None
        if mode == "Sample image":
            cols = st.columns(len(backend.examples))
            for i, (col, ex) in enumerate(zip(cols, backend.examples), 1):
                col.image(ex["thumb"], width="stretch")
                col.markdown(f'<p class="sample-cap"><strong>Sample image&nbsp;{i}</strong><br>PTIC · Label: {A.dataset_label(ex)}</p>', unsafe_allow_html=True)
                col.button(f"Select {i}", key=f"pick-{ex['id']}", width="stretch", on_click=ss.__setitem__, args=("sample", ex["id"]),
                           type="primary" if ss.get("sample") == ex["id"] else "secondary")
            chosen = next((e for e in backend.examples if e["id"] == ss.get("sample")), None)
            if chosen:
                data, magnification, key, shown = chosen["path"].read_bytes(), chosen["scale"].get("magnification"), f"sample:{chosen['id']}", chosen["preview"]
        else:
            up = st.file_uploader("Drop an H&E thyroid image here (JPEG, PNG or TIFF)", type=["jpg", "jpeg", "png", "tif", "tiff"])
            if up is not None:
                data = up.getvalue()
                shown = preview(data)
                scan = st.radio("Scanned at", ["40×", "20×", "Exact size"], index=None, horizontal=True,
                                help="A wrong magnification gives wrong answers.")
                if scan == "Exact size":
                    mpp = st.number_input("Micrometres per pixel", min_value=0.05, max_value=5.0, value=None, step=0.001, format="%.3f", placeholder="e.g. 0.252")
                elif scan:
                    magnification = scan.replace("×", "x")
                scale_key = magnification or (f"mpp:{mpp}" if mpp else None)
                key = f"upload:{hashlib.sha256(data).hexdigest()[:16]}:{scale_key}" if scale_key else None
                if shown is None:
                    st.caption("No preview for this file type. It can still be analyzed.")
        ready = data is not None and key is not None
        clicked = st.button("Analyze the image", type="primary", disabled=not ready)
        if not ready:
            st.caption("Upload an image and say how it was scanned." if mode == "Upload an image" else "Pick a sample image.")
        if shown is not None:
            st.image(shown, width="stretch")

    with right:
        st.subheader("Result")
        if clicked and ready:
            with st.spinner("Checking the image, scoring it and drawing the heat map…"):
                ss["result"] = {"key": key, "res": score(backend, data, mpp, magnification)}
        current = ss.get("result")
        if current and current["key"] == key:
            render_result(current["res"])
        else:
            st.write("No image analyzed yet." if not ready else "The image is ready. Press Analyze the image.")

    with st.expander("About this demo"):
        st.markdown(ABOUT)


main()
