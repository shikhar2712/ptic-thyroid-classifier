"""Streamlit host for the PTIC research demo, for Streamlit Community Cloud. Research use only, not a medical device.

    streamlit run deploy/streamlit/streamlit_app.py

This is not a second interface: it shows the demo's own page (src/demo/static, unchanged) in a Streamlit component. The page's requests
for "api/..." are answered by Python (src/demo/rpc.py) through deploy/streamlit/ui_shim.js, so the page looks and behaves as it does
when served by the FastAPI app.

The model files are not in this repository: they are downloaded at the first start from a private Hugging Face repository with a read
token, given as the secret HF_TOKEN (and, if it is not the default, PTIC_ASSETS_REPO). Uploads are read into memory, scored and dropped.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path

# Thread counts are fixed before PyTorch is imported (it is imported only when the model loads). Shared hosts report many cores that this
# process may not use, so the default is small.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("PTIC_TORCH_THREADS", "2")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402
import streamlit.components.v1 as components  # noqa: E402

from src.demo import rpc  # noqa: E402
from src.demo.assets import ensure_assets  # noqa: E402

log = logging.getLogger("ptic.streamlit")
DEFAULT_ASSETS_REPO = "shikharr27/ptic-private-assets"
STATIC = ROOT / "src" / "demo" / "static"
SHIM = Path(__file__).with_name("ui_shim.js")

# The page fills the window: Streamlit's own header, padding and "running" indicators are hidden, and the frame is as tall as the window.
CSS = """<style>
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"], footer { display: none !important; }
.stMainBlockContainer, [data-testid="stMainBlockContainer"], .block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stVerticalBlock"] { gap: 0 !important; }
iframe { display: block; width: 100%; border: 0; height: 100vh !important; }
[data-stale="true"], [data-testid="stElementContainer"] { opacity: 1 !important; }
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


@st.cache_resource(show_spinner="Loading the model (the first visit after a restart takes about a minute)…")
def load_backend() -> rpc.Backend:
    base = assets_dir()
    gone = ensure_assets(base, setting("PTIC_ASSETS_REPO", DEFAULT_ASSETS_REPO), setting("HF_TOKEN"))
    if gone:
        log.error("model files missing: %s", ", ".join(gone))
        raise AssetsMissing("The model files could not be loaded.")
    return rpc.load_backend(base)


@st.cache_resource(show_spinner=False)
def build_frontend() -> str:
    """The component's folder: the demo's page and assets as they are, plus the shim, loaded before the page's own script."""
    out = Path(tempfile.gettempdir()) / "ptic_ui_frontend"
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(STATIC, out / "static")
    shutil.copy(SHIM, out / "static" / "streamlit_shim.js")
    html = (STATIC / "index.html").read_text()
    marker = '<script src="static/theme.js"></script>'
    if marker not in html:
        raise RuntimeError("index.html changed: the shim has no place to go")
    (out / "index.html").write_text(html.replace(marker, '<script src="static/streamlit_shim.js"></script>\n' + marker, 1))
    return str(out)


def main() -> None:
    st.set_page_config(page_title="Thyroid patch check", page_icon=":material/biotech:", layout="wide", initial_sidebar_state="collapsed")
    st.markdown(CSS, unsafe_allow_html=True)
    try:
        backend = load_backend()
    except AssetsMissing as exc:
        st.error(f"{exc} The app owner needs to add the secret HF_TOKEN in the app's settings.")
        st.stop()
    ui = components.declare_component("ptic_ui", path=build_frontend())
    ss = st.session_state
    req = ui(reply=ss.get("reply"), key="ptic-ui", default=None)
    if req and req.get("batch") != ss.get("done"):                 # a new batch of the page's requests: answer it, then show the answers
        try:
            items = rpc.handle_batch(backend, req["items"])
        except Exception as exc:                                  # logged without image content
            log.exception("batch failed: %s", type(exc).__name__)
            body = base64.b64encode(json.dumps({"detail": "internal error while scoring the image"}).encode()).decode("ascii")
            items = [{"id": it.get("id"), "status": 500, "type": rpc.JSON, "body": body} for it in req.get("items", [])]
        ss["reply"] = {"batch": req["batch"], "items": items}
        ss["done"] = req["batch"]
        st.rerun()


main()
