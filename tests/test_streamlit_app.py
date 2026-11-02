"""The Streamlit host degrades gracefully: without the private model files it says so instead of crashing or showing a dead page."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="the Streamlit host is optional (see deploy/streamlit/requirements.txt)")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "deploy" / "streamlit" / "streamlit_app.py"
SHIM = APP.with_name("ui_shim.js")
INDEX = Path(__file__).resolve().parents[1] / "src" / "demo" / "static" / "index.html"


def test_without_model_files_the_page_explains_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PTIC_ASSETS_DIR", str(tmp_path))                 # an empty folder: nothing to load
    monkeypatch.delenv("HF_TOKEN", raising=False)
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, at.exception
    assert any("model files could not be loaded" in e.value and "HF_TOKEN" in e.value for e in at.error)
    assert "hf_" not in str(at.error)                                      # no token or path is shown to visitors


def test_the_shim_has_a_place_in_the_page_and_the_page_is_otherwise_untouched():
    html = INDEX.read_text()
    assert html.count('<script src="static/theme.js"></script>') == 1       # build_frontend() puts the shim right before this line
    assert "api/" in (INDEX.parent / "app.js").read_text()                 # the requests the shim answers


def test_the_entry_point_never_prints_or_renders_a_secret_value():
    source = APP.read_text()
    assert "st.write(st.secrets" not in source and "print(" not in source
    assert "console.log" not in SHIM.read_text()
