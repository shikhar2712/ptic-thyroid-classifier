"""The Streamlit front end degrades gracefully: without the private model files it says so instead of crashing."""
from pathlib import Path

import pytest

pytest.importorskip("streamlit", reason="the Streamlit front end is optional (see deploy/streamlit/requirements.txt)")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "deploy" / "streamlit" / "streamlit_app.py"


def test_without_model_files_the_page_explains_what_is_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("PTIC_ASSETS_DIR", str(tmp_path))                 # an empty folder: nothing to load
    monkeypatch.delenv("HF_TOKEN", raising=False)
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, at.exception
    assert at.title[0].value == "Thyroid patch check"
    assert any("model files could not be loaded" in e.value and "HF_TOKEN" in e.value for e in at.error)
    assert not at.button and not at.radio                                  # no controls without a model
    assert "hf_" not in str(at.get("markdown")) + str(at.error)           # no token or path is shown to visitors


def test_the_entry_point_never_prints_or_renders_a_secret_value():
    source = APP.read_text()
    assert "st.write(st.secrets" not in source and "print(" not in source
