"""The web demo's API as plain functions (src/demo/rpc.py): the same paths and JSON as the FastAPI service, with no patient fields."""
import base64
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.demo import rpc  # noqa: E402


class FakePredictor:
    versions = {"model": "FAKE", "gate_params_sha256": "0" * 16}

    def __init__(self, status="ok"):
        self.status, self.calls = status, []

    def predict(self, image, mpp=None, magnification=None, member_folds=None):
        self.calls.append((mpp, magnification, member_folds))
        if self.status != "ok":
            return {"status": "refused", "answer": None, "probability_ptc_like": None, "checks": {}, "refusal_reasons": ["could not read image: boom"]}
        return {"status": "ok", "answer": "PTC-like", "probability_ptc_like": 0.99, "checks": {}, "refusal_reasons": [], "ensemble": {"members": 3}}


def png(w=64, h=48) -> bytes:
    b = io.BytesIO(); Image.fromarray(np.full((h, w, 3), 200, np.uint8)).save(b, "PNG"); return b.getvalue()


@pytest.fixture
def backend(tmp_path):
    import hashlib
    data = png()
    path = tmp_path / "ex.png"; path.write_bytes(data)
    ex = {"id": "ptic_1", "group": "PTIC image", "title": "PTIC, patient 7", "note": "n", "scale": {"magnification": "40x"}, "dataset_label": "PTC-like",
          "path": path, "thumb": b"\xff\xd8thumb", "preview": b"\xff\xd8preview"}
    known = {hashlib.sha256(data).hexdigest(): {"test_fold": 3, "dataset_label": "PTC-like"}}
    return rpc.Backend(FakePredictor(), lambda *a: {"scan_jpeg": "QQ==", "overlay_jpeg": "Qg=="}, known, {"ptic_1": ex})


def call(b, method, target, body=None):
    status, ctype, out = rpc.handle(b, method, target, body)
    return status, ctype, (json.loads(out) if ctype == rpc.JSON else out)


def test_health_and_examples_carry_no_patient_fields_or_titles(backend):
    s, _, h = call(backend, "GET", "api/health")
    assert s == 200 and h["model"] == "FAKE" and h["token_required"] is False and h["figures"] == 0 and h["examples"] == 1 and h["max_upload_mb"] == 25
    s, _, ex = call(backend, "GET", "api/examples")
    assert s == 200 and ex["examples"] == [{"id": "ptic_1", "group": "PTIC image", "note": "n", "scale": {"magnification": "40x"}, "dataset_label": "PTC-like"}]
    assert call(backend, "GET", "api/examples/ptic_1/thumb")[2] == b"\xff\xd8thumb" and call(backend, "GET", "api/examples/ptic_1/preview")[1] == "image/jpeg"
    assert call(backend, "GET", "api/examples/nope/thumb")[0] == 404 and call(backend, "GET", "api/figures/gradcam_random")[0] == 404


def test_example_prediction_is_scored_fairly_with_a_heat_map_and_no_patient_number(backend):
    s, _, r = call(backend, "POST", "api/examples/ptic_1/predict?explain=1")
    assert s == 200 and r["answer"] == "PTC-like" and r["heatmap"]["made"] is True and r["device"] == "cpu"
    assert backend.predictor.calls[-1] == (None, "40x", {3})                       # only the models that never saw its patient
    assert "title" not in r["example"] and not re.search(r"patient\s*#?\s*\d|patient_id|image_id", json.dumps({k: v for k, v in r.items() if k != "heatmap"}), re.I)
    s, _, r = call(backend, "POST", "api/examples/ptic_1/predict")                  # no explain: no heat map
    assert "heatmap" not in r


def test_upload_validation_matches_the_fastapi_service(backend):
    for target, want in (("api/predict", 422), ("api/predict?magnification=40x&mpp=0.25", 422), ("api/predict?magnification=7x", 422),
                         ("api/predict?mpp=-1", 422), ("api/predict?mpp=abc", 422), ("api/predict?magnification=40x", 422)):      # the last: empty body
        assert call(backend, "POST", target)[0] == want, target
    s, _, r = call(backend, "POST", "api/predict?magnification=20x&explain=1", png(65, 48))
    assert s == 200 and "known_ptic_image" not in r and backend.predictor.calls[-1] == (None, "20x", None)   # another image: all members
    s, _, r = call(backend, "POST", "api/predict?mpp=0.252", png(65, 48))
    assert s == 200 and backend.predictor.calls[-1] == (0.252, None, None)
    s, _, r = call(backend, "POST", "api/predict?magnification=40x", png())              # byte-identical to the sample
    assert r["known_ptic_image"] == {"dataset_label": "PTC-like", "test_fold": 3, "scored_by": "only the models that never saw this image's patient (cross-validation test fold 3)"}
    assert call(backend, "POST", "api/predict?magnification=40x", b"x" * (rpc.MAX_MB * 2**20 + 1))[0] == 413


def test_whole_slides_are_refused_and_unreadable_files_are_refused_with_a_reason(backend, monkeypatch):
    monkeypatch.setattr(rpc, "MAX_PIXELS", 1000)
    s, _, r = call(backend, "POST", "api/predict?magnification=40x", png(64, 48))
    assert s == 413 and "too many" in r["detail"] or "megapixels" in r["detail"]
    monkeypatch.setattr(rpc, "MAX_PIXELS", 40_000_000)
    backend.predictor = FakePredictor("refused")
    s, _, r = call(backend, "POST", "api/predict?magnification=40x&explain=1", b"not an image")
    assert s == 200 and r["status"] == "refused" and r["refusal_reasons"] == [rpc.UNREADABLE] and r["heatmap"]["made"] is False


def test_a_failing_heat_map_leaves_the_answer_and_a_server_error_is_a_clean_500(backend):
    def boom(*a):
        raise RuntimeError("x")
    backend.heatmap = boom
    s, _, r = call(backend, "POST", "api/examples/ptic_1/predict?explain=1")
    assert s == 200 and r["answer"] == "PTC-like" and r["heatmap"] == {"made": False, "reason": rpc.NO_MAP["failed"]}
    backend.predictor.predict = boom
    s, _, r = call(backend, "POST", "api/examples/ptic_1/predict")
    assert s == 500 and r == {"detail": "internal error while scoring the image"}


def test_batch_wire_format_round_trips_binary_bodies(backend):
    items = [{"id": 1, "method": "GET", "path": "api/health", "body": None},
             {"id": 2, "method": "POST", "path": "api/predict?magnification=40x", "body": base64.b64encode(png(65, 48)).decode()},
             {"id": 3, "method": "GET", "path": "api/examples/ptic_1/thumb", "body": None}]
    out = rpc.handle_batch(backend, items)
    assert [o["id"] for o in out] == [1, 2, 3] and [o["status"] for o in out] == [200, 200, 200]
    assert json.loads(base64.b64decode(out[1]["body"]))["answer"] == "PTC-like" and base64.b64decode(out[2]["body"]) == b"\xff\xd8thumb"
