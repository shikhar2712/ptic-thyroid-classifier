"""Web demo API: validation, limits, access code, static page, heat-map figures; one end-to-end call with the deployed model."""
import io
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("fastapi", reason="web demo packages live in .venv (see requirements.txt)")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import PROJECT_ROOT  # noqa: E402
from src.demo.app import NO_MAP, STATIC, Settings, create_app  # noqa: E402

DEPLOY = PROJECT_ROOT / "artifacts" / "deploy"


class FakePredictor:
    versions = {"model": "FAKE", "gate_params_sha256": "0" * 16}

    def __init__(self):
        self.calls = []

    def predict(self, image, mpp=None, magnification=None, member_folds=None):
        self.calls.append((image.read(), mpp, magnification))
        self.folds = member_folds
        return {"status": "ok", "answer": "uncertain", "probability_ptc_like": 0.5, "checks": {}, "refusal_reasons": [],
                "versions": self.versions, "disclaimer": "demo"}


def png(w=64, h=48) -> bytes:
    b = io.BytesIO(); Image.fromarray(np.full((h, w, 3), 200, np.uint8)).save(b, "PNG"); return b.getvalue()


def client(env=None, known=None, examples=None, figures=None):
    fake = FakePredictor()
    return TestClient(create_app(fake, Settings(env or {}), known_images=known or {}, examples=examples or {}, figures=figures or {})), fake


def test_page_assets_and_health():
    c, _ = client()
    with c:
        r = c.get("/")
        assert r.status_code == 200 and "Not a medical device" in r.text and "frame-ancestors 'none'" in r.headers["content-security-policy"]
        assert c.get("/static/app.js").status_code == 200 and c.get("/static/style.css").status_code == 200
        assert c.get("/static/app.py").status_code == 404
        f = c.get("/static/fonts/ibm-plex-sans-latin-400-normal.woff2")                   # the page's own typeface, from this server
        assert f.status_code == 200 and f.headers["content-type"] == "font/woff2" and f.content[:4] == b"wOF2"
        assert c.get("/static/fonts/OFL-IBM-Plex-Sans.txt").status_code == 404 and c.get("/static/fonts/..%2Fapp.js").status_code == 404
        assert c.get("/static/favicon.svg").headers["content-type"].startswith("image/svg+xml")
        assert c.get("/static/theme.js").status_code == 200                       # applies a saved light/dark choice before paint
        h = c.get("/api/health").json()
        assert h["model"] == "FAKE" and h["token_required"] is False and h["magnifications"]["40x"] == 0.26
        assert c.get("/docs").status_code == 404                           # no Swagger UI (it would load scripts from a CDN)
    for f in ("index.html", "app.js", "style.css"):                        # the page loads nothing from other hosts
        assert not re.search(r"https?://", (STATIC / f).read_text()), f


def test_scale_is_required_and_validated():
    c, fake = client()
    with c:
        for q in ("", "?magnification=40x&mpp=0.25", "?magnification=7x", "?mpp=-1", "?mpp=nan"):
            assert c.post("/api/predict" + q, content=png()).status_code == 422, q
        assert c.post("/api/predict?magnification=40x", content=b"").status_code == 422
        r = c.post("/api/predict?magnification=20", content=png())
        assert r.status_code == 200 and r.json()["request"]["magnification"] == "20x"
        assert c.post("/api/predict?mpp=0.252", content=png()).status_code == 200
    assert fake.calls[0][0] == png() and fake.calls[0][2] == "20x" and fake.calls[1][1] == 0.252
    assert len(fake.calls) == 2                                            # rejected requests never reach the model


def test_upload_and_pixel_limits():
    c, fake = client({"PTIC_DEMO_MAX_MB": "0.001"})                        # ~1 KB
    with c:
        assert c.post("/api/predict?magnification=40x", content=b"x" * 5000).status_code == 413
    c, fake = client({"PTIC_DEMO_MAX_MPIX": "0.001"})                      # 1,000 pixels
    with c:
        r = c.post("/api/predict?magnification=40x", content=png(64, 48))
        assert r.status_code == 413 and "whole slide" in r.json()["detail"]
    assert not fake.calls
    assert Image.MAX_IMAGE_PIXELS > 1_000_000                              # the process-wide Pillow limit is restored


def test_access_code():
    c, fake = client({"PTIC_DEMO_TOKEN": "s3cret"})
    with c:
        assert c.get("/api/health").json()["token_required"] is True
        assert c.post("/api/predict?magnification=40x", content=png()).status_code == 401
        assert c.post("/api/predict?magnification=40x", content=png(), headers={"Authorization": "Bearer nope"}).status_code == 401
        assert c.post("/api/predict?magnification=40x", content=png(), headers={"Authorization": "Bearer s3cret"}).status_code == 200
    assert len(fake.calls) == 1


class BusyGPUPredictor(FakePredictor):
    device = "cuda"

    def __init__(self, message):
        super().__init__(); self.message = message

    def predict(self, image, mpp=None, magnification=None, member_folds=None):
        raise RuntimeError(self.message)


def test_gpu_out_of_memory_falls_back_to_cpu_other_errors_do_not():
    cpu = FakePredictor()
    app = create_app(BusyGPUPredictor("CUDA error: CUBLAS_STATUS_ALLOC_FAILED"), Settings({}), cpu_fallback=lambda: cpu, known_images={}, examples={},
                     figures={})
    with TestClient(app) as c:
        r = c.post("/api/predict?magnification=40x", content=png())
        assert r.status_code == 200 and r.json()["device"] == "cpu (GPU busy)" and len(cpu.calls) == 1
    app = create_app(BusyGPUPredictor("shape mismatch"), Settings({}), cpu_fallback=lambda: cpu, known_images={}, examples={}, figures={})
    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.post("/api/predict?magnification=40x", content=png()).status_code == 500
    assert len(cpu.calls) == 1                                             # a real bug is not hidden by the fallback


def test_gpu_full_at_start_up_loads_on_cpu(monkeypatch):
    import src.demo.app as demo

    def busy(model, device=None):
        raise RuntimeError("CUDA error: out of memory")
    monkeypatch.setattr(demo, "load_predictor", busy)
    cpu = FakePredictor(); cpu.device = "cpu"
    with TestClient(demo.create_app(settings=Settings({}), cpu_fallback=lambda: cpu, known_images={}, examples={}, figures={})) as c:
        assert c.get("/api/health").json()["device"] == "cpu"
        assert c.post("/api/predict?magnification=40x", content=png()).json()["device"] == "cpu"
    with pytest.raises(RuntimeError):                                      # an explicitly requested GPU is not silently replaced
        with TestClient(demo.create_app(settings=Settings({"PTIC_DEMO_DEVICE": "cuda"}), cpu_fallback=lambda: cpu, known_images={}, examples={},
                                        figures={})):
            pass


def test_not_an_image_reaches_the_predictor_which_refuses_it():
    c, fake = client()
    with c:
        assert c.post("/api/predict?magnification=40x", content=b"hello").status_code == 200
    assert fake.calls[0][0] == b"hello"


@pytest.mark.skipif(not ((DEPLOY / "gate_params.json").exists() and (DEPLOY / "model" / "CURRENT").exists()), reason="deploy artifacts not built")
def test_end_to_end_with_deployed_model():
    import pandas as pd
    folds = pd.read_csv(PROJECT_ROOT / "data/splits/folds_v1.csv")
    img = PROJECT_ROOT / "data/raw/images" / sorted(folds.loc[folds.role_fold0 == "val", "image_id"])[0]
    with TestClient(create_app(settings=Settings({}))) as c:
        r = c.post("/api/predict?magnification=40x", content=img.read_bytes(), headers={"Content-Type": "image/jpeg"})
        assert r.status_code == 200
        res = r.json()
        assert res["status"] == "ok" and res["answer"] in ("PTC-like", "not PTC-like", "uncertain") and res["disclaimer"]
        assert res["versions"]["model"] == (DEPLOY / "model" / "CURRENT").read_text().strip()
        bad = c.post("/api/predict?magnification=40x", content=(PROJECT_ROOT / "README.md").read_bytes()).json()
        assert bad["status"] == "refused" and bad["refusal_reasons"] == ["not a readable image file (use JPEG, PNG or TIFF)"]
        low = c.post("/api/predict?magnification=10x", content=img.read_bytes()).json()
        assert low["status"] == "refused" and any("magnification" in x for x in low["refusal_reasons"])


def test_known_ptic_image_is_scored_only_by_models_that_never_saw_the_patient():
    import hashlib
    data = png()
    known = {hashlib.sha256(data).hexdigest(): {"image_id": "PTC-like/7_1 (3).jpg", "patient_id": 7, "test_fold": 3, "dataset_label": "PTC-like"}}
    c, fake = client(known=known)
    with c:
        r = c.post("/api/predict?magnification=40x", content=data).json()
        assert fake.folds == {3} and r["known_ptic_image"]["patient_id"] == 7 and "fold 3" in r["known_ptic_image"]["scored_by"]
        r = c.post("/api/predict?magnification=40x", content=png(65, 48)).json()          # any other image: all members
        assert fake.folds is None and "known_ptic_image" not in r


def test_public_mode_sends_no_patient_number_image_id_or_example_title(tmp_path):
    import hashlib
    data = png()
    known = {hashlib.sha256(data).hexdigest(): {"image_id": "PTC-like/7_1 (3).jpg", "patient_id": 7, "test_fold": 3, "dataset_label": "PTC-like"}}
    img = tmp_path / "ex.png"; img.write_bytes(data)
    ex = {"id": "ptic_1", "group": "PTIC image", "title": "PTIC, patient 7", "note": "n", "scale": {"magnification": "40x"},
          "dataset_label": "PTC-like", "path": img, "thumb": b"t", "preview": b"p"}
    c, fake = client({"PTIC_DEMO_PUBLIC": "1"}, known=known, examples={"ptic_1": ex})
    with c:
        r = c.post("/api/predict?magnification=40x", content=data).json()
        k = r["known_ptic_image"]
        assert fake.folds == {3} and k["dataset_label"] == "PTC-like" and k["test_fold"] == 3     # still scored fairly, and the page still gets its label
        assert "patient_id" not in k and "image_id" not in k and "7" not in k["scored_by"].replace("fold 3", "")
        listed = c.get("/api/examples").json()["examples"][0]
        assert "title" not in listed and listed["id"] == "ptic_1"
        assert "title" not in c.post("/api/examples/ptic_1/predict").json()["example"]
    c, _ = client({}, known=known, examples={"ptic_1": ex})                                          # default (internal) mode: unchanged
    with c:
        assert c.post("/api/predict?magnification=40x", content=data).json()["known_ptic_image"]["patient_id"] == 7
        assert c.get("/api/examples").json()["examples"][0]["title"] == "PTIC, patient 7"


def test_page_framing_is_refused_unless_the_operator_allows_a_host():
    c, _ = client({"PTIC_DEMO_FRAME_ANCESTORS": "https://huggingface.co https://*.hf.space"})
    with c:
        csp = c.get("/").headers["content-security-policy"]
        assert "frame-ancestors https://huggingface.co https://*.hf.space" in csp and "frame-ancestors 'none'" not in csp
        assert "default-src 'self'" in csp and "object-src 'none'" in csp                              # the rest of the policy is untouched
    for bad in ("'none'; script-src *", "x\ny"):
        with pytest.raises(ValueError):
            Settings({"PTIC_DEMO_FRAME_ANCESTORS": bad})


def test_examples_list_images_and_predict(tmp_path):
    img = tmp_path / "ex.png"; img.write_bytes(png())
    ex = {"id": "ptic_1", "group": "PTIC image", "title": "t", "note": "n", "scale": {"magnification": "40x"}, "dataset_label": "PTC-like",
          "path": img, "thumb": b"thumbjpeg", "preview": b"previewjpeg"}
    c, fake = client({"PTIC_DEMO_TOKEN": "s3cret"}, examples={"ptic_1": ex})
    with c:
        lst = c.get("/api/examples").json()["examples"]
        assert lst == [{k: ex[k] for k in ("id", "group", "title", "note", "scale", "dataset_label")}]
        assert c.get("/api/examples/ptic_1/thumb").content == b"thumbjpeg" and c.get("/api/examples/ptic_1/preview").status_code == 200
        assert c.get("/api/examples/ptic_1/file").status_code == 404 and c.get("/api/examples/nope/thumb").status_code == 404
        assert c.post("/api/examples/ptic_1/predict").status_code == 401                 # same access code as uploads
        r = c.post("/api/examples/ptic_1/predict", headers={"Authorization": "Bearer s3cret"}).json()
        assert r["example"]["id"] == "ptic_1" and r["request"]["magnification"] == "40x"
        assert c.post("/api/examples/nope/predict", headers={"Authorization": "Bearer s3cret"}).status_code == 404
    assert fake.calls[0][0] == png() and fake.calls[0][2] == "40x"


def test_built_example_pack_is_complete():
    from src.demo.app import EXAMPLES_FILE, load_examples
    if not EXAMPLES_FILE.exists():
        pytest.skip("demo pack not built")
    exs = load_examples()
    assert len(exs) == len(json.loads(EXAMPLES_FILE.read_text())["examples"])       # every referenced image exists
    assert {e["group"] for e in exs.values()} == {"PTIC image"}                    # since 2026-10-29 only images the model gets right
    assert {e["dataset_label"] for e in exs.values()} == {"PTC-like", "non-PTC-like"}   # both classes are on the page
    titles = [e["title"] for e in exs.values()]
    assert len(set(titles)) == len(titles) and "PTIC, patient 1" not in titles          # one image per patient; patient 1 is never a sample


def test_every_sample_image_is_answered_correctly():
    """The page shows only images the model gets right (src/demo/select_examples.py). Re-checks that with the deployed model, scored
    as the page scores a PTIC file (only the models that never saw its patient), and that none is a close call."""
    from src.demo.app import EXAMPLES_FILE, load_examples
    if not EXAMPLES_FILE.exists():
        pytest.skip("demo pack not built")
    exs = load_examples()
    with TestClient(create_app(settings=Settings({}))) as c:
        for ex_id, ex in exs.items():
            r = c.post(f"/api/examples/{ex_id}/predict").json()
            want = "PTC-like" if ex["dataset_label"] == "PTC-like" else "not PTC-like"
            assert r["status"] == "ok" and r["answer"] == want, (ex_id, r["status"], r["answer"])
            p = r["probability_ptc_like"]
            assert (p if want == "PTC-like" else 1 - p) >= 0.9, (ex_id, p)


def test_heat_map_figures(tmp_path):
    from src.demo.app import FIGURES, load_figures
    assert load_figures(tmp_path) == {}                                    # not made: the page keeps the section hidden
    for name in FIGURES.values():
        Image.fromarray(np.full((20, 30, 4), 255, np.uint8)).save(tmp_path / name)   # RGBA PNG, as matplotlib writes it
    figs = load_figures(tmp_path)
    assert set(figs) == set(FIGURES) and all(Image.open(io.BytesIO(b)).format == "JPEG" for b in figs.values())
    c, _ = client(figures=figs)
    with c:
        assert c.get("/api/health").json()["figures"] == 3
        for fid in FIGURES:
            r = c.get(f"/api/figures/{fid}")
            assert r.status_code == 200 and r.content == figs[fid] and r.headers["content-type"] == "image/jpeg"
        for bad in ("nope", "11_gradcam_random_examples.png", "11_gradcam_per_image.csv.gz", "..%2F11_gradcam.md"):
            assert c.get(f"/api/figures/{bad}").status_code == 404           # only these figures, never other report files
    page = (STATIC / "index.html").read_text()
    assert "not made for your upload" in page and "cannot rule out the main worry" in page   # the caveats sit next to the pictures


class RefusingPredictor(FakePredictor):
    def predict(self, image, mpp=None, magnification=None, member_folds=None):
        out = super().predict(image, mpp, magnification, member_folds)
        return {**out, "status": "refused", "answer": None, "refusal_reasons": ["colour cast"]}


def test_heat_map_of_a_scored_image_only_when_asked_and_only_for_answers():
    seen = []

    def explainer(pred, data, mpp, magnification, folds, result):
        seen.append((data, magnification, folds, result["probability_ptc_like"]))
        return {"scan_jpeg": "c2Nhbg==", "overlay_jpeg": "aGVhdA==", "width": 2, "height": 1}

    with TestClient(create_app(FakePredictor(), Settings({}), known_images={}, examples={}, figures={}, explainer=explainer)) as c:
        assert c.get("/api/health").json()["heatmaps"] is True
        assert "heatmap" not in c.post("/api/predict?magnification=40x", content=png()).json()      # not asked: as before
        r = c.post("/api/predict?magnification=40x&explain=1", content=png()).json()
        assert r["heatmap"] == {"made": True, "scan_jpeg": "c2Nhbg==", "overlay_jpeg": "aGVhdA==", "width": 2, "height": 1}
        assert seen == [(png(), "40x", None, 0.5)]                      # the same bytes, scale and models as the score
    with TestClient(create_app(RefusingPredictor(), Settings({}), known_images={}, examples={}, figures={}, explainer=explainer)) as c:
        r = c.post("/api/predict?magnification=40x&explain=1", content=png()).json()
        assert r["heatmap"] == {"made": False, "reason": NO_MAP["refused"]} and len(seen) == 1   # no map without an answer
    with TestClient(create_app(FakePredictor(), Settings({}), known_images={}, examples={}, figures={}, explainer=False)) as c:
        assert c.get("/api/health").json()["heatmaps"] is False
        assert c.post("/api/predict?magnification=40x&explain=1", content=png()).json()["heatmap"]["made"] is False


def test_a_failed_heat_map_keeps_the_answer_and_a_busy_gpu_moves_both_to_the_cpu():
    def broken(*a):
        raise ValueError("boom")

    with TestClient(create_app(FakePredictor(), Settings({}), known_images={}, examples={}, figures={}, explainer=broken)) as c:
        r = c.post("/api/predict?magnification=40x&explain=1", content=png()).json()
        assert r["status"] == "ok" and r["answer"] == "uncertain" and r["heatmap"] == {"made": False, "reason": NO_MAP["failed"]}
    gpu, cpu = FakePredictor(), FakePredictor()
    gpu.device, cpu.device = "cuda", "cpu"

    def oom_on_gpu(pred, *a):
        if pred is gpu:
            raise RuntimeError("CUDA error: out of memory")
        return {"scan_jpeg": "", "overlay_jpeg": "", "width": 1, "height": 1}

    app = create_app(gpu, Settings({}), cpu_fallback=lambda: cpu, known_images={}, examples={}, figures={}, explainer=oom_on_gpu)
    with TestClient(app) as c:
        r = c.post("/api/predict?magnification=40x&explain=1", content=png()).json()
        assert r["device"] == "cpu (GPU busy)" and r["heatmap"]["made"] is True and len(cpu.calls) == 1   # score and map on the CPU


def test_report_11_figures_match_the_page():
    from src.demo.app import FIGURES, FIGURES_DIR, load_figures
    figs = load_figures()
    if not figs:
        pytest.skip("Report 11 figures not made")
    page = (STATIC / "index.html").read_text()
    for fid, name in FIGURES.items():
        w, h = Image.open(io.BytesIO(figs[fid])).size
        assert (w, h) == Image.open(FIGURES_DIR / name).size
        assert f'data-src="api/figures/{fid}" width="{w}" height="{h}"' in page        # the page reserves the right shape


def test_link_puts_the_access_code_after_the_hash():
    env = {**os.environ, "PTIC_DEMO_HOST": "127.0.0.1", "PTIC_DEMO_PORT": "1234", "PTIC_DEMO_TOKEN": "a b&c#d"}  # never the real code
    out = subprocess.run([str(PROJECT_ROOT / "scripts" / "run_demo_server.sh"), "link"], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.splitlines()[0] == "http://127.0.0.1:1234/#code=a%20b%26c%23d"            # URL-quoted, as app.js decodes it
    page = (STATIC / "app.js").read_text()
    assert "location.hash" in page and "localStorage" in page and "restoreCode();" in page
