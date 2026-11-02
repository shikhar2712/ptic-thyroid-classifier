import json

import numpy as np
import pytest
from PIL import Image, ImageFilter

from src.common import PROJECT_ROOT
from src.inference.gate import GateParams, evaluate_checks, image_stats, read_image, rescale_to_model, resolve_scale

DEPLOY = PROJECT_ROOT / "artifacts" / "deploy"


def params():
    p = GateParams()
    p.thresholds = {"appearance_distance_max": 50.0, "saturation_min": 0.06, "he_hue_share_min": 0.8,
                    "white_point_rb_range": [-0.01, 0.02], "white_point_g_range": [-0.04, 0.01], "glass_noise_max": 6.0, "focus_ratio_min": 0.5}
    return p


def test_scale_is_required_and_low_magnification_is_refused():
    p = params()
    assert not evaluate_checks(p, None, None, None, None)["scale_known"]["passed"]
    assert not evaluate_checks(p, 1.04, None, None, None)["magnification_sufficient"]["passed"]      # 10x
    assert evaluate_checks(p, 0.52, None, None, None)["magnification_sufficient"]["passed"]          # 20x
    assert evaluate_checks(p, 0.26, None, None, None)["magnification_sufficient"]["passed"]          # 40x


GOOD = {"background_fraction": 0.1, "dark_fraction": 0.0, "saturation_mean": 0.15, "he_hue_share": 0.99, "white_point_rb": 0.002,
        "white_point_g": -0.015, "glass_noise": 3.5, "focus_ratio": 0.8}


def test_every_check_has_a_refusal_path():
    p = params()
    d = {"appearance_distance": 10.0}
    assert all(v["passed"] for v in evaluate_checks(p, 0.26, (512, 512), GOOD, d).values())
    cases = [("size_sufficient", dict(shape=(200, 512))), ("enough_tissue", dict(st={"background_fraction": 0.7})),
             ("no_large_dark_areas", dict(st={"dark_fraction": 0.2})), ("colour_present", dict(st={"saturation_mean": 0.0})),
             ("stain_looks_like_HE", dict(st={"he_hue_share": 0.3})), ("white_balance_neutral", dict(st={"white_point_rb": 0.07})),
             ("white_balance_neutral", dict(st={"white_point_g": 0.05})), ("noise_low", dict(st={"glass_noise": 9.0})),
             ("in_focus", dict(st={"focus_ratio": 0.3})), ("appearance_like_training", dict(d={"appearance_distance": 99.0}))]
    for name, kw in cases:
        c = evaluate_checks(p, 0.26, kw.get("shape", (512, 512)), {**GOOD, **kw.get("st", {})}, kw.get("d", d))
        assert not c[name]["passed"] and c[name]["reason"], name
        assert sum(not v["passed"] for v in c.values()) == 1, name


def test_unmeasurable_checks_pass_with_a_note():
    p = params()
    st = {**GOOD, "white_point_rb": float("nan"), "white_point_g": float("nan"), "glass_noise": float("nan"), "he_hue_share": float("nan")}
    c = evaluate_checks(p, 0.26, (512, 512), st, {"appearance_distance": 10.0})
    for name in ("white_balance_neutral", "noise_low", "stain_looks_like_HE"):
        assert c[name]["passed"] and c[name].get("note"), name


def test_read_image_handles_bad_files_alpha_and_grayscale(tmp_path):
    bad = tmp_path / "x.jpg"; bad.write_bytes(b"not an image")
    arr, info = read_image(bad)
    assert arr is None and "error" in info
    arr, _ = read_image(Image.new("RGBA", (300, 300), (200, 100, 150, 128)))
    assert arr.shape == (300, 300, 3) and arr.flags.writeable
    arr, info = read_image(Image.new("L", (300, 300), 128))
    assert info.get("grayscale_file") and arr.shape == (300, 300, 3)


def test_pixel_size_from_tiff_metadata(tmp_path):
    a = tmp_path / "aperio.tif"
    Image.new("RGB", (64, 64), (200, 150, 200)).save(a, tiffinfo={270: "Aperio Image Library v12 |AppMag = 40|MPP = 0.2500|"})
    assert abs(read_image(a)[1]["metadata_mpp"] - 0.25) < 1e-9
    b = tmp_path / "res.tif"
    Image.new("RGB", (64, 64)).save(b, dpi=(101600, 101600))           # 101,600 px per inch = 0.25 um/px
    assert abs(read_image(b)[1]["metadata_mpp"] - 0.25) < 1e-6
    c = tmp_path / "screen.tif"
    Image.new("RGB", (64, 64)).save(c, dpi=(96, 96))                   # screen dpi is not a microscope scale: ignored
    assert read_image(c)[1]["metadata_mpp"] is None


def test_scale_precedence_and_rescaling():
    info = {"metadata_mpp": 0.25}
    assert resolve_scale(info, 0.3, "20x") == (0.3, "declared_mpp")
    assert resolve_scale(info, None, "20x")[0] == 0.52
    assert resolve_scale(info, None, None) == (0.25, "file_metadata")
    assert resolve_scale({}, None, None)[0] is None
    assert resolve_scale({}, None, "63x")[0] is None
    z = np.zeros((100, 200, 3), np.uint8)
    assert rescale_to_model(z, 0.52).shape[:2] == (200, 400) and rescale_to_model(z, 0.26).shape == z.shape


def test_image_stats_background_dark_focus_and_white_point():
    rng = np.random.default_rng(0)
    tissue = rng.integers(60, 200, (256, 256, 3), dtype=np.uint8)
    glass = np.full((256, 256, 3), 240, np.uint8)
    assert image_stats(glass)["background_fraction"] == 1.0
    warm = np.concatenate([tissue, np.clip(glass * np.array([1.06, 1.0, 0.94]), 0, 255).astype(np.uint8)], axis=0)
    neutral = np.concatenate([tissue, glass], axis=0)
    assert image_stats(warm)["white_point_rb"] > 0.08 and abs(image_stats(neutral)["white_point_rb"]) < 0.01
    assert image_stats(np.full((256, 256, 3), 128, np.uint8))["saturation_mean"] == 0.0
    assert image_stats(np.zeros((256, 256, 3), np.uint8))["dark_fraction"] == 1.0
    blurred = np.asarray(Image.fromarray(tissue).filter(ImageFilter.GaussianBlur(3)))
    assert image_stats(blurred)["focus_ratio"] < image_stats(tissue)["focus_ratio"]


@pytest.mark.skipif(not ((DEPLOY / "gate_params.json").exists() and (DEPLOY / "model" / "CURRENT").exists()), reason="deploy artifacts not built")
def test_predictor_end_to_end():
    import pandas as pd
    from src.inference.predictor import PTICPredictor
    pred = PTICPredictor()
    folds = pd.read_csv(PROJECT_ROOT / "data/splits/folds_v1.csv")
    img = PROJECT_ROOT / "data/raw/images" / sorted(folds.loc[folds.role_fold0 == "val", "image_id"])[0]
    ok = pred.predict(img, magnification="40x")
    assert ok["status"] == "ok" and ok["answer"] in ("PTC-like", "not PTC-like", "uncertain") and 0 <= ok["probability_ptc_like"] <= 1
    assert ok["ensemble"]["members"] == len(pred.members) and ok["disclaimer"]
    json.dumps(ok)                                                         # stable, serialisable output
    k = int(folds.loc[folds.image_id == img.relative_to(PROJECT_ROOT / "data/raw/images").as_posix(), "test_fold"].iloc[0])
    fair = pred.predict(img, magnification="40x", member_folds={k})           # only the members that never saw this patient
    assert fair["ensemble"]["members"] == sum(f == k for f in pred.member_folds) and fair["ensemble"]["held_out_folds"] == [k]
    with pytest.raises(ValueError):
        pred.predict(img, magnification="40x", member_folds={99})
    assert pred.predict(img)["status"] == "refused"                        # scale unknown
    assert any("magnification" in r for r in pred.predict(img, magnification="10x")["refusal_reasons"])
    assert pred.predict(PROJECT_ROOT / "README.md", magnification="40x")["status"] == "refused"   # not an image
    assert pred.predict("/workspace/tutorials/_static/img/pruning.png", magnification="40x")["status"] == "refused"  # a diagram
