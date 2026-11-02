import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from src.common import PROJECT_ROOT
from src.explain.gradcam_classic import evidence, live_maps, overlay


def test_evidence_smooths_first_so_for_and_against_cancel_and_keeps_only_ptc_like():
    checker = np.indices((24, 24)).sum(axis=0) % 2 * 2.0 - 1.0          # alternating +1 / -1 cells
    assert evidence(checker)[6:-6, 6:-6].max() < 0.05                    # nearby evidence for and against cancels (away from the edge)
    block = np.zeros((24, 24))
    block[8:16, 8:16] = 10.0
    assert evidence(-block).max() == 0.0                                 # evidence against "PTC-like" is not drawn
    assert evidence(block)[12, 12] > evidence(block)[0, 0] >= 0.0


def test_overlay_colours_only_the_area_the_model_reads():
    arr = np.full((300, 520, 3), 200, np.uint8)                          # 1 x 2 squares of 256 px, plus unseen strips
    ev = np.zeros((16, 32))
    ev[:, 16:] = 100.0                                                   # right square: strong evidence, left: none
    out = overlay(arr, ev, 16, 50.0, plt.get_cmap("jet"))
    assert out.shape == (75, 130, 3)                                     # a quarter of the pixels
    assert np.allclose(out[70, 2], out[70, 2].mean())                    # bottom strip: plain grey
    assert out[10, 100, 0] > out[10, 100, 2]                             # strong evidence: red
    assert out[10, 10, 2] > out[10, 10, 0]                               # no evidence: blue


def test_live_heat_map_of_the_demo_matches_the_report_11_map():
    """The demo's live path (its own reading and rescaling, then the 3 fair models) reproduces the saved map of picture A1."""
    shown_file, maps_file = PROJECT_ROOT / "reports" / "11_gradcam_shown.json", PROJECT_ROOT / "reports" / "11_gradcam_shown_maps.npz"
    if not (shown_file.exists() and maps_file.exists()):
        pytest.skip("Report 11 maps not made")
    s = next(x for x in json.loads(shown_file.read_text())["images"] if x["id"] == "A1")
    img = PROJECT_ROOT / "data" / "raw" / "images" / s["image_id"]
    if not img.exists():
        pytest.skip("PTIC images not on this machine")
    from src.inference.predictor import PTICPredictor
    pred = PTICPredictor(device="cpu")
    _, ens, diffs, cell, n = live_maps(pred, img.read_bytes(), None, "40x", {s["test_fold"]})
    ref = np.load(maps_file)["A1"]
    assert n == 3 and cell == 16 and ens.shape == ref.shape
    assert np.abs(ens - ref).max() < 2e-3 * np.abs(ref).max()
    assert abs(float(np.mean(1 / (1 + np.exp(-diffs)))) - s["e3_score_fp32"]) < 1e-3
