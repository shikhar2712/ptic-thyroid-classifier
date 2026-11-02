"""Report 11 heat maps redrawn in the classic Grad-CAM style (user request, 2026-10-27): a smooth jet-coloured map of the evidence
for "PTC-like" over a black-and-white copy of the tissue. Display only: it reads the maps that gradcam.py saved
(reports/11_gradcam_shown_maps.npz) and runs no model.

Per image: the saved ensemble map (one value per 16-px cell: the local logit difference, positive = towards "PTC-like") is smoothed
first, so nearby evidence for and against cancels as it does in the model's own average; then only its positive part is kept and
divided by one scale shared by all 29 pictures (the 99th percentile), so colours can be compared between pictures. Blue = no
"PTC-like" evidence, red = strong.

  .venv/bin/python -I src/explain/gradcam_classic.py                    # reports/figures/11_gradcam_classic_*.png, reports/11_gradcam_classic.json
  .venv/bin/python -I src/explain/gradcam_classic.py --only A1,B1,C7 --out-dir <scratch>     # a quick look at a few pictures

live_heatmap() draws the same kind of map for an image the web demo has just scored (in memory, on the figures' colour scale).
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import math
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

from src.common import PROJECT_ROOT, load_config, resolve, write_json  # noqa: E402
from src.inference.gate import MODEL_MPP, rescale_to_model  # noqa: E402

REPORTS = PROJECT_ROOT / "reports"
SIGMA_CELLS = 3.0                 # display smoothing: 3 cells of 16 px = 12.5 um at the model's 0.26 um per pixel
ALPHA = 0.5                       # heat-map share of the blend (the usual 50/50 Grad-CAM overlay)
DOWN = 4                          # the figures show every image at a quarter of its pixels
UNSEEN = 0.55                     # grey level of the strips the model never reads
LIVE_MAX_WIDTH = 1280             # web demo heat maps: at most this many pixels wide
MAX_SCORE_GAP = 0.02              # the map's own score (fp32) vs the demo's score (bf16 on the GPU); a larger gap = no map
log = logging.getLogger("uvicorn.error")
FIGS = {"A": ("random_examples", "Random examples: where the model (E3) finds evidence for “PTC-like”",
              "E3 scores of these images are descriptive only (E3 had no final exam). Images chosen at random before any map was made."),
        "B": ("patient1", "Patient 1: the images E1 missed (top two rows) and got right (bottom two rows)",
              "Same images as rows 1-2 of the Report 6 error gallery. E3 scores are descriptive only."),
        "C": ("outside", "Outside hospitals (Report 8): where the model finds evidence for “PTC-like” on other hospitals' scans",
              "Model alone with the safety checks off (information only); the checks refused these regions except where marked accepted.")}


def evidence(cam: np.ndarray, sigma: float = SIGMA_CELLS) -> np.ndarray:
    """Signed cell map -> smoothed "PTC-like" evidence (>= 0). Smoothing comes first, so for and against cancel locally."""
    return np.maximum(gaussian_filter(cam.astype(np.float64), sigma, mode="nearest"), 0.0)


def overlay(arr: np.ndarray, ev: np.ndarray, cell: int, scale: float, cmap, down: int = DOWN) -> np.ndarray:
    """uint8 HxWx3 scan + evidence map -> float RGB at 1/down size: the jet map blended over a grey copy of the scan; the strips the
    model never reads stay plain grey, dimmed."""
    small = np.asarray(Image.fromarray(arr).resize((arr.shape[1] // down, arr.shape[0] // down), Image.BOX), np.float32) / 255
    grey = small @ np.array([0.299, 0.587, 0.114], np.float32)
    out = np.repeat((0.45 * grey + 0.55 * UNSEEN)[..., None], 3, axis=2)
    h, w = ev.shape[0] * cell // down, ev.shape[1] * cell // down
    mask = np.asarray(Image.fromarray(np.clip(ev / scale, 0, 1).astype(np.float32), "F").resize((w, h), Image.BILINEAR))
    heat = cmap(np.clip(mask, 0, 1))[..., :3].astype(np.float32)
    out[:h, :w] = (1 - ALPHA) * grey[:h, :w, None] + ALPHA * heat
    return out


def scan_view(arr: np.ndarray, shape: tuple[int, int], cell: int, down: int = DOWN) -> np.ndarray:
    """uint8 HxWx3 scan -> uint8 RGB at 1/down size, with the strips outside the map (never read by the model) dimmed."""
    h, w = shape[0] * cell, shape[1] * cell
    img = arr.astype(np.float32) / 255
    img[h:], img[:, w:] = 0.45 * img[h:] + 0.55 * UNSEEN, 0.45 * img[:, w:] + 0.55 * UNSEEN
    return np.asarray(Image.fromarray((img * 255).astype(np.uint8)).resize((arr.shape[1] // down, arr.shape[0] // down), Image.BOX))


def classic_scale(path: Path = REPORTS / "11_gradcam_classic.json") -> float | None:
    """The fixed colour scale of the classic figures, or None when they were not made (then the demo makes no heat maps)."""
    try:
        return float(json.loads(path.read_text())["colour_scale_99th_percentile"])
    except (OSError, KeyError, ValueError):
        return None


def jpeg_b64(rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def live_maps(pred, data: bytes, mpp: float | None, magnification: str | None,
              folds: set[int] | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """The predictor's own preprocessing (read, pixel size, rescale), then gradcam.image_maps with the members it scores with ->
    (rescaled image, ensemble map, per-model logit differences, cell px, number of models)."""
    from src.explain.gradcam import image_maps
    from src.inference.gate import read_image, resolve_scale

    arr, info = read_image(io.BytesIO(data))
    src_mpp, _ = resolve_scale(info, mpp, magnification)
    scaled = rescale_to_model(arr, src_mpp, MODEL_MPP)
    use = [i for i, f in enumerate(pred.member_folds) if folds is None or f in folds]
    maps, diffs, cell = image_maps([pred.members[i] for i in use], scaled, pred.device)
    return scaled, maps.mean(axis=0), diffs, cell, len(use)


def live_heatmap(pred, data: bytes, mpp: float | None, magnification: str | None, folds: set[int] | None, result: dict,
                 scale: float) -> dict | None:
    """Heat map of an image the web demo has just scored (status ok): the same pixels and the same models, drawn like the
    classic figures and on their colour scale. Kept in memory only. None (logged) when a guard fails."""
    from matplotlib import colormaps

    scaled, ens, diffs, cell, n = live_maps(pred, data, mpp, magnification, folds)
    if list(scaled.shape[:2]) != list(result["input"]["rescaled_shape"]):
        log.warning("heat map skipped: image shape %s differs from the scored shape", scaled.shape[:2])
        return None
    p = float(np.mean(1 / (1 + np.exp(-diffs))))
    if abs(p - result["probability_ptc_like"]) > MAX_SCORE_GAP:
        log.warning("heat map skipped: its score %.4f differs from the demo's %.4f", p, result["probability_ptc_like"])
        return None
    ev = evidence(ens)
    down = max(2, math.ceil(scaled.shape[1] / LIVE_MAX_WIDTH))
    heat = (np.clip(overlay(scaled, ev, cell, scale, colormaps["jet"], down), 0, 1) * 255).astype(np.uint8)
    return {"scan_jpeg": jpeg_b64(scan_view(scaled, ev.shape, cell, down)), "overlay_jpeg": jpeg_b64(heat),
            "width": int(heat.shape[1]), "height": int(heat.shape[0]), "models": n, "colour_scale": round(scale, 1),
            "smoothing_um": round(SIGMA_CELLS * cell * MODEL_MPP, 1), "shows": "evidence for 'PTC-like'"}


def load_scan(s: dict, raw: Path, regions: pd.DataFrame) -> np.ndarray:
    """The image exactly as gradcam.py fed it to the model (PTIC as stored; outside regions rescaled to the model's pixel size)."""
    if "image_id" in s:
        with Image.open(raw / s["image_id"]) as im:
            return np.array(im.convert("RGB"))
    r = regions.loc[(s["source"], s["slide_id"], s["region"])]
    with Image.open(PROJECT_ROOT / "data" / "external" / "regions" / r.file) as im:
        return rescale_to_model(np.array(im.convert("RGB")), float(r.mpp_x), MODEL_MPP)


def caption(s: dict) -> str:
    if s["group"] == "A":
        return (f"{s['id']} · {'PTC-like' if s['label'] else 'not PTC-like'} (label) · patient {s['patient_id']}\n"
                f"E3 {s['e3_score_fp32']:.2f} → {s['e3_answer']}")
    if s["group"] == "B":
        return (f"{s['id']} · cancer-like (label) · patient 1, slide {s['slide']}\n"
                f"E1 {s['E1_test_score_report6']:.2f} (Report 6) · E3 {s['e3_score_fp32']:.2f} → {s['e3_answer']}")
    kind, short = (("cancer slide", s["slide_id"].split("-")[:3]) if s["source"] == "TCGA-THCA"
                   else ("normal thyroid", s["slide_id"].split("-")[:2]))
    return (f"{s['id']} · {s['source']} {kind} {'-'.join(short)}, region {s['region']}\n"
            f"E3 alone {s['e3_score_fp32']:.2f} · safety checks: {'refused' if s['checks_status_report8'] == 'refused' else 'accepted'}")


def draw(items: list[dict], title: str, note: str, scale: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    cmap = plt.get_cmap("jet")
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "text.color": "#0b0b0b", "font.size": 9})
    nrow = (len(items) + 1) // 2
    fig, axes = plt.subplots(nrow, 4, figsize=(16, 2.45 * nrow + 1.5), squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        it = items[k // 2] if k // 2 < len(items) else None
        if it is None:
            continue
        arr, ev, cell = it["arr"], it["ev"], it["cell"]
        if k % 2 == 0:                                                         # the scan, unseen strips dimmed
            img = scan_view(arr, ev.shape, cell)
            ax.set_title(it["caption"], loc="left", fontsize=8.6)
        else:                                                                  # the heat map over a black-and-white copy
            img = (np.clip(overlay(arr, ev, cell, scale, cmap), 0, 1) * 255).astype(np.uint8)
        ax.imshow(img, interpolation="nearest")
    H = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.12 / H, ha="left", va="top", fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.1 / H, "\n".join(textwrap.fill(p, 100) for p in note.split("\n")), fontsize=8.6, va="bottom", color="#52514e")
    cax = fig.add_axes([0.56, 0.72 / H, 0.38, 0.12 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, scale), cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([0, scale])
    cb.set_ticklabels(["0\nno evidence for 'PTC-like'", f"{scale:.1f} or more\nstrong evidence for 'PTC-like'"])
    cb.ax.tick_params(labelsize=8.6)
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.75 / H, bottom=1.25 / H, wspace=0.03, hspace=0.24)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPORTS))
    ap.add_argument("--only", default="", help="comma-separated picture ids (e.g. A1,B1,C7): one preview figure, no JSON")
    a = ap.parse_args()
    out = Path(a.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    shown = json.loads((REPORTS / "11_gradcam_shown.json").read_text())["images"]
    cell = json.loads((REPORTS / "11_gradcam_summary.json").read_text())["cell_px"]
    maps_file = REPORTS / "11_gradcam_shown_maps.npz"
    maps = np.load(maps_file)
    ev = {s["id"]: evidence(maps[s["id"]]) for s in shown}
    scale = float(np.percentile(np.concatenate([e.ravel() for e in ev.values()]), 99))   # one scale for all 29 pictures
    raw = resolve(load_config("data.yaml")["paths"]["raw_images"])
    regions = pd.read_csv(PROJECT_ROOT / "data" / "external" / "regions" / "manifest.csv").set_index(["source", "slide_id", "region"])
    only = [i for i in a.only.split(",") if i]
    items = {}
    for s in shown:
        if only and s["id"] not in only:
            continue
        arr = load_scan(s, raw, regions)
        assert ev[s["id"]].shape == (arr.shape[0] // 256 * 16, arr.shape[1] // 256 * 16), (s["id"], ev[s["id"]].shape, arr.shape)
        items[s["id"]] = {"arr": arr, "ev": ev[s["id"]], "cell": cell, "caption": caption(s), "group": s["group"]}
    um = SIGMA_CELLS * cell * MODEL_MPP
    common = (f"Left: the scan. Right: Grad-CAM heat map of the evidence for “PTC-like” over a black-and-white copy of the tissue "
              f"(smoothed for display, σ = {um:.1f} µm). Blue: no such evidence; yellow to red: strong. The same colour scale on every "
              "picture. Plain grey strips: never looked at by the model.")
    if only:
        draw([items[i] for i in only], "Preview", common, scale, out / "figures" / "preview_classic.png")
        print(f"preview written; scale = {scale:.3f}")
        return 0
    files = {}
    for g, (stem, title, extra) in FIGS.items():
        path = out / "figures" / f"11_gradcam_classic_{stem}.png"
        draw([it for it in items.values() if it["group"] == g], title, common + "\n" + extra, scale, path)
        files[g] = path.name
    write_json(out / "11_gradcam_classic.json", {
        "note": "Display only: the Report 11 maps redrawn in the classic Grad-CAM style. No model was run; Report 11's numbers are unchanged.",
        "made_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "source_maps": maps_file.name,
        "source_maps_sha256": hashlib.sha256(maps_file.read_bytes()).hexdigest(), "shown": "evidence for 'PTC-like' (positive part)",
        "smoothing_sigma_cells": SIGMA_CELLS, "smoothing_sigma_um": round(um, 2), "colour_scale_99th_percentile": round(scale, 4),
        "colormap": "jet", "heat_map_share_of_blend": ALPHA, "figures": files})
    print(f"done; scale = {scale:.3f}; {files}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
