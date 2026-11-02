"""Extract 10 random tissue regions per selected external slide (rules: reports/08_external_test.md section 1). No model is run here.

Per slide: open the DICOM pyramid on the public IDC bucket (range reads, no full download), check the pixel spacing, build a tissue
mask on the lowest-resolution level, draw region positions uniformly at random (seed from the slide ID) among non-overlapping
positions with >= 70% tissue, read those regions at full resolution and save them losslessly (PNG) with a manifest.

  .venv/bin/python -I src/external/extract_regions.py                       # all 40 slides -> data/external/regions/
  .venv/bin/python -I src/external/extract_regions.py --slides 2 --out <dir> # smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.common import PROJECT_ROOT  # noqa: E402
from src.external.dicom_remote import HttpSource, list_series_files, open_pyramid  # noqa: E402
from src.inference.gate import MODEL_MPP  # noqa: E402

PTIC_SIZE_PX = (1916, 1053)              # a PTIC image (width, height) at MODEL_MPP: 498 x 274 um
EXPECTED_MPP = {"TCGA-THCA": 0.25, "GTEx": 0.494}
N_REGIONS, MIN_TISSUE, MAX_DRAWS = 10, 0.70, 200_000


def tissue_mask(rgb: np.ndarray) -> np.ndarray:
    """Tissue = not glass (bright and neutral, the same rule as the safety checks' background) and not near-black. Pen ink counts
    as non-glass; it is not removed."""
    x = rgb.astype(np.int16); mn, mx = x.min(2), x.max(2)
    return ~((mn > 215) & ((mx - mn) < 20)) & (mx > 40)


def slide_seed(slide_id: str) -> int:
    return int.from_bytes(hashlib.sha256(slide_id.encode()).digest()[:8], "little")


def draw_regions(mask: np.ndarray, full_wh: tuple[int, int], region_wh: tuple[int, int], seed: int) -> tuple[list[tuple[int, int, float]], int]:
    """Uniform random top-left corners at full resolution, kept if >= MIN_TISSUE tissue on the mask and not overlapping earlier ones."""
    W, H = full_wh; w, h = region_wh
    mh, mw = mask.shape
    fx, fy = mw / W, mh / H
    ii = np.zeros((mh + 1, mw + 1)); ii[1:, 1:] = mask.astype(np.float64).cumsum(0).cumsum(1)
    rng = np.random.default_rng(seed)
    kept: list[tuple[int, int, float]] = []
    for draw in range(1, MAX_DRAWS + 1):
        x, y = int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1))
        x0, y0 = int(np.floor(x * fx)), int(np.floor(y * fy))
        x1, y1 = min(mw, max(x0 + 1, int(np.ceil((x + w) * fx)))), min(mh, max(y0 + 1, int(np.ceil((y + h) * fy))))
        frac = (ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]) / ((y1 - y0) * (x1 - x0))
        if frac < MIN_TISSUE or any(x < kx + w and kx < x + w and y < ky + h and ky < y + h for kx, ky, _ in kept):
            continue
        kept.append((x, y, float(frac)))
        if len(kept) == N_REGIONS:
            return kept, draw
    return kept, MAX_DRAWS


def process(row: pd.Series, out: Path) -> tuple[list[dict], dict]:
    t0 = time.time()
    srcs = [HttpSource(u, s) for u, s in list_series_files(row.series_aws_url)]
    levels, infos = open_pyramid(srcs)
    L0, low = levels[0], levels[-1]
    sy, sx = L0.spacing_um
    exp = EXPECTED_MPP[row.source]
    assert abs(sx / exp - 1) < 0.05 and abs(sy / exp - 1) < 0.05, f"pixel spacing {sx}, {sy} um not within 5% of {exp}"
    w, h = round(PTIC_SIZE_PX[0] * MODEL_MPP / sx), round(PTIC_SIZE_PX[1] * MODEL_MPP / sy)
    mask = tissue_mask(low.whole())
    regions, draws = draw_regions(mask, (L0.width, L0.height), (w, h), slide_seed(row.slide_id))
    d = out / row.source / row.slide_id
    d.mkdir(parents=True, exist_ok=False)
    Image.fromarray((mask * 255).astype(np.uint8)).save(d / "tissue_mask_lowest_level.png")
    rows = []
    for k, (x, y, frac) in enumerate(regions):
        arr = L0.region(x, y, w, h)
        p = d / f"region_{k:02d}.png"
        Image.fromarray(arr).save(p, compress_level=3)
        rows.append({"source": row.source, "slide_id": row.slide_id, "region": k, "file": str(p.relative_to(out)),
                     "x": x, "y": y, "width": w, "height": h, "mpp_x": round(sx, 5), "mpp_y": round(sy, 5),
                     "tissue_fraction_lowest_level": round(frac, 4), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()})
    offset_tables = sorted({t.info["offset_table"] for lv in (L0, low) for t in lv.parts})
    for lv in levels:
        for t in lv.parts:
            t.release()                                          # frame bytes held in memory for slides without an offset table
    info = {"slide_id": row.slide_id, "source": row.source, "regions": len(regions), "draws": draws, "offset_tables_used": offset_tables,
            "level0": [L0.width, L0.height], "lowest_level": [low.width, low.height], "levels": len(levels),
            "mask_tissue_share": round(float(mask.mean()), 4), "region_px": [w, h],
            "bytes_read": sum(s.bytes_read for s in srcs), "requests": sum(s.requests for s in srcs),
            "full_series_bytes": sum(s.size for s in srcs), "seconds": round(time.time() - t0, 1), "instances": infos}
    return rows, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(PROJECT_ROOT / "data" / "external" / "regions"))
    ap.add_argument("--slides", type=int, default=None, help="smoke test: first N slides of each source")
    ap.add_argument("--only", nargs="*", default=None, help="smoke test: these slide IDs")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)                     # fresh directory only
    S = pd.read_csv(PROJECT_ROOT / "data" / "external" / "selection.csv")
    if a.slides:
        S = S.groupby("source", sort=False).head(a.slides)
    if a.only:
        S = S[S.slide_id.isin(a.only)]
    t0 = time.time()
    rows, infos, errors = [], [], []
    with ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(process, r, out): r.slide_id for r in S.itertuples(index=False)}
        for f, sid in futs.items():
            try:
                r, i = f.result()
                rows += r; infos.append(i)
                print(f"{sid}: {i['regions']} regions, {i['bytes_read'] / 1e6:.1f} of {i['full_series_bytes'] / 1e6:.0f} MB read, {i['seconds']}s", flush=True)
            except Exception as exc:                            # reported, never silently replaced
                errors.append({"slide_id": sid, "error": f"{type(exc).__name__}: {exc}"})
                print(f"{sid}: FAILED {type(exc).__name__}: {exc}", flush=True)
    M = pd.DataFrame(rows)
    M = M.sort_values(["source", "slide_id", "region"]) if len(M) else M
    M.to_csv(out / "manifest.csv", index=False)
    (out / "slides.json").write_text(json.dumps({"slides": infos, "errors": errors, "seconds": round(time.time() - t0, 1)}, indent=1))
    print(f"done: {len(M)} regions from {M.slide_id.nunique() if len(M) else 0} slides, {len(errors)} failed, "
          f"{sum(i['bytes_read'] for i in infos) / 1e9:.2f} GB read, {time.time() - t0:.0f}s")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
