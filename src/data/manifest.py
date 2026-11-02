"""Build the image manifest: one row per file, with integrity, geometry, colour/quality statistics and hashes.

Every image is fully decoded (so truncated/corrupt files are caught), not just header-sniffed.
No file is modified or removed; problems are recorded as columns and flags.
Run:  python -I src/data/manifest.py
"""
from __future__ import annotations

import hashlib
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, JpegImagePlugin  # noqa: E402
from scipy import fft, ndimage  # noqa: E402

from src.common import get_logger, load_config, resolve, write_json  # noqa: E402
from src.data.metadata import label_from_folder, load_wsi_metadata, parse_filename, reconcile_with_xlsx  # noqa: E402

_GRAY = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def phash64(gray_img: Image.Image) -> str:
    """64-bit DCT perceptual hash (32x32 -> top-left 8x8 DCT block, thresholded at its median, DC excluded)."""
    a = np.asarray(gray_img.resize((32, 32), Image.LANCZOS), dtype=np.float32)
    d = fft.dctn(a, norm="ortho")[:8, :8].ravel()
    bits = d > np.median(d[1:])
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def dhash64(gray_img: Image.Image) -> str:
    """64-bit difference hash (9x8 -> horizontal gradient sign)."""
    a = np.asarray(gray_img.resize((9, 8), Image.LANCZOS), dtype=np.float32)
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return f"{int(''.join('1' if b else '0' for b in bits), 2):016x}"


def inspect_image(path: Path) -> dict[str, Any]:
    """Fully decode one image and return integrity, geometry, colour statistics and hashes."""
    out: dict[str, Any] = {"readable": False, "error": ""}
    try:
        data = path.read_bytes()
        out["file_size"] = len(data)
        out["sha256"] = hashlib.sha256(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            out.update(format=im.format, mode=im.mode, width=im.width, height=im.height)
            icc = im.info.get("icc_profile") or b""
            out["icc_len"] = len(icc)
            out["icc_sha256"] = hashlib.sha256(icc).hexdigest() if icc else ""
            if im.format == "JPEG":
                q = im.quantization
                out["jpeg_q0_mean"] = float(np.mean(list(q[0]))) if 0 in q else float("nan")
                out["jpeg_subsampling"] = JpegImagePlugin.get_sampling(im)
            im.load()  # raises on truncated / corrupt data
            rgb = im.convert("RGB")
        arr = np.asarray(rgb)
        f = arr.astype(np.float32)
        gray = f @ _GRAY
        out.update(
            mean_r=float(f[..., 0].mean()), mean_g=float(f[..., 1].mean()), mean_b=float(f[..., 2].mean()),
            std_r=float(f[..., 0].std()), std_g=float(f[..., 1].std()), std_b=float(f[..., 2].std()),
            gray_std=float(gray.std()),
            lap_var=float(ndimage.laplace(gray).var()),
            white_frac=float((arr.min(axis=2) > 220).mean()),
            nan_or_inf=bool(not np.isfinite(f).all()),
        )
        g = rgb.convert("L")
        out["phash"], out["dhash"] = phash64(g), dhash64(g)
        out["readable"] = True
    except Exception as exc:  # recorded, never raised: bad files must show up in the report
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def build_manifest(images_root: Path, cfg: dict, workers: int = 48, log=None) -> pd.DataFrame:
    regex = cfg["filename_regex"]
    records: list[dict[str, Any]] = []
    for folder in sorted(p for p in images_root.iterdir() if p.is_dir()):
        for path in sorted(folder.iterdir()):
            rec: dict[str, Any] = {"image_id": f"{folder.name}/{path.name}", "path": str(path.relative_to(images_root)),
                                   "original_filename": path.name, "folder": folder.name, "extension": path.suffix.lower()}
            parsed = parse_filename(path.name, regex)
            rec["filename_parse_ok"] = parsed is not None
            if parsed:
                rec.update(patient_id=parsed["patient"], wsi_id=parsed["wsi"], patch_no=parsed["patch"])
            try:
                rec["class_name"], rec["label"] = label_from_folder(folder.name, cfg["labels"])
            except KeyError:
                rec["class_name"], rec["label"] = "", -1
            records.append(rec)
    paths = [images_root / r["path"] for r in records]
    if log:
        log.info("inspecting %d files with %d threads", len(paths), workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(inspect_image, paths))
    for rec, res in zip(records, results):
        rec.update(res)
    df = pd.DataFrame(records)
    df["wsi_key"] = df.patient_id.astype("Int64").astype(str) + "_" + df.wsi_id.astype("Int64").astype(str)
    return df.sort_values(["label", "patient_id", "wsi_id", "patch_no"]).reset_index(drop=True)


def main() -> int:
    cfg = load_config("data.yaml")
    log = get_logger("manifest", resolve(cfg["paths"]["logs"]) / "manifest.log")
    images_root = resolve(cfg["paths"]["raw_images"])
    meta_dir = resolve(cfg["paths"]["metadata"])
    df = build_manifest(images_root, cfg, log=log)
    wsi_meta = load_wsi_metadata(resolve(cfg["paths"]["raw_docs"]) / "Data file 3-PTIC_metadata.xlsx",
                                 cfg["labels"]["xlsx_map"])
    rec = reconcile_with_xlsx(df[df.filename_parse_ok], wsi_meta)
    key = df.assign(k=list(zip(df.patient_id, df.wsi_id, df.class_name))).k
    xl_keys = set(zip(wsi_meta.patient_id, wsi_meta.wsi_id, wsi_meta.label_name))
    df["in_xlsx"] = key.isin(xl_keys)
    meta_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(meta_dir / "manifest_raw.csv", index=False)
    wsi_meta.to_csv(meta_dir / "wsi_metadata_from_xlsx.csv", index=False)
    write_json(meta_dir / "xlsx_reconciliation.json", rec)
    log.info("manifest: %d rows, %d readable, %d parse failures, %d not in xlsx",
             len(df), int(df.readable.sum()), int((~df.filename_parse_ok).sum()), int((~df.in_xlsx).sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
