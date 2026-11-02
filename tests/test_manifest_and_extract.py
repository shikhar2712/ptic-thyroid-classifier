import zipfile

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.common import load_config
from src.data.extract import check_member_names, safe_extract
from src.data.manifest import build_manifest, dhash64, inspect_image, phash64
from src.data.quality import add_flags

CFG = load_config("data.yaml")


def _write_jpg(path, rgb=(200, 120, 180), size=(64, 48), noise=0, icc=b"", seed=0):
    rng = np.random.default_rng(seed)
    arr = np.clip(np.full((size[1], size[0], 3), rgb, dtype=np.int16) + rng.integers(-noise, noise + 1, (size[1], size[0], 3)), 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    kw = {"icc_profile": icc} if icc else {}
    Image.fromarray(arr).save(path, quality=95, **kw)


def _tiny_dataset(root):
    _write_jpg(root / "PTC-like" / "1_1 (1).jpg", noise=30, seed=1)
    _write_jpg(root / "PTC-like" / "1_1 (2).jpg", noise=30, seed=2)
    _write_jpg(root / "Non PTC-like" / "2_1 (1).jpg", rgb=(120, 140, 200), noise=30, seed=3)
    _write_jpg(root / "Non PTC-like" / "weird name.jpg", noise=30, seed=4)


def test_manifest_columns_labels_and_ids(tmp_path):
    _tiny_dataset(tmp_path)
    cfg = {**CFG, "expected": {**CFG["expected"], "image_size_wh": [64, 48]}}
    df = build_manifest(tmp_path, cfg, workers=2)
    assert len(df) == 4 and df.readable.all()
    good = df[df.filename_parse_ok]
    assert set(zip(good.patient_id, good.wsi_id, good.patch_no, good.class_name, good.label)) == {
        (1, 1, 1, "PTC-like", 1), (1, 1, 2, "PTC-like", 1), (2, 1, 1, "Non-PTC-like", 0)}
    assert (df[~df.filename_parse_ok].original_filename == "weird name.jpg").all()  # kept, flagged, never invented IDs
    assert df[~df.filename_parse_ok].patient_id.isna().all()
    assert set(df.columns) >= {"sha256", "icc_sha256", "width", "height", "mode", "file_size", "phash", "dhash"}


def test_unreadable_and_truncated_files_are_recorded_not_raised(tmp_path):
    _write_jpg(tmp_path / "PTC-like" / "1_1 (1).jpg", noise=30)
    good = (tmp_path / "PTC-like" / "1_1 (1).jpg").read_bytes()
    (tmp_path / "PTC-like" / "1_1 (2).jpg").write_bytes(good[: len(good) // 2])  # truncated
    (tmp_path / "PTC-like" / "1_1 (3).jpg").write_bytes(b"not a jpeg")
    r_trunc = inspect_image(tmp_path / "PTC-like" / "1_1 (2).jpg")
    r_junk = inspect_image(tmp_path / "PTC-like" / "1_1 (3).jpg")
    assert not r_trunc["readable"] and r_trunc["error"]
    assert not r_junk["readable"] and r_junk["error"]
    assert inspect_image(tmp_path / "PTC-like" / "1_1 (1).jpg")["readable"]


def test_icc_profile_is_hashed(tmp_path):
    _write_jpg(tmp_path / "a.jpg", icc=b"\x00" * 200 + b"abc", noise=10)
    _write_jpg(tmp_path / "b.jpg", noise=10)
    assert inspect_image(tmp_path / "a.jpg")["icc_len"] > 0 and inspect_image(tmp_path / "b.jpg")["icc_len"] == 0


def test_perceptual_hashes_stable_and_discriminative():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (80, 120), dtype=np.uint8)
    a = Image.fromarray(base)
    b = Image.fromarray(np.clip(base.astype(int) + 2, 0, 255).astype(np.uint8))
    c = Image.fromarray(rng.integers(0, 255, (80, 120), dtype=np.uint8))
    ham = lambda x, y: bin(int(x, 16) ^ int(y, 16)).count("1")  # noqa: E731
    assert ham(phash64(a), phash64(b)) <= 4 and ham(dhash64(a), dhash64(b)) <= 4
    assert ham(phash64(a), phash64(c)) > 10


def test_exact_duplicate_flag(tmp_path):
    _write_jpg(tmp_path / "PTC-like" / "1_1 (1).jpg", noise=30, seed=7)
    (tmp_path / "Non PTC-like").mkdir()
    (tmp_path / "Non PTC-like" / "2_1 (1).jpg").write_bytes((tmp_path / "PTC-like" / "1_1 (1).jpg").read_bytes())
    _write_jpg(tmp_path / "Non PTC-like" / "3_1 (1).jpg", rgb=(100, 100, 100), noise=40, seed=9)
    cfg = {**CFG, "expected": {**CFG["expected"], "image_size_wh": [64, 48]}}
    df = build_manifest(tmp_path, cfg, workers=2)
    df["in_xlsx"] = True
    out, pairs, _ = add_flags(df, cfg)
    dup = out[out.exact_duplicate]
    assert set(dup.patient_id) == {1, 2} and dup.dup_group.nunique() == 1
    assert out.loc[out.patient_id == 3, "exact_duplicate"].eq(False).all()
    assert bool(pairs.label_conflict.any())  # the duplicate sits in both class folders -> a label conflict


def test_unsafe_zip_member_names_rejected(tmp_path):
    assert check_member_names(["ok/a.jpg"], tmp_path) == []
    assert check_member_names(["../evil.jpg"], tmp_path)
    assert check_member_names(["/abs/evil.jpg"], tmp_path)
    zp = tmp_path / "bad.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("../escape.txt", "x")
    with pytest.raises(ValueError, match="refusing"):
        safe_extract(zp, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_refuses_nonempty_target_and_sets_readonly(tmp_path):
    zp = tmp_path / "ok.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("PTC-like/1_1 (1).jpg", b"data")
    busy = tmp_path / "busy"
    busy.mkdir()
    (busy / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        safe_extract(zp, busy)
    out = tmp_path / "out"
    stats = safe_extract(zp, out)
    assert stats["files_extracted"] == 1
    f = out / "PTC-like" / "1_1 (1).jpg"
    assert f.read_bytes() == b"data" and not (f.stat().st_mode & 0o222)
    for p in [f, f.parent, out]:  # restore permissions so pytest can clean up
        p.chmod(0o755)
