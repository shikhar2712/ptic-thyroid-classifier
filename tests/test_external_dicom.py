"""Remote DICOM reader: frames and regions from a synthetic TILED_FULL slide, with and without a basic offset table."""
import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("pydicom", reason="DICOM packages live in .venv (see requirements.txt)")
from pydicom.dataset import Dataset, FileMetaDataset  # noqa: E402
from pydicom.encaps import encapsulate  # noqa: E402
from pydicom.uid import JPEGBaseline8Bit, generate_uid  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.external.dicom_remote import Level, LocalSource, TiledInstance  # noqa: E402
from src.external.extract_regions import draw_regions, tissue_mask  # noqa: E402

TILE, W, H = 32, 100, 70                                   # 4 x 3 tiles, edge tiles padded


def make_slide(path: Path, with_bot: bool) -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.kron(rng.integers(0, 256, (H // 5 + 1, W // 5 + 1, 3)), np.ones((5, 5, 1)))[:H, :W].astype(np.uint8)   # blocky, JPEG-friendly
    across, down = -(-W // TILE), -(-H // TILE)
    pad = np.zeros((down * TILE, across * TILE, 3), np.uint8); pad[:H, :W] = img
    frames = []
    for r in range(down):
        for c in range(across):
            b = io.BytesIO(); Image.fromarray(pad[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]).save(b, "JPEG", quality=100, subsampling=0)
            frames.append(b.getvalue())
    meta = FileMetaDataset(); meta.TransferSyntaxUID = JPEGBaseline8Bit
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.77.1.6"; meta.MediaStorageSOPInstanceUID = generate_uid()
    ds = Dataset(); ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.ImageType = ["ORIGINAL", "PRIMARY", "VOLUME", "NONE"]
    ds.Rows = ds.Columns = TILE; ds.TotalPixelMatrixColumns, ds.TotalPixelMatrixRows = W, H
    ds.NumberOfFrames = len(frames); ds.SamplesPerPixel = 3; ds.PhotometricInterpretation = "YBR_FULL"
    ds.BitsAllocated = ds.BitsStored = 8; ds.HighBit = 7; ds.PixelRepresentation = 0; ds.PlanarConfiguration = 0
    ds.DimensionOrganizationType = "TILED_FULL"; ds.TotalPixelMatrixFocalPlanes = 1; ds.NumberOfOpticalPaths = 1
    pm = Dataset(); pm.PixelSpacing = [0.00025, 0.00025]
    sfg = Dataset(); sfg.PixelMeasuresSequence = [pm]; ds.SharedFunctionalGroupsSequence = [sfg]
    ds.PixelData = encapsulate(frames, has_bot=with_bot); ds["PixelData"].VR = "OB"
    ds.save_as(path, enforce_file_format=True)
    return img


def test_regions_and_scan_match(tmp_path):
    out = {}
    for bot in (True, False):
        p = tmp_path / f"s_{bot}.dcm"
        img = make_slide(p, bot)
        t = TiledInstance(LocalSource(p))
        assert t.offset_table == ("basic" if bot else "none")
        lv = Level([t])
        out[bot] = [lv.region(0, 0, W, H), lv.region(37, 21, 50, 40), lv.region(W - 5, H - 3, 5, 3)]
        assert out[bot][0].shape == (H, W, 3)
        assert np.abs(out[bot][0].astype(int) - img).mean() < 3            # JPEG quality 100, 4:4:4
        assert np.array_equal(out[bot][1], out[bot][0][21:61, 37:87])     # a region equals the same window of the whole
        assert t.info["spacing_um"] == [0.25, 0.25]
    for a, b in zip(out[True], out[False]):                                  # scanned offsets == offset-table offsets
        assert np.array_equal(a, b)


def test_draw_regions_rules():
    mask = np.zeros((100, 200), bool); mask[20:80, 30:170] = True           # tissue block
    regs, draws = draw_regions(mask, (2000, 1000), (200, 110), seed=7)
    assert len(regs) == 10 and draws >= 10
    assert all(f >= 0.7 for _, _, f in regs)
    for i, (x, y, _) in enumerate(regs):                                    # never overlapping
        for (x2, y2, _) in regs[i + 1:]:
            assert not (x < x2 + 200 and x2 < x + 200 and y < y2 + 110 and y2 < y + 110)
    assert draw_regions(mask, (2000, 1000), (200, 110), seed=7) == (regs, draws)   # deterministic
    rgb = np.full((4, 4, 3), 240, np.uint8); rgb[0, 0] = (150, 60, 120); rgb[1, 1] = (10, 10, 10)
    assert tissue_mask(rgb).sum() == 1                                      # glass and black are not tissue
