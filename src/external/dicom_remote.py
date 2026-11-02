"""Read regions of a DICOM whole-slide image without downloading the whole file.

Only the header, the frame offset table and the tiles a region needs are fetched (HTTP range requests on the public IDC
buckets, or a local file). Supports what the selected TCGA/GTEx slides use: TILED_FULL, one focal plane, one optical path,
encapsulated frames with a basic or extended offset table, decoded with Pillow. Anything else raises rather than guessing.
"""
from __future__ import annotations

import io
import re
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pydicom
import requests
from PIL import Image

PIXEL_DATA_TAG = b"\xe0\x7f\x10\x00"
ITEM_TAG = b"\xfe\xff\x00\xe0"
_local = threading.local()


def _session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


class HttpSource:
    def __init__(self, url: str, size: int) -> None:
        self.url, self.size, self.bytes_read, self.requests = url, int(size), 0, 0

    def read(self, start: int, end: int) -> bytes:
        """Bytes [start, end), with retries."""
        end = min(end, self.size)
        for attempt in range(6):
            try:
                r = _session().get(self.url, headers={"Range": f"bytes={start}-{end - 1}"}, timeout=120)
                if r.status_code == 206 and len(r.content) == end - start:
                    self.bytes_read += end - start; self.requests += 1
                    return r.content
                err = f"HTTP {r.status_code}, {len(r.content)} bytes"
            except requests.RequestException as exc:
                err = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
        raise IOError(f"range read failed for {self.url} [{start}, {end}): {err}")


class LocalSource:
    def __init__(self, path: str | Path) -> None:
        self.path, self.size, self.bytes_read, self.requests = Path(path), Path(path).stat().st_size, 0, 0

    def read(self, start: int, end: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(start); b = f.read(min(end, self.size) - start)
        self.bytes_read += len(b); self.requests += 1
        return b


def list_series_files(series_aws_url: str) -> list[tuple[str, int]]:
    """'s3://bucket/<uuid>/*' -> [(https url, size)] from the public bucket listing."""
    m = re.match(r"s3://([^/]+)/([^/]+)/", series_aws_url)
    bucket, prefix = m.group(1), m.group(2) + "/"
    r = _session().get(f"https://{bucket}.s3.amazonaws.com/", params={"list-type": "2", "prefix": prefix}, timeout=60)
    r.raise_for_status()
    ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
    root = ET.fromstring(r.content)
    assert root.findtext("s3:IsTruncated", namespaces=ns) == "false", "listing truncated"
    return [(f"https://{bucket}.s3.amazonaws.com/{c.findtext('s3:Key', namespaces=ns)}", int(c.findtext("s3:Size", namespaces=ns)))
            for c in root.findall("s3:Contents", ns)]


class TiledInstance:
    """One DICOM instance (one pyramid level, or one part of a concatenated level)."""

    def __init__(self, src) -> None:
        self.src = src
        n = 1 << 18
        while True:                                     # read a growing prefix until the header parses up to Pixel Data
            head = src.read(0, n)
            fp = io.BytesIO(head)
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True)
                pos = fp.tell()
                if head[pos:pos + 4] == PIXEL_DATA_TAG:
                    break
            except Exception:
                pass
            if n >= src.size:
                raise ValueError("no encapsulated Pixel Data found")
            n *= 4
        self.ds = ds
        self.n_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
        self.image_type = list(ds.ImageType)
        self.offsets, self.offset_table = None, "native"        # thumbnail / overview / label: not read
        if len(self.image_type) > 2 and self.image_type[2] == "VOLUME":
            assert head[pos + 4:pos + 6] == b"OB" and head[pos + 8:pos + 12] == b"\xff\xff\xff\xff", "Pixel Data is not encapsulated"
            bot = src.read(pos + 12, pos + 20)
            assert bot[:4] == ITEM_TAG
            bot_len = int.from_bytes(bot[4:8], "little")
            self.first_item = pos + 20 + bot_len             # offsets are relative to the first frame's item tag
            if bot_len:
                self.offsets = np.frombuffer(src.read(pos + 20, pos + 20 + bot_len), "<u4").astype(np.int64)
                self.offset_table = "basic"
            elif "ExtendedOffsetTable" in ds:
                self.offsets = np.frombuffer(ds.ExtendedOffsetTable, "<u8").astype(np.int64)
                self.offset_table = "extended"
            else:
                self.offset_table = "none"                   # indexed on first use by streaming the frames (see _scan)
            if self.offsets is not None:
                assert len(self.offsets) == self.n_frames and np.all(np.diff(self.offsets) > 0)
        self._buf: bytes | None = None
        self.tile_w, self.tile_h = int(ds.Columns), int(ds.Rows)
        self.width = int(getattr(ds, "TotalPixelMatrixColumns", self.tile_w))
        self.height = int(getattr(ds, "TotalPixelMatrixRows", self.tile_h))
        self.tiles_across = -(-self.width // self.tile_w)
        self.tiles_down = -(-self.height // self.tile_h)
        self.concat_offset = int(getattr(ds, "ConcatenationFrameOffsetNumber", 0) or 0)
        try:
            pm = ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0]
            self.spacing_um = (float(pm.PixelSpacing[0]) * 1000, float(pm.PixelSpacing[1]) * 1000)   # (row, column) in um
        except (AttributeError, IndexError):
            self.spacing_um = (float("nan"), float("nan"))
        self.info = {"image_type": "\\".join(self.image_type), "width": self.width, "height": self.height,
                     "tile": [self.tile_w, self.tile_h], "frames": self.n_frames, "offset_table": self.offset_table,
                     "spacing_um": [round(s, 5) for s in self.spacing_um],
                     "transfer_syntax": str(ds.file_meta.TransferSyntaxUID), "photometric": str(ds.PhotometricInterpretation),
                     "organisation": str(getattr(ds, "DimensionOrganizationType", "")),
                     "focal_planes": int(getattr(ds, "TotalPixelMatrixFocalPlanes", 1) or 1),
                     "optical_paths": int(getattr(ds, "NumberOfOpticalPaths", 1) or 1), "concat_offset": self.concat_offset,
                     "manufacturer": str(getattr(ds, "Manufacturer", "")),
                     "model": str(getattr(ds, "ManufacturerModelName", "")), "bytes": src.size}

    def _scan(self, threads: int = 8, chunk: int = 32 << 20) -> None:
        """No offset table: fetch all frame bytes (parallel range reads, kept in memory only) and walk the item headers."""
        n = self.src.size - self.first_item
        spans = [(s, min(n, s + chunk)) for s in range(0, n, chunk)]
        with ThreadPoolExecutor(threads) as ex:
            parts = list(ex.map(lambda sp: self.src.read(self.first_item + sp[0], self.first_item + sp[1]), spans))
        buf = b"".join(parts); del parts
        offs, p = [], 0
        while buf[p:p + 4] == ITEM_TAG:
            offs.append(p); p += 8 + int.from_bytes(buf[p + 4:p + 8], "little")
        assert buf[p:p + 4] == b"\xfe\xff\xdd\xe0", "frames not followed by the sequence delimiter"
        assert len(offs) == self.n_frames, f"{len(offs)} frame items, header says {self.n_frames}"
        self.offsets, self._buf, self.offset_table = np.array(offs, np.int64), buf, "none (scanned)"
        self.info["offset_table"] = self.offset_table

    def release(self) -> None:
        self._buf = None

    def frames(self, a: int, b: int) -> list[bytes]:
        """Encoded frames a..b (local indices, inclusive) with one contiguous read."""
        if self.offsets is None:
            self._scan()
        start = int(self.offsets[a])
        end = int(self.offsets[b + 1]) if b + 1 < self.n_frames else self.src.size - self.first_item
        buf = self._buf[start:end] if self._buf is not None else self.src.read(self.first_item + start, self.first_item + end)
        out = []
        for i in range(a, b + 1):
            p = int(self.offsets[i] - self.offsets[a])
            assert buf[p:p + 4] == ITEM_TAG, "frame item tag not where the offset table says"
            ln = int.from_bytes(buf[p + 4:p + 8], "little")
            out.append(buf[p + 8:p + 8 + ln])
        return out


class Level:
    """A pyramid level, possibly split over several concatenated instances. Pixels as uint8 RGB."""

    def __init__(self, parts: list[TiledInstance]) -> None:
        self.parts = sorted(parts, key=lambda t: t.concat_offset)
        t = self.parts[0]
        for k in ("transfer_syntax", "organisation"):
            assert all(p.info[k] == t.info[k] for p in self.parts)
        assert t.info["organisation"] == "TILED_FULL", f"unsupported organisation {t.info['organisation']}"
        assert t.info["focal_planes"] == 1 and t.info["optical_paths"] == 1
        assert t.info["transfer_syntax"] in ("1.2.840.10008.1.2.4.50", "1.2.840.10008.1.2.4.90", "1.2.840.10008.1.2.4.91"), t.info["transfer_syntax"]
        self.width, self.height, self.tile_w, self.tile_h = t.width, t.height, t.tile_w, t.tile_h
        self.tiles_across, self.tiles_down, self.spacing_um = t.tiles_across, t.tiles_down, t.spacing_um
        total = sum(p.n_frames for p in self.parts)
        assert total == self.tiles_across * self.tiles_down, f"{total} frames for {self.tiles_across}x{self.tiles_down} tiles"

    def _frames(self, g0: int, g1: int) -> list[bytes]:
        out = []
        for p in self.parts:
            a, b = max(g0, p.concat_offset), min(g1, p.concat_offset + p.n_frames - 1)
            if a <= b:
                out += p.frames(a - p.concat_offset, b - p.concat_offset)
        assert len(out) == g1 - g0 + 1
        return out

    def region(self, x: int, y: int, w: int, h: int) -> np.ndarray:
        assert 0 <= x and 0 <= y and x + w <= self.width and y + h <= self.height
        c0, c1 = x // self.tile_w, (x + w - 1) // self.tile_w
        r0, r1 = y // self.tile_h, (y + h - 1) // self.tile_h
        canvas = np.zeros(((r1 - r0 + 1) * self.tile_h, (c1 - c0 + 1) * self.tile_w, 3), np.uint8)
        for r in range(r0, r1 + 1):
            for k, data in enumerate(self._frames(r * self.tiles_across + c0, r * self.tiles_across + c1)):
                tile = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
                assert tile.shape[:2] == (self.tile_h, self.tile_w), tile.shape
                canvas[(r - r0) * self.tile_h:(r - r0 + 1) * self.tile_h, k * self.tile_w:(k + 1) * self.tile_w] = tile
        oy, ox = y - r0 * self.tile_h, x - c0 * self.tile_w
        return canvas[oy:oy + h, ox:ox + w].copy()

    def whole(self) -> np.ndarray:
        return self.region(0, 0, self.width, self.height)


def open_pyramid(sources: list) -> tuple[list[Level], list[dict]]:
    """Instances of one series -> VOLUME levels sorted from full resolution to lowest, plus per-instance info."""
    insts = [TiledInstance(s) for s in sources]
    vol = [t for t in insts if len(t.image_type) > 2 and t.image_type[2] == "VOLUME"]
    by_size: dict[tuple[int, int], list[TiledInstance]] = {}
    for t in vol:
        by_size.setdefault((t.width, t.height), []).append(t)
    levels = sorted((Level(v) for v in by_size.values()), key=lambda lv: -lv.width)
    return levels, [t.info for t in insts]
