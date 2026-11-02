"""Resumable, checksum-verified download of the PTIC Zenodo record.

Idempotent: a file that already exists with the published size and MD5 is not fetched again.
Run:  python -I src/data/download.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import requests  # noqa: E402

from src.common import PROJECT_ROOT, file_hash, get_logger, load_config, resolve, write_json  # noqa: E402

CHUNK = 8 << 20


def fetch_record(api_url: str) -> dict:
    r = requests.get(api_url, timeout=60)
    r.raise_for_status()
    return r.json()


def download_file(url: str, dest: Path, size: int, log, retries: int = 8) -> None:
    """Download with HTTP Range resume into `<dest>.part`, then rename. Never overwrites a complete dest."""
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        if have > size:  # corrupt partial: restart this file only
            part.unlink()
            have = 0
        if have == size:
            break
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
                if have and r.status_code != 206:
                    log.warning("server ignored Range (HTTP %s); restarting %s", r.status_code, dest.name)
                    part.unlink(missing_ok=True)
                    have = 0
                    continue
                r.raise_for_status()
                t0, last, got = time.time(), time.time(), 0
                with open(part, "ab" if have else "wb") as fh:
                    for block in r.iter_content(CHUNK):
                        fh.write(block)
                        got += len(block)
                        if time.time() - last > 30:
                            done = have + got
                            log.info("%s: %.2f / %.2f GB (%.0f MB/s)", dest.name, done / 1e9, size / 1e9,
                                     got / 1e6 / max(time.time() - t0, 1e-6))
                            last = time.time()
        except (requests.RequestException, OSError) as exc:
            log.warning("attempt %d/%d failed for %s: %r", attempt, retries, dest.name, exc)
            time.sleep(min(60, 5 * attempt))
    if not part.exists() or part.stat().st_size != size:
        raise RuntimeError(f"incomplete download for {dest.name}")
    part.replace(dest)


def download_segmented(url: str, dest: Path, size: int, log, connections: int = 16,
                       segment: int = 64 << 20, retries: int = 12) -> None:
    """Parallel ranged download for hosts that throttle each connection (Zenodo: ~0.2-0.8 MB/s per stream).

    The target is pre-sized, segments are written with `pwrite` at their own offsets, and finished segments are
    recorded in `<dest>.part.json` so an interrupted run resumes without refetching them. A legacy sequential
    `<dest>.part` (no sidecar) is treated as a valid prefix. Integrity is decided by the caller's MD5 check.
    """
    import json
    import os
    import threading
    from concurrent.futures import ThreadPoolExecutor

    part = dest.with_name(dest.name + ".part")
    side = dest.with_name(dest.name + ".part.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    n_seg = -(-size // segment)
    done: set[int] = set()
    if side.exists():
        meta = json.loads(side.read_text())
        if meta.get("size") == size and meta.get("segment") == segment:
            done = set(meta["done"])
    elif part.exists():  # prefix written by the sequential downloader
        done = set(range(min(part.stat().st_size, size) // segment))
        log.info("reusing %d segments from a sequential partial download", len(done))
    fd = os.open(part, os.O_RDWR | os.O_CREAT)
    os.ftruncate(fd, size)
    lock, t0, got_total = threading.Lock(), time.time(), [0]

    def save() -> None:
        side.write_text(json.dumps({"size": size, "segment": segment, "done": sorted(done)}))

    def fetch(idx: int) -> None:
        a, b = idx * segment, min((idx + 1) * segment, size) - 1
        pos = a
        for attempt in range(1, retries + 1):
            try:
                with requests.get(url, headers={"Range": f"bytes={pos}-{b}"}, stream=True, timeout=(30, 120)) as r:
                    if r.status_code != 206:
                        raise OSError(f"HTTP {r.status_code} for range {pos}-{b}")
                    for block in r.iter_content(1 << 20):
                        os.pwrite(fd, block, pos)
                        pos += len(block)
                        with lock:
                            got_total[0] += len(block)
                if pos != b + 1:
                    raise OSError(f"short segment {idx}: ended at {pos}, expected {b + 1}")
                with lock:
                    done.add(idx)
                    if len(done) % 8 == 0:
                        save()
                        log.info("%d/%d segments (%.2f GB fetched this run, %.1f MB/s)", len(done), n_seg,
                                 got_total[0] / 1e9, got_total[0] / 1e6 / max(time.time() - t0, 1e-6))
                return
            except (requests.RequestException, OSError) as exc:
                log.warning("segment %d attempt %d/%d failed: %r", idx, attempt, retries, exc)
                time.sleep(min(60, 3 * attempt))
        raise RuntimeError(f"segment {idx} failed after {retries} attempts")

    try:
        todo = [i for i in range(n_seg) if i not in done]
        log.info("segmented download: %d segments pending of %d, %d connections", len(todo), n_seg, connections)
        with ThreadPoolExecutor(max_workers=connections) as ex:
            list(ex.map(fetch, todo))
    finally:
        with lock:
            save()
        os.close(fd)
    if len(done) != n_seg:
        raise RuntimeError(f"incomplete: {len(done)}/{n_seg} segments")
    part.replace(dest)
    side.unlink(missing_ok=True)


def main() -> int:
    cfg = load_config("data.yaml")
    log = get_logger("download", resolve(cfg["paths"]["logs"]) / "download.log")
    rec = fetch_record(cfg["zenodo"]["api_url"])
    files = {f["key"]: f for f in rec["files"]}
    acq_path = resolve(cfg["paths"]["metadata"]) / "acquisition.json"
    prev = {f["name"]: f for f in json.loads(acq_path.read_text())["files"]} if acq_path.exists() else {}
    acquisition = {"record_id": cfg["zenodo"]["record_id"], "doi": cfg["zenodo"]["doi"],
                   "title": rec["metadata"]["title"], "publication_date": rec["metadata"]["publication_date"],
                   "license": rec["metadata"].get("license"), "files": []}
    plan = [(n, resolve(cfg["paths"]["raw_docs"])) for n in cfg["zenodo"]["doc_names"]]
    plan.append((cfg["zenodo"]["archive_name"], resolve(cfg["paths"]["raw_archive"])))
    for name, folder in plan:
        if name not in files:
            raise KeyError(f"{name!r} not in Zenodo record; available: {list(files)}")
        meta = files[name]
        algo, _, expected = meta["checksum"].partition(":")
        dest = folder / name
        url = meta["links"]["self"]
        status = "downloaded"
        if dest.exists() and dest.stat().st_size == meta["size"] and file_hash(dest, algo) == expected:
            status = "already_present_verified"
            log.info("%s already present and verified", name)
        else:
            if dest.exists():
                raise RuntimeError(f"{dest} exists but fails size/checksum; refusing to overwrite. Inspect it manually.")
            log.info("downloading %s (%.2f GB)", name, meta["size"] / 1e9)
            if meta["size"] > (256 << 20):
                download_segmented(url, dest, meta["size"], log)
            else:
                download_file(url, dest, meta["size"], log)
        actual = file_hash(dest, algo)
        ok = actual == expected
        log.info("%s: %s %s (expected %s)", name, algo, actual, expected)
        if not ok:
            raise RuntimeError(f"CHECKSUM MISMATCH for {name}: got {actual}, expected {expected}")
        dest.chmod(0o444)  # raw data is immutable
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        old_rec = prev.get(name)
        keep = old_rec is not None and old_rec.get("checksum_actual") == actual   # same bytes as when first recorded
        acquisition["files"].append({"name": name, "path": str(dest.relative_to(PROJECT_ROOT)), "size": meta["size"],
                                     "checksum_algo": algo, "checksum_expected": expected, "checksum_actual": actual,
                                     "verified": ok, "source_url": url,
                                     # provenance is preserved across re-runs: first-seen time and status are never overwritten
                                     "status": old_rec["status"] if keep else status,
                                     "acquired_utc": old_rec["acquired_utc"] if keep else now,
                                     **({"acquired_utc_source": old_rec["acquired_utc_source"]} if keep and "acquired_utc_source" in old_rec else {}),
                                     "last_verified_utc": now})
    write_json(acq_path, acquisition)
    log.info("all files verified; acquisition record written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
