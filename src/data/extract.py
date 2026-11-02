"""Safe extraction of the PTIC archive into a *new, empty* directory.

Guards: refuses a non-empty target, rejects absolute / '..' / drive-letter member names and symlinks,
verifies the resolved destination stays inside the target, checks per-member CRC (`testzip`) before
extracting, and finally makes the tree read-only (raw data is immutable).
Run:  python -I src/data/extract.py
"""
from __future__ import annotations

import stat
import sys
import zipfile
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

from src.common import get_logger, load_config, resolve, write_json  # noqa: E402


def check_member_names(names: list[str], target: Path) -> list[str]:
    """Return a list of problems (empty if every member is safe)."""
    problems: list[str] = []
    target = target.resolve()
    for n in names:
        p = PurePosixPath(n)
        if p.is_absolute() or ".." in p.parts or (len(n) > 1 and n[1] == ":") or "\\" in n:
            problems.append(f"unsafe name: {n!r}")
            continue
        if not (target / n).resolve().is_relative_to(target):
            problems.append(f"escapes target: {n!r}")
    return problems


def safe_extract(zip_path: Path, target: Path, log=None) -> dict:
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"{target} is not empty; refusing to extract into it")
    with zipfile.ZipFile(zip_path) as z:
        infos = z.infolist()
        problems = check_member_names([i.filename for i in infos], target)
        problems += [f"symlink: {i.filename!r}" for i in infos if stat.S_ISLNK(i.external_attr >> 16)]
        if problems:
            raise ValueError("refusing to extract:\n" + "\n".join(problems[:20]))
        if log:
            log.info("%d members; running CRC test (reads all %0.1f GB)", len(infos), sum(i.compress_size for i in infos) / 1e9)
        bad = z.testzip()
        if bad is not None:
            raise ValueError(f"CRC failure in member {bad!r}")
        target.mkdir(parents=True, exist_ok=True)
        for i in infos:
            z.extract(i, target)
    for p in sorted(target.rglob("*"), reverse=True):  # files first, then dirs
        p.chmod(0o444 if p.is_file() else 0o555)
    target.chmod(0o555)
    files = [p for p in target.rglob("*") if p.is_file()]
    return {"members": len(infos), "files_extracted": len(files),
            "bytes_extracted": sum(p.stat().st_size for p in files)}


def main() -> int:
    cfg = load_config("data.yaml")
    log = get_logger("extract", resolve(cfg["paths"]["logs"]) / "extract.log")
    zip_path = resolve(cfg["paths"]["raw_archive"]) / cfg["zenodo"]["archive_name"]
    target = resolve(cfg["paths"]["raw_images"])
    marker = resolve(cfg["paths"]["metadata"]) / "extraction.json"
    if marker.exists():
        log.info("already extracted (%s); nothing to do", marker)
        return 0
    stats = safe_extract(zip_path, target, log)
    write_json(marker, {"archive": str(zip_path), "target": str(target), **stats})
    log.info("extracted: %s", stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
