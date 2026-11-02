#!/usr/bin/env bash
# Re-runs the whole data-preparation pipeline from the project root. Each stage is idempotent:
# download resumes / skips verified files, extraction is skipped once data/metadata/extraction.json exists,
# the derived tables (manifest, quality summaries, splits) are recomputed deterministically.
# `python -I` = isolated mode (ignores cwd/PYTHONPATH/user site), the safe way to run code that reads downloaded data.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 -I src/data/download.py      # resumable, 16-connection ranged download + MD5 verification
python3 -I src/data/extract.py       # path-checked, CRC-tested, read-only extraction into a fresh dir
python3 -I src/data/manifest.py      # decode every image; hashes, ICC, quality stats -> data/metadata/manifest_raw.csv
python3 -I src/data/quality.py       # duplicates, flags, tables and figures -> manifest.csv, reports/figures/
python3 -I src/data/splits.py        # patient-grouped folds + leakage assertions -> data/splits/, manifest.test_fold
python3 -I -m pytest -q -p no:cacheprovider tests
