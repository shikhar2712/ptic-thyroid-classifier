"""Filename parsing, label mapping and the stdlib-only reader for the published metadata workbook.

`openpyxl` is not installed (and installing into the NGC system site-packages needs approval), so the
.xlsx is read with `zipfile` + `xml.etree`. Only cell values are needed.
"""
from __future__ import annotations

import re
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd

_NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_T = "{%s}t" % _NS["m"]


def _col_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + (ord(ch) - 64)
    return n - 1


def read_xlsx_rows(path: Path, sheet: int = 1) -> list[list[str]]:
    """Return rows of one worksheet as lists of strings ('' for empty cells), honouring cell references."""
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            shared = ["".join(t.text or "" for t in si.iter(_T)) for si in root.findall("m:si", _NS)]
        sheet_xml = ET.fromstring(z.read(f"xl/worksheets/sheet{sheet}.xml"))
    rows: list[list[str]] = []
    for row in sheet_xml.find("m:sheetData", _NS):
        cells: dict[int, str] = {}
        for c in row:
            v = c.find("m:v", _NS)
            if c.get("t") == "s" and v is not None:
                val = shared[int(v.text)]
            elif c.get("t") == "inlineStr":
                val = "".join(t.text or "" for t in c.iter(_T))
            else:
                val = v.text if v is not None and v.text is not None else ""
            cells[_col_index(c.get("r"))] = val
        width = max(cells) + 1 if cells else 0
        rows.append([cells.get(i, "") for i in range(width)])
    return rows


def load_wsi_metadata(xlsx_path: Path, label_map: dict[str, str]) -> pd.DataFrame:
    """Parse `PTIC_metadata.xlsx` into columns patient_id, wsi_id, label_name, note, xlsx_row."""
    rows = read_xlsx_rows(xlsx_path)
    header, body = rows[0], rows[1:]
    expected = ["Patient_ID", "WSI_ID", "Label"]
    if header[:3] != expected:
        raise ValueError(f"unexpected workbook header {header!r}; expected it to start with {expected!r}")
    out = []
    for i, r in enumerate(body, start=2):
        r = r + [""] * (4 - len(r))
        if not any(r[:3]):
            continue
        if r[2] not in label_map:
            raise ValueError(f"row {i}: unknown label {r[2]!r}; known: {sorted(label_map)}")
        out.append({"patient_id": int(r[0]), "wsi_id": int(r[1]), "label_name": label_map[r[2]],
                    "note": r[3], "xlsx_row": i})
    return pd.DataFrame(out)


def parse_filename(name: str, regex: str) -> dict[str, int] | None:
    m = re.match(regex, name)
    if not m:
        return None
    return {k: int(v) for k, v in m.groupdict().items()}


def label_from_folder(folder: str, cfg_labels: dict[str, Any]) -> tuple[str, int]:
    """Map an archive folder name (e.g. 'Non PTC-like') to (class name, numeric label)."""
    entry = cfg_labels.get(folder)
    if entry is None or folder == "xlsx_map":
        raise KeyError(f"unknown class folder {folder!r}")
    return entry["name"], int(entry["id"])


def reconcile_with_xlsx(manifest: pd.DataFrame, meta: pd.DataFrame) -> dict[str, Any]:
    """Compare (patient, wsi, class) triples seen in files against the workbook. Report only; never 'fixes'."""
    files = manifest.groupby(["patient_id", "wsi_id", "class_name"]).size().rename("n_files").reset_index()
    xl = meta.drop_duplicates(["patient_id", "wsi_id", "label_name"]).rename(columns={"label_name": "class_name"})
    merged = files.merge(xl[["patient_id", "wsi_id", "class_name"]], how="outer",
                         on=["patient_id", "wsi_id", "class_name"], indicator=True)
    dup_keys = meta[meta.duplicated(["patient_id", "wsi_id"], keep=False)]
    return {
        "files_not_in_xlsx": merged[merged._merge == "left_only"][["patient_id", "wsi_id", "class_name"]].to_dict("records"),
        "xlsx_not_in_files": merged[merged._merge == "right_only"][["patient_id", "wsi_id", "class_name"]].to_dict("records"),
        "xlsx_rows": int(len(meta)),
        "xlsx_unique_patient_wsi": int(meta.groupby(["patient_id", "wsi_id"]).ngroups),
        "xlsx_duplicate_patient_wsi_rows": dup_keys.to_dict("records"),
        "file_patient_wsi_pairs": int(manifest.groupby(["patient_id", "wsi_id"]).ngroups),
        "file_patients": int(manifest.patient_id.nunique()),
    }
