import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src.common import load_config
from src.data.metadata import label_from_folder, load_wsi_metadata, parse_filename, read_xlsx_rows, reconcile_with_xlsx

CFG = load_config("data.yaml")


def _mini_xlsx(path: Path, rows: list[list[str]]) -> None:
    """Write a minimal .xlsx with inline strings, including an empty-cell gap, to exercise cell-reference handling."""
    def cell(r: int, c: int, v: str) -> str:
        return f'<c r="{chr(65 + c)}{r}" t="inlineStr"><is><t>{v}</t></is></c>' if v != "" else ""
    body = "".join(f'<row r="{i}">' + "".join(cell(i, j, v) for j, v in enumerate(row)) + "</row>" for i, row in enumerate(rows, 1))
    sheet = ('<?xml version="1.0"?><worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
             f"<sheetData>{body}</sheetData></worksheet>")
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("xl/worksheets/sheet1.xml", sheet)


def test_parse_filename_ok_and_bad():
    r = CFG["filename_regex"]
    assert parse_filename("1_1 (1).jpg", r) == {"patient": 1, "wsi": 1, "patch": 1}
    assert parse_filename("23_4 (117).jpg", r) == {"patient": 23, "wsi": 4, "patch": 117}
    assert parse_filename("1_1.jpg", r) is None
    assert parse_filename("x_1 (1).jpg", r) is None
    assert parse_filename("1_1 (1).png", r) is None


def test_label_mapping_matches_archive_folders():
    assert label_from_folder("PTC-like", CFG["labels"]) == ("PTC-like", 1)
    assert label_from_folder("Non PTC-like", CFG["labels"]) == ("Non-PTC-like", 0)
    with pytest.raises(KeyError):
        label_from_folder("Non-PTC-like", CFG["labels"])  # the documentation spelling is NOT the folder name
    with pytest.raises(KeyError):
        label_from_folder("xlsx_map", CFG["labels"])


def test_xlsx_reader_handles_gaps_and_header(tmp_path):
    p = tmp_path / "m.xlsx"
    _mini_xlsx(p, [["Patient_ID", "WSI_ID", "Label"], ["1", "1", "Non-PTC"], ["21", "1", "Non-PTC", "only patch number 1"],
                   ["21", "1", "PTC"]])
    rows = read_xlsx_rows(p)
    assert rows[0] == ["Patient_ID", "WSI_ID", "Label"]
    df = load_wsi_metadata(p, CFG["labels"]["xlsx_map"])
    assert list(df.columns) == ["patient_id", "wsi_id", "label_name", "note", "xlsx_row"]
    assert df.loc[1, "note"] == "only patch number 1" and df.loc[2, "note"] == ""
    assert df.label_name.tolist() == ["Non-PTC-like", "Non-PTC-like", "PTC-like"]


def test_xlsx_rejects_unknown_label_and_header(tmp_path):
    p = tmp_path / "bad.xlsx"
    _mini_xlsx(p, [["Patient_ID", "WSI_ID", "Label"], ["1", "1", "Maybe"]])
    with pytest.raises(ValueError, match="unknown label"):
        load_wsi_metadata(p, CFG["labels"]["xlsx_map"])
    _mini_xlsx(p, [["A", "B", "C"], ["1", "1", "PTC"]])
    with pytest.raises(ValueError, match="header"):
        load_wsi_metadata(p, CFG["labels"]["xlsx_map"])


def test_reconcile_reports_without_fixing():
    files = pd.DataFrame({"patient_id": [1, 1, 2], "wsi_id": [1, 2, 1],
                          "class_name": ["PTC-like", "PTC-like", "Non-PTC-like"]})
    xl = pd.DataFrame({"patient_id": [1, 1, 3], "wsi_id": [1, 2, 1],
                       "label_name": ["PTC-like", "PTC-like", "Non-PTC-like"], "note": ["", "", ""]})
    rec = reconcile_with_xlsx(files, xl)
    assert rec["files_not_in_xlsx"] == [{"patient_id": 2, "wsi_id": 1, "class_name": "Non-PTC-like"}]
    assert rec["xlsx_not_in_files"] == [{"patient_id": 3, "wsi_id": 1, "class_name": "Non-PTC-like"}]


@pytest.mark.skipif(not (Path(__file__).parents[1] / "data/raw/docs/Data file 3-PTIC_metadata.xlsx").exists(),
                    reason="published workbook not downloaded")
def test_real_workbook_parses():
    df = load_wsi_metadata(Path(__file__).parents[1] / "data/raw/docs/Data file 3-PTIC_metadata.xlsx", CFG["labels"]["xlsx_map"])
    assert len(df) == 48 and df.patient_id.nunique() == 24
    assert set(df.label_name) == {"PTC-like", "Non-PTC-like"}
