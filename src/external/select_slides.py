"""Deterministic, label-free selection of external slides (rules: reports/08_external_test.md section 1). No image is opened.

  .venv/bin/python -I src/external/select_slides.py
"""
from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import PROJECT_ROOT  # noqa: E402

N_PER_SOURCE = 20
EXCLUDE_NOTES = re.compile(r"carcinoma|papillary|malignan|neoplas|tumou?r|thyroiditis|hashimoto", re.I)


def flat(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return ",".join(sorted({str(x) for x in v}))
    return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


def main() -> int:
    from idc_index import IDCClient
    c = IDCClient(); c.fetch_index("sm_index")
    sm = c.sm_index.copy()
    for col in sm.columns:
        if col != "SeriesInstanceUID":
            sm[col] = sm[col].map(flat)
    m = c.index[c.index.Modality == "SM"].merge(sm, on="SeriesInstanceUID", how="left")
    meta = PROJECT_ROOT / "data" / "external" / "metadata"

    # ---- TCGA-THCA: classic PTC, FFPE diagnostic slides at 40x, round-robin across submitting sites ----
    th = m[(m.collection_id == "tcga_thca") & m.ContainerIdentifier.str.contains(r"-DX\d*$", na=False)].copy()
    th = th[(th.embeddingMedium_CodeMeaning == "Paraffin wax") & (th.tissueFixative_CodeMeaning == "Formalin") & (th.min_PixelSpacing_2sf == "0.00025")]
    gdc = pd.read_csv(meta / "gdc_thca_dx_diagnoses.csv")
    th = th.merge(gdc, left_on="ContainerIdentifier", right_on="slide_id", how="inner")
    th = th[th.primary_diagnosis == "Papillary adenocarcinoma, NOS"]
    th["site"] = th.PatientID.str.split("-").str[1]
    th = th.sort_values(["site", "PatientID", "ContainerIdentifier"]).drop_duplicates("PatientID")      # one slide per patient
    th["rank_in_site"] = th.groupby("site").cumcount()
    tsel = th.sort_values(["rank_in_site", "site"]).head(N_PER_SOURCE)

    # ---- GTEx: thyroid, pathology note without neoplasm/thyroiditis terms, evenly spaced donors ----
    gt = m[(m.collection_id == "gtex") & m.primaryAnatomicStructure_CodeMeaning.fillna("").str.contains("Thyroid")].copy()
    att = pd.read_csv(meta / "GTEx_v8_SampleAttributesDS.txt", sep="\t", low_memory=False)
    att = att[att.SMTSD.fillna("").str.contains("Thyroid")].copy()
    att["donor"] = att.SAMPID.str.split("-").str[:2].str.join("-")
    notes = att.groupby("donor").SMPTHNTS.apply(lambda s: " | ".join(sorted({str(x) for x in s.dropna()})))
    gt["pathology_note"] = gt.PatientID.map(notes)
    gt = gt[gt.pathology_note.notna() & ~gt.pathology_note.str.contains(EXCLUDE_NOTES)].sort_values("PatientID").reset_index(drop=True)
    pick = sorted({int(round(i * (len(gt) - 1) / (N_PER_SOURCE - 1))) for i in range(N_PER_SOURCE)})
    gsel = gt.iloc[pick]

    cols = ["source", "slide_id", "PatientID", "SeriesInstanceUID", "crdc_series_uuid", "series_aws_url", "series_size_MB", "min_PixelSpacing_2sf", "label_note"]
    tsel = tsel.assign(source="TCGA-THCA", slide_id=tsel.ContainerIdentifier, label_note=tsel.primary_diagnosis)
    gsel = gsel.assign(source="GTEx", slide_id=gsel.ContainerIdentifier, label_note=gsel.pathology_note)
    S = pd.concat([tsel[cols], gsel[cols]], ignore_index=True)
    out = PROJECT_ROOT / "data" / "external" / "selection.csv"
    S.to_csv(out, index=False)
    print(f"TCGA candidates {len(th)} (sites {th.site.nunique()}) -> selected {len(tsel)} from {tsel.site.nunique()} sites | "
          f"GTEx candidates {len(gt)} after note filter -> selected {len(gsel)}")
    print("selection sha256:", hashlib.sha256(out.read_bytes()).hexdigest()[:16])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
