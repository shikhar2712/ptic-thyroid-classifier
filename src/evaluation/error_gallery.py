"""Descriptive error gallery for the one-time test evaluation (reads reports/06_test_predictions.csv; scores nothing new).

Rows: patient 1's PTC-like images the E1 model missed (lowest scores); patient 1's PTC-like images it got right; typical PTC-like
images of other test patients; patient 1's benign images. Nothing here feeds back into any model or threshold.

  python -I src/evaluation/error_gallery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.common import PROJECT_ROOT, load_config, resolve  # noqa: E402


def main() -> int:
    cfg = load_config("data.yaml")
    root = resolve(cfg["paths"]["raw_images"])
    T = pd.read_csv(PROJECT_ROOT / "reports" / "06_test_predictions.csv")
    man = pd.read_csv(resolve(cfg["paths"]["metadata"]) / "manifest.csv", keep_default_na=False, na_values=[]).set_index("image_id")
    p1_ptc = T[(T.patient_id == 1) & (T.label == 1)].sort_values("E1_ens")
    rows = [("Patient 1, cancer-like, MISSED (E1 score < 0.5)", p1_ptc[p1_ptc.E1_ens < .5].head(4)),
            ("Patient 1, cancer-like, correct", p1_ptc[p1_ptc.E1_ens >= .5].tail(4)),
            ("Other patients, cancer-like (typical)", T[(T.patient_id != 1) & (T.label == 1)].sample(4, random_state=0)),
            ("Patient 1, benign (slide 1_1)", T[(T.patient_id == 1) & (T.label == 0)].sample(4, random_state=0))]
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "text.color": "#0b0b0b"})
    fig, axes = plt.subplots(4, 4, figsize=(15, 10.8))
    for r, (title, df) in enumerate(rows):
        for c in range(4):
            ax = axes[r, c]; ax.axis("off")
            if c >= len(df):
                continue
            row = df.iloc[c]
            with Image.open(root / man.loc[row.image_id, "path"]) as im:
                im.draft("RGB", (480, 264)); ax.imshow(im.convert("RGB"))
            ax.set_title(f"{row.image_id.split('/')[1]}  E1 {row.E1_ens:.2f} | E2 {row.E2_ens:.2f} | B2 {row.B2_colour_texture:.2f}", fontsize=8, loc="left")
        axes[r, 0].annotate(f"Row {r + 1}: {title}", xy=(0, 1.2), xycoords="axes fraction", ha="left", va="bottom", fontsize=10, fontweight="bold")
    fig.suptitle("Test errors in context. Scores = model's probability of 'cancer-like'; 0.5 is the cut-off. E1 = full resolution, E2 = 1.04 um/px, B2 = colour+texture rule",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.subplots_adjust(left=0.01, right=0.995, top=0.905, bottom=0.01, wspace=0.04, hspace=0.5)
    out = PROJECT_ROOT / "reports" / "figures" / "06_patient1_error_gallery.png"
    fig.savefig(out, dpi=110); plt.close(fig)
    print("written", out.relative_to(PROJECT_ROOT))
    b2 = ((p1_ptc.B2_colour_texture >= .5).astype(int) != 1).sum()
    print(f"patient 1 cancer-like images: {len(p1_ptc)} | missed by E1 {int((p1_ptc.E1_ens < .5).sum())}, E2 {int((p1_ptc.E2_ens < .5).sum())}, "
          f"B2 {int(b2)}, B1 {int((p1_ptc.B1_texture < .5).sum())}, B0 {int((p1_ptc.B0_colour < .5).sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
