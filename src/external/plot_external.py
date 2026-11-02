"""Figure for Report 8: what the deployed demo (E3 + safety checks v3) did with each external slide's 10 regions.

  .venv/bin/python -I src/external/plot_external.py      # reads reports/08_external_slides.csv -> reports/figures/08_external_per_slide.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import PROJECT_ROOT  # noqa: E402

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
SEGMENTS = (("not_ptc_like", "not PTC-like", "#2a78d6", None), ("ptc_like", "PTC-like", "#eb6834", None),
            ("uncertain", "uncertain (0.2-0.8)", "#1baf7a", None), ("refused", "refused by the safety checks", "#9a9993", "///"))
PANELS = (("TCGA-THCA", "TCGA: thyroid-cancer slides (classic PTC), 20 US hospitals"),
          ("GTEx", "GTEx: normal thyroid, 20 donors"))


def main() -> int:
    S = pd.read_csv(PROJECT_ROOT / "reports" / "08_external_slides.csv")
    S["refused"] = S.regions - S.accepted
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), facecolor=SURF, sharex=True)
    for ax, (src, title) in zip(axes, PANELS):
        g = S[S.source == src].sort_values("slide_id", ascending=False).reset_index(drop=True)
        left = pd.Series(0.0, index=g.index)
        for col, label, color, hatch in SEGMENTS:
            ax.barh(g.index, g[col], left=left, height=0.72, color=color, hatch=hatch, edgecolor=SURF, linewidth=1.5, label=label)
            left += g[col]
        ax.set_yticks(g.index, [s.replace("-01Z-00-DX1", "") for s in g.slide_id], fontsize=8)
        ax.set_xlim(0, 10); ax.set_xticks(range(0, 11, 2))
        ax.set_xlabel("regions per slide (10 random tissue regions, each 498 x 274 um)")
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_facecolor(SURF); ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", ncol=4, frameon=False, bbox_to_anchor=(0.01, 0.955))
    fig.suptitle("External test (one shot): what the deployed demo (E3 + safety checks v3) said about each slide's regions",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = PROJECT_ROOT / "reports" / "figures" / "08_external_per_slide.png"
    fig.savefig(out, dpi=150, facecolor=SURF)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
