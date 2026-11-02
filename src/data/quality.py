"""Data-quality analysis on the manifest: duplicates, flags, ICC/colour-profile audit, summaries and figures.

Nothing is deleted or modified. Questionable images are *flagged*; the proposed handling is written
into the report. Run:  python -I src/data/quality.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.common import get_logger, load_config, resolve, write_json  # noqa: E402

# Reference-palette categorical slots 1 and 2 (blue, orange); text uses ink tokens, never series colours.
C_PTC, C_NON = "#2a78d6", "#eb6834"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
CLASS_COLOR = {"PTC-like": C_PTC, "Non-PTC-like": C_NON}


def _style() -> None:
    plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                         "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                         "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left"})


def hamming_matrix(hexes: list[str]) -> np.ndarray:
    a = np.array([int(h, 16) for h in hexes], dtype=np.uint64)
    x = (a[:, None] ^ a[None, :]).astype(np.uint64)
    return np.unpackbits(x.view(np.uint8).reshape(len(a), len(a), 8), axis=2).sum(axis=2).astype(np.int16)


def add_flags(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Add duplicate indicators and quality flags. Returns (manifest, near-duplicate pairs, thresholds used)."""
    df = df.copy()
    ew, eh = cfg["expected"]["image_size_wh"]
    ok = df.readable.astype(bool)

    # exact duplicates by file hash
    grp = df[ok].groupby("sha256").image_id.transform("size")
    df["exact_duplicate"] = False
    df.loc[ok, "exact_duplicate"] = grp.values > 1
    df["dup_group"] = ""
    df.loc[df.exact_duplicate, "dup_group"] = df.loc[df.exact_duplicate, "sha256"].str[:12]

    # near duplicates by perceptual hashes (candidates for human review, not for removal)
    thr = {"phash_max": 6, "dhash_max": 6, "blur_lap_var_percentile": 1.0, "low_contrast_gray_std": 10.0,
           "mostly_background_white_frac": 0.5}
    idx = np.where(ok.values)[0]
    hp, hd = hamming_matrix(df.phash.iloc[idx].tolist()), hamming_matrix(df.dhash.iloc[idx].tolist())
    iu = np.triu_indices(len(idx), k=1)
    sel = (hp[iu] <= thr["phash_max"]) & (hd[iu] <= thr["dhash_max"])
    pairs = pd.DataFrame({"i": idx[iu[0][sel]], "j": idx[iu[1][sel]], "phash_dist": hp[iu][sel], "dhash_dist": hd[iu][sel]})
    cols = ["image_id", "patient_id", "wsi_id", "class_name", "sha256"]
    a = df.loc[pairs.i, cols].reset_index(drop=True).add_suffix("_a")
    b = df.loc[pairs.j, cols].reset_index(drop=True).add_suffix("_b")
    pairs = pd.concat([pairs[["phash_dist", "dhash_dist"]].reset_index(drop=True), a, b], axis=1)
    pairs["same_patient"] = pairs.patient_id_a == pairs.patient_id_b
    pairs["same_wsi"] = pairs.same_patient & (pairs.wsi_id_a == pairs.wsi_id_b)
    pairs["label_conflict"] = pairs.class_name_a != pairs.class_name_b
    pairs["byte_identical"] = pairs.sha256_a == pairs.sha256_b
    df["near_duplicate_candidate"] = df.image_id.isin(set(pairs.image_id_a) | set(pairs.image_id_b))

    # nearest-neighbour distance distribution (to justify the threshold)
    np.fill_diagonal(hp, 64)
    thr["nn_phash_dist_quantiles"] = {str(q): float(np.quantile(hp.min(axis=1), q)) for q in (0.0, 0.01, 0.05, 0.5)}

    # per-image flags
    lap_cut = np.percentile(df.loc[ok, "lap_var"], thr["blur_lap_var_percentile"])
    flags = pd.Series([[] for _ in range(len(df))], index=df.index)
    def add(mask: pd.Series, name: str) -> None:
        for i in df.index[mask.fillna(False)]:
            flags[i].append(name)
    add(~ok, "unreadable")
    add(ok & ((df.width != ew) | (df.height != eh)), "unexpected_dimensions")
    add(ok & (df["mode"] != "RGB"), "not_rgb")
    add(ok & (df.gray_std < thr["low_contrast_gray_std"]), "low_contrast")
    add(ok & (df.white_frac > thr["mostly_background_white_frac"]), "mostly_background")
    add(ok & (df.lap_var <= lap_cut), "lowest_1pct_sharpness")
    add(df.exact_duplicate, "exact_duplicate")
    add(df.near_duplicate_candidate, "near_duplicate_candidate")
    add(~df.filename_parse_ok, "filename_not_parsed")
    add(~df.in_xlsx.astype(bool), "not_in_xlsx")
    add(ok & df.nan_or_inf.astype(bool), "nan_or_inf")
    df["quality_flags"] = flags.map(";".join)
    thr["lap_var_cutoff"] = float(lap_cut)
    return df, pairs, thr


def patient_wsi_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    def piv(by: list[str]) -> pd.DataFrame:
        t = df.pivot_table(index=by, columns="class_name", values="image_id", aggfunc="count", fill_value=0)
        for c in ("PTC-like", "Non-PTC-like"):
            if c not in t:
                t[c] = 0
        t = t[["PTC-like", "Non-PTC-like"]]
        t["n_images"] = t.sum(axis=1)
        t["kind"] = np.where((t["PTC-like"] > 0) & (t["Non-PTC-like"] > 0), "mixed",
                             np.where(t["PTC-like"] > 0, "PTC-like only", "Non-PTC-like only"))
        return t.reset_index()
    p = piv(["patient_id"])
    p["n_wsi"] = p.patient_id.map(df.groupby("patient_id").wsi_id.nunique())
    return p, piv(["patient_id", "wsi_id"])


def fig_class_distribution(df: pd.DataFrame, p: pd.DataFrame, w: pd.DataFrame, out: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    rows = [("Images", [int((df.class_name == c).sum()) for c in CLASS_COLOR]),
            ("WSIs with ≥1 image", [int((w[c] > 0).sum()) for c in CLASS_COLOR]),
            ("Patients with ≥1 image", [int((p[c] > 0).sum()) for c in CLASS_COLOR])]
    for ax, (title, vals) in zip(axes, rows):
        bars = ax.bar(list(CLASS_COLOR), vals, color=list(CLASS_COLOR.values()), width=0.55)
        for b, v, hatch in zip(bars, vals, ("", "//")):
            b.set_hatch(hatch); b.set_edgecolor(SURFACE)
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v}", ha="center", va="bottom", color=INK, fontsize=10)
        ax.set_title(title); ax.set_ylim(0, max(vals) * 1.18); ax.grid(axis="x", visible=False)
    fig.suptitle("Images are nearly balanced (490 vs 516); independent patients and slides are fewer and less balanced",
                 x=0.01, ha="left", fontsize=11, color=INK, y=1.02)
    fig.tight_layout(); fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def fig_patient_distribution(p: pd.DataFrame, out: Path) -> None:
    _style()
    order = ["PTC-like only", "mixed", "Non-PTC-like only"]
    q = p.assign(o=p.kind.map({k: i for i, k in enumerate(order)})).sort_values(["o", "n_images"], ascending=[True, False]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 7.2))
    row, ys, headers = 0, [], []
    for kind in order:                      # one header row per group, then its bars; y axis is inverted (top to bottom)
        headers.append((row, kind, int((q.kind == kind).sum())))
        row += 1
        for _ in range(int((q.kind == kind).sum())):
            ys.append(row); row += 1
    ys = np.array(ys)
    ax.barh(ys, q["PTC-like"], color=C_PTC, height=0.62, label="PTC-like", edgecolor=SURFACE, linewidth=1.5)
    ax.barh(ys, q["Non-PTC-like"], left=q["PTC-like"], color=C_NON, height=0.62, label="Non-PTC-like",
            edgecolor=SURFACE, linewidth=1.5, hatch="//")
    for yi, (_, r) in zip(ys, q.iterrows()):
        ax.text(r.n_images + 1.5, yi, f"{int(r.n_images)}  ({int(r.n_wsi)} WSI)", va="center", fontsize=8.5, color=INK2)
    for yh, kind, n in headers:
        ax.text(0, yh, f"{kind}: {n} patients", va="center", ha="left", fontsize=9, fontweight="bold", color=INK)
    ax.set_yticks(ys); ax.set_yticklabels([f"patient {int(i)}" for i in q.patient_id], fontsize=8.5)
    ax.invert_yaxis(); ax.set_xlabel("images"); ax.grid(axis="y", visible=False); ax.set_xlim(0, q.n_images.max() * 1.18)
    ax.set_title("Images per patient, by class (grouped by patient type)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def fig_sample_grid(df: pd.DataFrame, images_root: Path, out: Path, n_per_class: int = 8, seed: int = 0) -> None:
    _style(); plt.rcParams["axes.grid"] = False
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, n_per_class, figsize=(2.6 * n_per_class, 4.2))
    for r, cls in enumerate(CLASS_COLOR):
        sub = df[(df.class_name == cls) & df.readable.astype(bool)]
        # one image from each of n distinct patients where possible
        pats = rng.permutation(sub.patient_id.unique())[:n_per_class]
        for c in range(n_per_class):
            ax = axes[r, c]; ax.axis("off")
            if c >= len(pats):
                continue
            row = sub[sub.patient_id == pats[c]].sample(1, random_state=int(rng.integers(1 << 30))).iloc[0]
            with Image.open(images_root / row.path) as im:
                im.draft("RGB", (480, 264)); ax.imshow(im.convert("RGB"))
            ax.set_title(f"{cls}\nP{row.patient_id}  WSI {row.wsi_id}  #{row.patch_no}", fontsize=8, color=INK, loc="left")
            for s in ax.spines.values():
                s.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02, wspace=0.06, hspace=0.45)
    fig.savefig(out, dpi=130); plt.close(fig)


def fig_colour_by_class(df: pd.DataFrame, out: Path) -> None:
    """Per-image mean R vs mean B coloured by class: a first look at whether colour alone separates classes."""
    _style()
    fig, ax = plt.subplots(figsize=(6.2, 5))
    ok = df[df.readable.astype(bool)]
    for cls, col in CLASS_COLOR.items():
        s = ok[ok.class_name == cls]
        ax.scatter(s.mean_r, s.mean_b, s=12, color=col, alpha=0.65, linewidths=0, label=f"{cls} (n={len(s)})")
    ax.set_xlabel("mean red (0-255)"); ax.set_ylabel("mean blue (0-255)")
    ax.set_title("Per-image mean colour: how much do the classes overlap?")
    ax.legend(frameon=False, loc="best"); fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)


def main() -> int:
    cfg = load_config("data.yaml")
    log = get_logger("quality", resolve(cfg["paths"]["logs"]) / "quality.log")
    meta_dir, rep_dir = resolve(cfg["paths"]["metadata"]), resolve(cfg["paths"]["reports"])
    fig_dir = rep_dir / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(meta_dir / "manifest_raw.csv", keep_default_na=False, na_values=[])
    for c in ("readable", "filename_parse_ok", "in_xlsx", "nan_or_inf"):
        df[c] = df[c].astype(str).str.lower().eq("true")
    for c in ("patient_id", "wsi_id", "patch_no"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df, pairs, thr = add_flags(df, cfg)
    df.to_csv(meta_dir / "manifest.csv", index=False)
    pairs.to_csv(meta_dir / "near_duplicate_candidates.csv", index=False)
    p, w = patient_wsi_tables(df[df.filename_parse_ok])
    p.to_csv(meta_dir / "patient_summary.csv", index=False); w.to_csv(meta_dir / "wsi_summary.csv", index=False)
    dims = df.groupby(["width", "height", "mode", "format"], dropna=False).agg(n=("image_id", "size"),
            min_mb=("file_size", lambda s: s.min() / 1e6), max_mb=("file_size", lambda s: s.max() / 1e6)).reset_index()
    dims.to_csv(meta_dir / "dimension_summary.csv", index=False)
    icc = df.groupby(["icc_sha256", "icc_len"]).agg(n_images=("image_id", "size"), n_patients=("patient_id", "nunique"),
            n_wsis=("wsi_key", "nunique"), n_ptc=("label", "sum")).reset_index()
    icc.to_csv(meta_dir / "icc_profile_summary.csv", index=False)

    fig_class_distribution(df, p, w, fig_dir / "class_distribution.png")
    fig_patient_distribution(p, fig_dir / "patient_distribution.png")
    fig_sample_grid(df, resolve(cfg["paths"]["raw_images"]), fig_dir / "sample_grid.png")
    fig_colour_by_class(df, fig_dir / "colour_by_class.png")

    ok = df.readable
    summary = {
        "n_files": int(len(df)), "n_readable": int(ok.sum()), "n_unreadable": int((~ok).sum()),
        "unreadable_examples": df.loc[~ok, ["image_id", "error"]].head(10).to_dict("records"),
        "class_counts": df.class_name.value_counts().to_dict(),
        "n_patients": int(df.patient_id.nunique()), "n_wsi_with_images": int(df.wsi_key.nunique()),
        "dimension_groups": dims.to_dict("records"),
        "exact_duplicate_files": int(df.exact_duplicate.sum()),
        "exact_duplicate_groups": int(df.loc[df.exact_duplicate, "dup_group"].nunique()),
        "near_duplicate_candidate_pairs": int(len(pairs)),
        "near_duplicate_pairs_same_wsi": int(pairs.same_wsi.sum()) if len(pairs) else 0,
        "near_duplicate_pairs_cross_patient": int((~pairs.same_patient).sum()) if len(pairs) else 0,
        "near_duplicate_pairs_label_conflict": int(pairs.label_conflict.sum()) if len(pairs) else 0,
        "flag_counts": pd.Series([f for s in df.quality_flags for f in s.split(";") if f]).value_counts().to_dict(),
        "icc_unique_profiles": int(df.icc_sha256.nunique()), "icc_len_values": sorted(df.icc_len.unique().tolist())[:10],
        "jpeg_q0_mean_unique": sorted(df.jpeg_q0_mean.round(3).unique().tolist())[:10],
        "jpeg_subsampling_unique": sorted(df.jpeg_subsampling.astype(str).unique().tolist()),
        "thresholds": thr,
        "colour_means_by_class": df.groupby("class_name")[["mean_r", "mean_g", "mean_b", "gray_std", "white_frac", "lap_var"]].mean().round(2).to_dict("index"),
    }
    write_json(meta_dir / "quality_summary.json", summary)
    log.info("quality summary written: %s", {k: summary[k] for k in ("n_files", "n_readable", "exact_duplicate_files", "near_duplicate_candidate_pairs", "icc_unique_profiles")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
