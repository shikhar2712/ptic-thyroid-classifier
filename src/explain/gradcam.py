"""Grad-CAM heat maps for the E3 fold ensemble (protocol: reports/11_gradcam.md section 1). Descriptive only.

Last-layer Grad-CAM without the final ReLU. Explained score: each 256-px square's logit difference z(PTC-like) - z(not PTC-like).
Layer: output of the last residual stage (192 x 16 x 16 per square; one cell = 16 px = 4.2 um). The head is global average pooling
-> linear, so this equals CAM: a square's local scores (Z x the Grad-CAM map, Z = 256 cells) average to its logit difference minus
the bias difference. The scorer averages logits over the squares, so the same holds for the whole image, per model.

  .venv/bin/python -I src/explain/gradcam.py                                  # reports/11_gradcam_* and reports/figures/11_*
  .venv/bin/python -I src/explain/gradcam.py --out-dir <scratch> --limit 20   # dry run: 20 images per set in the summaries
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):          # before torch starts: run-time changes can crash CPU ops (src.common)
    os.environ.setdefault(_var, os.environ.get("PTIC_TORCH_THREADS", "8"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from src.common import PROJECT_ROOT, env_snapshot, limit_cpu_threads, load_config, resolve, write_json  # noqa: E402
from src.data.dataset import _HED_ROWS, tile_images  # noqa: E402
from src.inference.gate import MODEL_MPP, rescale_to_model  # noqa: E402
from src.inference.predict import predict_uint8  # noqa: E402
from src.inference.predictor import PTICPredictor  # noqa: E402
from src.models.factory import build_model  # noqa: E402

SEED = 1100
WHITE = 220                                   # min(R, G, B) above this = white pixel (the manifest's white_frac)
MIN_WHITE_AREA = 0.05                         # white-area ratio only for images with at least this share of white pixels
HEMA = np.linalg.inv(_HED_ROWS)[:, 0]         # optical density -> hematoxylin (as in GPUPreprocess._hed_jitter)
MAX_REL_ERR = 1e-3                            # Grad-CAM vs closed-form CAM, and cells vs logit difference
MAX_SCORE_GAP = 0.02                          # fp32 score vs the stored score, per shown image
TCGA, GTEX = "TCGA-THCA", "GTEx"


# ----------------------------------------------------------------------------------------------- maps
def gradcam_tiles(net: torch.nn.Module, tiles: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalised squares [N,3,S,S] -> (local scores [N,h,w], logit differences [N]), fp32.

    alpha_k = mean over cells of d(z1 - z0)/dA_k, with A the last stage's output; local score = Z * sum_k alpha_k A_k (no ReLU).
    Raises if this differs from the closed-form CAM (w1 - w0) . A, or if the cells do not average to z1 - z0 minus the bias
    difference."""
    assert not net.training
    with torch.no_grad():
        feats = net.stages(net.stem(tiles.float()))
    A = feats.float().detach().requires_grad_(True)
    with torch.enable_grad():
        z = net.head(A)
        d = z[:, 1] - z[:, 0]
        (g,) = torch.autograd.grad(d.sum(), A)
    A, d = A.detach(), d.detach()
    local = A.shape[-2] * A.shape[-1] * torch.einsum("nk,nkhw->nhw", g.mean(dim=(2, 3)), A)
    w, b = net.head[-1].weight.detach().float(), net.head[-1].bias.detach().float()
    cam = torch.einsum("k,nkhw->nhw", w[1] - w[0], A)
    err_cam = float((local - cam).abs().max() / cam.abs().max().clamp_min(1e-6))
    err_sum = float((local.mean(dim=(1, 2)) + b[1] - b[0] - d).abs().max() / d.abs().max().clamp_min(1.0))
    if max(err_cam, err_sum) > MAX_REL_ERR:
        raise RuntimeError(f"Grad-CAM check failed: vs closed-form CAM {err_cam:.2e}, cells vs logit difference {err_sum:.2e}")
    return local, d


def stitch(local: torch.Tensor, rows: int, cols: int) -> np.ndarray:
    """Per-square maps [rows*cols, h, w] in tile_images' row-major order -> one map [rows*h, cols*w]."""
    n, h, w = local.shape
    assert n == rows * cols, (n, rows, cols)
    return local.reshape(rows, cols, h, w).permute(0, 2, 1, 3).reshape(rows * h, cols * w).cpu().numpy()


def image_maps(nets: list[tuple[torch.nn.Module, torch.nn.Module, int]], arr: np.ndarray,
               device: torch.device) -> tuple[np.ndarray, np.ndarray, int]:
    """uint8 HxWx3 at the model's pixel size -> (per-model local-score maps [M,h,w], per-model image logit differences [M],
    cell size in pixels). Same path as scoring: GPUPreprocess (evaluation mode) -> fixed non-overlapping squares."""
    x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None].to(device)
    maps, diffs, cell = [], [], 0
    for net, pre, tile in nets:
        tiles, _ = tile_images(x, pre, tile)
        local, d = gradcam_tiles(net, tiles)
        maps.append(stitch(local, arr.shape[0] // tile, arr.shape[1] // tile))
        diffs.append(float(d.mean()))                       # the scorer averages the logits over the squares
        cell = tile // local.shape[-1]
    return np.stack(maps), np.array(diffs), cell


def cell_stats(arr: np.ndarray, cell: int, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Per map cell, over the scored area: share of white pixels and mean hematoxylin optical density."""
    h, w = shape
    a = arr[: h * cell, : w * cell].astype(np.float32)
    white = (a.min(axis=2) > WHITE).astype(np.float32)
    hema = -np.log(np.clip(a / 255.0, 1 / 255, 1.0)) @ HEMA
    return (white.reshape(h, cell, w, cell).mean(axis=(1, 3)), hema.reshape(h, cell, w, cell).mean(axis=(1, 3)))


def spearman_rows(m: np.ndarray) -> np.ndarray:
    """Spearman correlation matrix between the rows of m [k, n]."""
    r = np.apply_along_axis(rankdata, 1, m.reshape(len(m), -1))
    r = (r - r.mean(axis=1, keepdims=True)) / r.std(axis=1, keepdims=True)
    return r @ r.T / r.shape[1]


def summary_stats(maps: np.ndarray, white: np.ndarray, hema: np.ndarray, cells_per_square: int) -> dict[str, float]:
    """Descriptive statistics of one image's maps (no model answer is used or returned)."""
    mag = np.abs(maps.mean(axis=0))
    area = float(white.mean())
    on_white = float((mag * white).sum() / mag.sum())
    ii, jj = np.indices(mag.shape) % cells_per_square
    ring = (ii == 0) | (ii == cells_per_square - 1) | (jj == 0) | (jj == cells_per_square - 1)
    pairs = spearman_rows(maps)[np.triu_indices(len(maps), 1)] if len(maps) > 1 else np.array([np.nan])
    return {"white_area": round(area, 4), "evidence_on_white": round(on_white, 4),
            "white_ratio": round(on_white / area, 4) if area >= MIN_WHITE_AREA else np.nan,
            "rho_hematoxylin": round(float(spearman_rows(np.stack([mag.ravel(), hema.ravel()]))[0, 1]), 4),
            "seed_agreement": round(float(np.median(pairs)), 4),
            "border_ratio": round(float(mag[ring].mean() / mag[~ring].mean()), 4)}


# ----------------------------------------------------------------------------------------------- selection (protocol table)
def select(man: pd.DataFrame, t6: pd.DataFrame, r8: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for label in (1, 0):                                                       # A: random PTIC examples, PTC-like first
        pats = sorted(set(man[(man.label == label) & (man.patient_id != 1)].patient_id))
        for p in rng.choice(pats, 6, replace=False):
            ids = sorted(man[(man.label == label) & (man.patient_id == p)].image_id)
            rows.append({"group": "A", "kind": "ptic", "image_id": ids[int(rng.integers(len(ids)))]})
    p1 = t6[(t6.patient_id == 1) & (t6.label == 1)].sort_values("E1_ens")      # B: rows 1-2 of the Report 6 error gallery
    for df in (p1[p1.E1_ens < .5].head(4), p1[p1.E1_ens >= .5].tail(4)):
        rows += [{"group": "B", "kind": "ptic", "image_id": i} for i in df.image_id]
    tc = r8[r8.source == TCGA]                                                 # C: outside regions
    hi = tc[tc.p_without_checks >= 0.8]
    assert len(hi) == 3 and hi.slide_id.nunique() == 1, hi
    rows += [{"group": "C", "kind": "outside", "source": TCGA, "slide_id": s, "region": int(g)} for s, g in zip(hi.slide_id, hi.region)]
    for src, pool in ((TCGA, tc[tc.slide_id != hi.slide_id.iloc[0]]), (GTEX, r8[r8.source == GTEX])):
        for s in rng.choice(sorted(set(pool.slide_id)), 3, replace=False):
            regs = sorted(pool[pool.slide_id == s].region)
            rows.append({"group": "C", "kind": "outside", "source": src, "slide_id": s, "region": int(regs[int(rng.integers(len(regs)))])})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------- figures
def draw(items: list[dict], title: str, note: str, L: float, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, Normalize

    cmap = LinearSegmentedColormap.from_list("evidence", ["#104281", "#3987e5", "#b7d3f6", "#f0efec", "#f5c4c3", "#e34948", "#8e1b1a"])
    unseen = np.array([0.55, 0.545, 0.53], np.float32)
    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "text.color": "#0b0b0b", "font.size": 9})
    nrow = (len(items) + 1) // 2
    fig, axes = plt.subplots(nrow, 4, figsize=(16, 2.45 * nrow + 1.5), squeeze=False)
    for k, ax in enumerate(axes.flat):
        ax.axis("off")
        it = items[k // 2] if k // 2 < len(items) else None
        if it is None:
            continue
        arr, ens, cell = it["arr"], it["map"], it["cell"]
        h, w = ens.shape[0] * cell, ens.shape[1] * cell
        if k % 2 == 0:                                                         # the scan, unseen strips dimmed
            img = arr.astype(np.float32) / 255
            img[h:], img[:, w:] = 0.45 * img[h:] + 0.55 * unseen, 0.45 * img[:, w:] + 0.55 * unseen
            ax.set_title(it["caption"], loc="left", fontsize=8.6)
        else:                                                                  # the map over a grey copy of the tissue
            gray = 0.6 + 0.4 * (arr.astype(np.float32) / 255) @ np.array([0.299, 0.587, 0.114], np.float32)
            col = cmap(np.clip(ens / L, -1, 1) * 0.5 + 0.5)[..., :3].astype(np.float32)
            col = np.repeat(np.repeat(col, cell, axis=0), cell, axis=1)        # one block per cell: no blending across squares
            img = np.broadcast_to(unseen, arr.shape).copy()
            img[:h, :w] = col * gray[:h, :w, None]
        small = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).resize((arr.shape[1] // 4, arr.shape[0] // 4), Image.BOX)
        ax.imshow(np.asarray(small), interpolation="nearest")
    H = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.12 / H, ha="left", va="top", fontsize=12.5, fontweight="bold")
    fig.text(0.01, 0.1 / H, "\n".join(textwrap.fill(p, 100) for p in note.split("\n")), fontsize=8.6, va="bottom", color="#52514e")
    cax = fig.add_axes([0.56, 0.72 / H, 0.38, 0.12 / H])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-L, L), cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([-L, 0, L])
    cb.set_ticklabels([f"-{L:.1f}\nevidence for 'not PTC-like'", "0\nno evidence", f"+{L:.1f}\nevidence for 'PTC-like'"])
    cb.ax.tick_params(labelsize=8.6)
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 0.75 / H, bottom=1.25 / H, wspace=0.03, hspace=0.24)
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- run
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "reports"))
    ap.add_argument("--limit", type=int, default=0, help="dry run: only the first N images of each set in the summaries")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    limit_cpu_threads()
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = False     # full fp32 (protocol)
    out = Path(a.out_dir)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg = load_config("data.yaml")
    root = resolve(cfg["paths"]["raw_images"])
    man = pd.read_csv(resolve(cfg["paths"]["metadata"]) / "manifest.csv", keep_default_na=False, na_values=[])
    t6 = pd.read_csv(PROJECT_ROOT / "reports" / "06_test_predictions.csv")
    r8 = pd.read_csv(PROJECT_ROOT / "reports" / "08_external_regions.csv.gz")
    r8 = r8[r8.model == "E3"]
    reg_root = PROJECT_ROOT / "data" / "external" / "regions"
    regions = pd.read_csv(reg_root / "manifest.csv")
    sel = select(man, t6, r8, np.random.default_rng(SEED)).astype({"region": "Int64"})
    sel.to_csv(out / "11_gradcam_selection.csv", index=False)
    print(f"selection fixed: {len(sel)} images ({sel.group.value_counts().sort_index().to_dict()})", flush=True)

    assert (PROJECT_ROOT / "artifacts" / "deploy" / "model" / "CURRENT").read_text().strip() == "E3_fold_ensemble"
    pred = PTICPredictor(model_name="E3_fold_ensemble", device=a.device)
    mdir = pred.deploy_dir / "model" / "E3_fold_ensemble"
    ck = torch.load(mdir / pred.manifest["members"][0]["file"], map_location="cpu", weights_only=False)
    torch.manual_seed(SEED)
    rand_net = build_model(ck["model_cfg"]).to(pred.device).to(memory_format=torch.channels_last).eval()   # sanity check only
    mi = man.set_index("image_id")
    ri = regions.set_index(["source", "slide_id", "region"])

    def load(kind: str, key) -> tuple[np.ndarray, list[int]]:
        if kind == "ptic":
            with Image.open(root / key) as im:
                arr = np.array(im.convert("RGB"))                          # embedded ICC ignored, as in training and scoring
            use = [i for i, f in enumerate(pred.member_folds) if f == int(mi.loc[key, "test_fold"])]
            assert len(use) == 3
            return arr, use
        r = ri.loc[key]
        assert abs(r.mpp_x / r.mpp_y - 1) < 0.01
        with Image.open(reg_root / r.file) as im:
            return rescale_to_model(np.array(im.convert("RGB")), float(r.mpp_x), MODEL_MPP), list(range(len(pred.members)))

    # --- the shown images: maps, score checks, sanity check
    shown, figs = [], {"A": [], "B": [], "C": []}
    for n, r in enumerate(sel.itertuples(index=False), 1):
        key = r.image_id if r.kind == "ptic" else (r.source, r.slide_id, int(r.region))
        arr, use = load(r.kind, key)
        nets = [pred.members[i] for i in use]
        maps, diffs, cell = image_maps(nets, arr, pred.device)
        p = float(np.mean(1 / (1 + np.exp(-diffs))))
        x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None].to(pred.device)
        if r.kind == "ptic":   # the demo's fair-mode path: same 3 models, bf16 on the GPU
            stored = float(np.mean([float(torch.softmax(predict_uint8(net, x, pre, t, pred.amp), 1)[0, 1]) for net, pre, t in nets]))
            row = mi.loc[key]
            info = {"image_id": key, "label": int(row.label), "patient_id": int(row.patient_id), "slide": row.wsi_key,
                    "test_fold": int(row.test_fold)}
            if r.group == "B":
                info["E1_test_score_report6"] = round(float(t6.set_index("image_id").loc[key, "E1_ens"]), 4)
        else:
            e = r8[(r8.source == r.source) & (r8.slide_id == r.slide_id) & (r8.region == r.region)].iloc[0]
            stored = float(e.p_without_checks)
            info = {"source": r.source, "slide_id": r.slide_id, "region": int(r.region), "checks_status_report8": e.status}
        if abs(p - stored) > MAX_SCORE_GAP:
            raise SystemExit(f"STOP: fp32 score {p:.4f} differs from the stored score {stored:.4f} for {key}")
        _, pre0, tile0 = nets[0]
        tiles, _ = tile_images(x, pre0, tile0)
        rand_map = stitch(gradcam_tiles(rand_net, tiles)[0], arr.shape[0] // tile0, arr.shape[1] // tile0)
        ens = maps.mean(axis=0)
        white, hema = cell_stats(arr, cell, ens.shape)
        answer = "PTC-like" if p >= pred.band[1] else ("not PTC-like" if p <= pred.band[0] else "uncertain")
        shown.append({"id": f"{r.group}{len(figs[r.group]) + 1}", "group": r.group, **info, "models": len(use),
                      "e3_score_fp32": round(p, 4), "stored_score": round(stored, 4), "e3_answer": answer,
                      "model_scores": [round(float(v), 4) for v in 1 / (1 + np.exp(-diffs))],
                      "mean_local_score": round(float(ens.mean()), 4),
                      "rho_abs_vs_random_network": round(float(spearman_rows(np.stack([np.abs(ens).ravel(), np.abs(rand_map).ravel()]))[0, 1]), 4),
                      **summary_stats(maps, white, hema, tile0 // cell)})
        figs[r.group].append({"arr": arr, "map": ens, "cell": cell, "info": shown[-1]})
        print(f"[{n}/{len(sel)}] {shown[-1]['id']} p={p:.4f} stored={stored:.4f} rho_random={shown[-1]['rho_abs_vs_random_network']}",
              flush=True)
    L = float(np.percentile(np.abs(np.concatenate([f["map"].ravel() for g in figs.values() for f in g])), 99))
    np.savez_compressed(out / "11_gradcam_shown_maps.npz", **{s["id"]: f["map"] for s, f in zip(shown, [f for g in "ABC" for f in figs[g]])})

    # --- summaries over all images (no model answers stored)
    items = [("ptic", i) for i in man.image_id] + [("outside", k) for k in zip(regions.source, regions.slide_id, regions.region)]
    if a.limit:
        items = items[: a.limit] + [it for it in items if it[0] == "outside"][: a.limit]
    per = []
    for n, (kind, key) in enumerate(items, 1):
        arr, use = load(kind, key)
        maps, _, cell = image_maps([pred.members[i] for i in use], arr, pred.device)
        white, hema = cell_stats(arr, cell, maps.shape[1:])
        group = (("PTIC PTC-like" if int(mi.loc[key, "label"]) == 1 else "PTIC not PTC-like") if kind == "ptic"
                 else ("TCGA-THCA (cancer slides)" if key[0] == TCGA else "GTEx (normal thyroid)"))
        per.append({"group": group, "image": key if kind == "ptic" else f"{key[0]}/{key[1]}/region_{key[2]:02d}",
                    "patient_id": int(mi.loc[key, "patient_id"]) if kind == "ptic" else None, "models": len(use),
                    **summary_stats(maps, white, hema, pred.members[use[0]][2] // cell)})
        if n % 200 == 0:
            print(f"summaries: {n}/{len(items)} ({time.time() - t0:.0f} s)", flush=True)
    P = pd.DataFrame(per)
    P.to_csv(out / "11_gradcam_per_image.csv.gz", index=False)

    def agg(g: pd.DataFrame) -> dict:
        w = g.white_ratio.dropna()
        q = lambda s: [round(float(v), 3) for v in s.quantile([.25, .5, .75])]  # noqa: E731
        return {"images": len(g), "patients": int(g.patient_id.nunique()) if g.patient_id.notna().any() else None,
                "white_ratio_images": len(w), "white_ratio_q25_median_q75": q(w), "share_white_ratio_ge_1": round(float((w >= 1).mean()), 3),
                "white_area_median": round(float(g.white_area.median()), 3), "rho_hematoxylin_q25_median_q75": q(g.rho_hematoxylin),
                "seed_agreement_q25_median_q75": q(g.seed_agreement), "border_ratio_q25_median_q75": q(g.border_ratio)}

    S = pd.DataFrame(shown)
    summary = {"note": "Descriptive only (Report 11). No model answers are aggregated; nothing here changes the model, checks or thresholds.",
               "colour_scale_limit_L": round(L, 4), "cell_px": figs["A"][0]["cell"], "seed": SEED,
               "groups": {k: agg(g) for k, g in P.groupby("group")},
               "sanity_check_rho_abs_vs_random_network": {"median": round(float(S.rho_abs_vs_random_network.median()), 4),
                                                          "min": round(float(S.rho_abs_vs_random_network.min()), 4),
                                                          "max": round(float(S.rho_abs_vs_random_network.max()), 4)},
               "max_gap_fp32_vs_stored_score": round(float((S.e3_score_fp32 - S.stored_score).abs().max()), 4),
               "limit": a.limit, "device": str(pred.device), "seconds": round(time.time() - t0, 1), "env": env_snapshot()}
    write_json(out / "11_gradcam_summary.json", summary)
    write_json(out / "11_gradcam_shown.json", {"note": "E3 scores of PTIC images are descriptive only: E3 never had its own final exam. "
                                                       "Outside regions: model alone, checks off (Report 8).", "images": shown})

    # --- figures (one fixed colour scale for all)
    for it in figs["A"]:
        s = it["info"]
        it["caption"] = (f"{s['id']} · {'PTC-like' if s['label'] else 'not PTC-like'} (label) · patient {s['patient_id']}\n"
                         f"E3 {s['e3_score_fp32']:.2f} → {s['e3_answer']}")
    for it in figs["B"]:
        s = it["info"]
        it["caption"] = (f"{s['id']} · cancer-like (label) · patient 1, slide {s['slide']}\n"
                         f"E1 {s['E1_test_score_report6']:.2f} (Report 6) · E3 {s['e3_score_fp32']:.2f} → {s['e3_answer']}")
    for it in figs["C"]:
        s = it["info"]
        kind, short = ("cancer slide", s["slide_id"].split("-")[:3]) if s["source"] == TCGA else ("normal thyroid", s["slide_id"].split("-")[:2])
        it["caption"] = (f"{s['id']} · {s['source']} {kind} {'-'.join(short)}, region {s['region']}\n"
                         f"E3 alone {s['e3_score_fp32']:.2f} · safety checks: {'refused' if s['checks_status_report8'] == 'refused' else 'accepted'}")
    common = ("Left: the scan. Right: where the model found its evidence, over a grey copy of the tissue. One colour block = 4.2 µm; "
              "the same colour scale on every picture. Dark grey strips: never looked at by the model.")
    fig_dir = out / "figures"
    draw(figs["A"], "Random examples: where the model (E3) finds its evidence",
         common + "\nE3 scores of these images are descriptive only (E3 had no final exam). Images chosen at random before any map was made.",
         L, fig_dir / "11_gradcam_random_examples.png")
    draw(figs["B"], "Patient 1: the images E1 missed (top two rows) and got right (bottom two rows)",
         common + "\nSame images as rows 1-2 of the Report 6 error gallery. E3 scores are descriptive only.", L, fig_dir / "11_gradcam_patient1.png")
    draw(figs["C"], "Outside hospitals (Report 8): where the model looks on other hospitals' scans",
         common + "\nModel alone with the safety checks off (information only); the checks refused these regions except where marked accepted.",
         L, fig_dir / "11_gradcam_outside.png")
    print(f"done in {time.time() - t0:.0f} s; L = {L:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
