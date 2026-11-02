"""Input safety checks ("gate") for the internal demo. Deliberately INDEPENDENT of the PTC classifier, so swapping the classifier
needs no re-calibration.

Checks (v2), in order: image readable -> scale known (metadata or declared) -> magnification high enough -> big enough after
rescaling to the model's 0.26 um/px -> enough tissue (not mostly glass) -> no large dark/black areas (viewer bars) -> colour present
(not grayscale) -> tissue colours typical of H&E -> white balance neutral (glass not tinted) -> not noisy -> in focus -> overall
appearance like the training scans (frozen ImageNet ResNet-18 features, Mahalanobis distance).
v1 used one generic colour distance, which refused harmless brightness/saturation changes; v2 targets what the classifier is
actually sensitive to; v3 measures sharpness at half resolution so heavy blur can no longer pass (Report 7 section 3).
Thresholds come from patient-grouped cross-fitting on the dataset (src/inference/calibrate_gate.py), never hand-tuned on the
images used to report acceptance rates.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.common import PROJECT_ROOT

MODEL_MPP = 0.26                          # the dataset's pixel size (Leica GT450 at 40x)
MAGNIFICATION_TO_MPP = {"40x": 0.26, "20x": 0.52, "10x": 1.04, "5x": 2.08}


# --------------------------------------------------------------------------------------------- reading + scale
def read_image(src: str | Path | Image.Image | np.ndarray) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Return (uint8 HxWx3 RGB array or None, info). Never raises on bad input; errors go into info['error']."""
    info: dict[str, Any] = {}
    try:
        if isinstance(src, np.ndarray):
            im = Image.fromarray(src)
        elif isinstance(src, Image.Image):
            im = src
        else:
            im = Image.open(src)
        info.update(format=im.format, mode=im.mode, width=im.width, height=im.height, metadata_mpp=_mpp_from_metadata(im))
        if im.mode in ("L", "LA", "1", "I", "I;16", "F"):
            info["grayscale_file"] = True
        im = im.convert("RGB")                           # drops alpha (screenshots) and palettes
        return np.array(im), info                     # writable copy
    except Exception as exc:                             # unreadable / not an image
        info["error"] = f"{type(exc).__name__}: {exc}"
        return None, info


def _mpp_from_metadata(im: Image.Image) -> float | None:
    """Micrometres per pixel from TIFF/SVS metadata if present (Aperio 'MPP = 0.2630', or X resolution in cm/inch)."""
    try:
        tags = getattr(im, "tag_v2", None)
        if tags is None:
            return None
        desc = str(tags.get(270, ""))                      # ImageDescription
        m = re.search(r"MPP\s*=\s*([0-9.]+)", desc)
        if m:
            return float(m.group(1))
        xres, unit = tags.get(282), tags.get(296, 2)       # XResolution (pixels per unit), ResolutionUnit (2 inch, 3 cm)
        if xres:
            ppu = float(xres)
            if ppu > 1000:                                 # ignore screen-style 72/96/300 dpi values; only microscope-scale densities
                return (25400.0 if unit == 2 else 10000.0) / ppu
    except Exception:
        return None
    return None


def resolve_scale(info: dict[str, Any], mpp: float | None, magnification: str | None) -> tuple[float | None, str]:
    if mpp is not None:
        return float(mpp), "declared_mpp"
    if magnification is not None:
        key = str(magnification).lower().replace(" ", "")
        key = key if key.endswith("x") else key + "x"
        if key in MAGNIFICATION_TO_MPP:
            return MAGNIFICATION_TO_MPP[key], f"declared_magnification_{key}"
        return None, f"unknown_magnification_{magnification}"
    if info.get("metadata_mpp"):
        return float(info["metadata_mpp"]), "file_metadata"
    return None, "unknown"


def rescale_to_model(arr: np.ndarray, src_mpp: float, dst_mpp: float = MODEL_MPP) -> np.ndarray:
    f = src_mpp / dst_mpp
    if abs(f - 1) < 0.02:
        return arr
    h, w = arr.shape[:2]
    size = (max(1, round(w * f)), max(1, round(h * f)))
    return np.array(Image.fromarray(arr).resize(size, Image.BOX if f < 1 else Image.BICUBIC))   # writable copy


# --------------------------------------------------------------------------------------------- image statistics
def image_stats(arr: np.ndarray | torch.Tensor, device: str | torch.device = "cpu") -> dict[str, float]:
    """Cheap statistics of the rescaled image (uint8 HxWx3, numpy or torch). Same code on CPU (deployment) and GPU (calibration).

    background = bright AND neutral pixels (glass); dark = near-black (viewer bars, borders); tissue = the rest.
    white_point_* = colour of the brightest 2% of pixels (normally the glass), relative to their luminance: a colour cast shows up
    here, while brightness/contrast/gamma/saturation changes leave the glass neutral. he_hue_share = share of coloured tissue pixels
    whose hue is in the H&E purple-pink range. glass_noise = high-pass residual std on glass pixels (independent of tissue type)."""
    x = (torch.as_tensor(arr) if not isinstance(arr, torch.Tensor) else arr).to(device).float()
    mn, mx = x.min(dim=2).values, x.max(dim=2).values
    background = (mn > 215) & ((mx - mn) < 20)
    dark = mx < 40
    tissue = ~background & ~dark
    out = {"background_fraction": float(background.float().mean()), "dark_fraction": float(dark.float().mean()),
           "tissue_fraction": float(tissue.float().mean())}
    t = x[tissue] if int(tissue.sum()) > 100 else x.reshape(-1, 3)
    tmax, tmin = t.max(dim=1).values, t.min(dim=1).values
    sat = (tmax - tmin) / tmax.clamp_min(1.0)
    out["saturation_mean"] = float(sat.mean())
    # hue (degrees) of clearly coloured tissue pixels; H&E sits in the purple-pink band
    col = t[sat > 0.05]
    if len(col) > 100:
        r, g, b_ = col[:, 0], col[:, 1], col[:, 2]
        hue = torch.rad2deg(torch.atan2(math.sqrt(3) * (g - b_), 2 * r - g - b_)) % 360
        out["he_hue_share"] = float(((hue >= 240) | (hue <= 20)).float().mean())
    else:
        out["he_hue_share"] = float("nan")
    # white point from the brightest 2% of pixels (only meaningful if they are bright, i.e. glass is present)
    lum = x @ torch.tensor([0.299, 0.587, 0.114], device=x.device)
    flat_lum, flat = lum.reshape(-1), x.reshape(-1, 3)
    k = max(50, int(0.02 * flat_lum.numel()))
    top = flat[torch.topk(flat_lum, k).indices]
    L = top.mean(dim=0)
    lm = float(L.mean())
    if lm >= 170:                                         # brightest pixels bright enough to be glass
        out["white_point_rb"] = float((L[0] - L[2]) / lm)
        out["white_point_g"] = float((L[1] - (L[0] + L[2]) / 2) / lm)
    else:
        out["white_point_rb"] = out["white_point_g"] = float("nan")
    # noise on glass: residual after a 3x3 box filter, only where >0.5% of the image is glass
    if float(background.float().mean()) > 0.005:
        l4 = lum[None, None]
        box = F.avg_pool2d(F.pad(l4, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
        res = (l4 - box)[0, 0][background]
        out["glass_noise"] = float(res.std())
    else:
        out["glass_noise"] = float("nan")
    # v3: sharpness is measured on the 2x box-downsampled image. At full resolution 8-bit rounding noise makes heavily blurred
    # images look sharp again (measured: 5 px blur scored like a clean image); the 2x measure falls steadily with blur.
    x2 = F.avg_pool2d(x.permute(2, 0, 1)[None], 2)[0].permute(1, 2, 0)
    out["focus_ratio"] = focus_ratio(x2)
    return out


def focus_ratio(x: torch.Tensor) -> float:
    """Re-blur sharpness: share of fine-detail (Laplacian) energy that a 1-px Gaussian blur removes. A sharp image loses most of
    it (ratio near 1); an already blurred image loses little. A relative measure, so less tied to how much texture the tissue has
    than the raw Laplacian variance (which tracks the class)."""
    g = (x.float() @ torch.tensor([0.299, 0.587, 0.114], device=x.device))[None, None]
    lap = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=x.device)[None, None]
    k1 = torch.exp(-torch.arange(-3, 4, dtype=torch.float32, device=x.device) ** 2 / 2); k1 = k1 / k1.sum()
    b = F.conv2d(F.pad(g, (3, 3, 0, 0), mode="reflect"), k1.view(1, 1, 1, -1))
    b = F.conv2d(F.pad(b, (0, 0, 3, 3), mode="reflect"), k1.view(1, 1, -1, 1))
    v0 = F.conv2d(g, lap).var().item()
    v1 = F.conv2d(b, lap).var().item()
    return float(1.0 - v1 / v0) if v0 > 1e-6 else 0.0


# --------------------------------------------------------------------------------------------- generic features
class GenericFeatures:
    """Frozen ImageNet ResNet-18; image descriptor = mean of 512-d pooled features over 256-px tiles of the rescaled image."""

    def __init__(self, device: str | torch.device = "cpu") -> None:
        import torchvision
        os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "artifacts" / "torch_cache"))
        m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        self.device = torch.device(device)
        self.model = m.eval().to(self.device)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    @torch.no_grad()
    def batch(self, x_u8: torch.Tensor, tile: int = 256, max_tiles: int = 64) -> np.ndarray:
        """uint8 [B,3,H,W] (same size) -> [B,512] descriptors. Single images go through the same code (see __call__)."""
        x = x_u8.to(self.device).float() / 255.0
        B, _, h, w = x.shape
        if h < tile or w < tile:
            x = F.interpolate(x, size=(max(h, tile), max(w, tile)), mode="bilinear", align_corners=False)
            h, w = x.shape[-2:]
        r, c = h // tile, w // tile
        t = x[:, :, : r * tile, : c * tile].reshape(B, 3, r, tile, c, tile).permute(0, 2, 4, 1, 3, 5).reshape(B, r * c, 3, tile, tile)
        if r * c > max_tiles:                              # deterministic subsample for very large inputs
            t = t[:, torch.linspace(0, r * c - 1, max_tiles).long()]
        n = t.shape[1]
        feats = self.model((t.reshape(B * n, 3, tile, tile) - self.mean) / self.std).view(B, n, -1)
        return feats.mean(dim=1).cpu().numpy().astype(np.float64)

    def __call__(self, arr: np.ndarray, tile: int = 256, max_tiles: int = 64) -> np.ndarray:
        x = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1)[None]
        return self.batch(x, tile, max_tiles)[0]


# --------------------------------------------------------------------------------------------- fitted gate
def _fit_gaussian(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.covariance import LedoitWolf
    lw = LedoitWolf().fit(X)
    return lw.location_, lw.precision_


def mahalanobis(X: np.ndarray, mean: np.ndarray, prec: np.ndarray) -> np.ndarray:
    d = np.atleast_2d(X) - mean
    return np.sqrt(np.einsum("ij,jk,ik->i", d, prec, d))


@dataclass
class GateParams:
    feat_mean: list[float] = field(default_factory=list)
    feat_prec: list[list[float]] = field(default_factory=list)
    thresholds: dict[str, float] = field(default_factory=dict)
    limits: dict[str, float] = field(default_factory=lambda: {
        "min_mpp": 0.10, "max_mpp": 0.55,               # >0.55 um/px (coarser than ~20x): fine detail missing (Report 4/7 evidence)
        "min_side_px": 256, "max_background_fraction": 0.60, "max_dark_fraction": 0.05})
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=1))

    @classmethod
    def load(cls, path: Path) -> "GateParams":
        return cls(**json.loads(Path(path).read_text()))


def fit_gate(feat_X: np.ndarray) -> GateParams:
    fm, fp = _fit_gaussian(feat_X)
    return GateParams(feat_mean=fm.tolist(), feat_prec=fp.tolist())


def gate_distances(p: GateParams, feat: np.ndarray) -> dict[str, float]:
    return {"appearance_distance": float(mahalanobis(feat[None], np.array(p.feat_mean), np.array(p.feat_prec))[0])}


def _nan(v: float | None) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def evaluate_checks(p: GateParams, src_mpp: float | None, scaled_shape: tuple[int, int] | None, stats: dict[str, float] | None,
                    dists: dict[str, float] | None) -> dict[str, dict[str, Any]]:
    """Pure function: every check -> {'passed', 'value', 'limit', 'reason'}. Checks that cannot be measured on this image (e.g.
    no glass visible for white balance or noise) pass with value None and a 'note' - a documented limitation, not a guess."""
    L, T = p.limits, p.thresholds
    out: dict[str, dict[str, Any]] = {}

    def add(name: str, value: Any, ok: bool, limit: Any, reason: str, note: str | None = None) -> None:
        out[name] = {"passed": bool(ok), "value": value, "limit": limit, "reason": None if ok else reason}
        if note:
            out[name]["note"] = note

    add("scale_known", src_mpp, src_mpp is not None, "metadata or declared", "pixel size unknown: pass mpp or magnification")
    if src_mpp is None:
        return out
    add("magnification_sufficient", round(src_mpp, 4), L["min_mpp"] <= src_mpp <= L["max_mpp"], [L["min_mpp"], L["max_mpp"]],
        "magnification too low (needs ~20x or higher, <= 0.55 um/px)" if src_mpp > L["max_mpp"] else "pixel size implausibly small")
    if scaled_shape is not None:
        add("size_sufficient", list(scaled_shape), min(scaled_shape) >= L["min_side_px"], L["min_side_px"],
            "image too small after rescaling (needs at least 256 x 256 px at 0.26 um/px, about 67 um)")
    if stats is not None:
        add("enough_tissue", round(stats["background_fraction"], 4), stats["background_fraction"] <= L["max_background_fraction"],
            L["max_background_fraction"], "mostly empty glass/white background")
        add("no_large_dark_areas", round(stats["dark_fraction"], 4), stats["dark_fraction"] <= L["max_dark_fraction"],
            L["max_dark_fraction"], "large black/dark areas (viewer bars, borders, out-of-tissue)")
        if "saturation_min" in T:
            add("colour_present", round(stats["saturation_mean"], 4), stats["saturation_mean"] >= T["saturation_min"], T["saturation_min"],
                "image has (almost) no colour - grayscale images are not supported")
        if "he_hue_share_min" in T:
            v = stats.get("he_hue_share")
            add("stain_looks_like_HE", None if _nan(v) else round(v, 4), _nan(v) or v >= T["he_hue_share_min"], T["he_hue_share_min"],
                "tissue colours are not typical of H&E staining (other stain?)", "too few coloured pixels to measure" if _nan(v) else None)
        if "white_point_rb_range" in T:
            rb, g = stats.get("white_point_rb"), stats.get("white_point_g")
            if _nan(rb) or _nan(g):
                add("white_balance_neutral", None, True, None, "", "no bright glass visible: white balance could not be verified")
            else:
                lo1, hi1 = T["white_point_rb_range"]; lo2, hi2 = T["white_point_g_range"]
                add("white_balance_neutral", [round(rb, 4), round(g, 4)], lo1 <= rb <= hi1 and lo2 <= g <= hi2,
                    [T["white_point_rb_range"], T["white_point_g_range"]], "colour cast: white balance differs from the training scans")
        if "glass_noise_max" in T:
            v = stats.get("glass_noise")
            add("noise_low", None if _nan(v) else round(v, 3), _nan(v) or v <= T["glass_noise_max"], T["glass_noise_max"],
                "image is noisy", "no glass visible: noise could not be measured" if _nan(v) else None)
        if "focus_ratio_min" in T:
            add("in_focus", round(stats["focus_ratio"], 4), stats["focus_ratio"] >= T["focus_ratio_min"], T["focus_ratio_min"],
                "image looks blurred / out of focus")
    if dists is not None and "appearance_distance_max" in T:
        add("appearance_like_training", round(dists["appearance_distance"], 3), dists["appearance_distance"] <= T["appearance_distance_max"],
            T["appearance_distance_max"], "overall appearance unlike the training scans (other organ, scanner or not histology)")
    return out
