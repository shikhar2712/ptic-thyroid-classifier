"""Model-ready data path: in-RAM uint8 image store, a Dataset, a patient-aware sampler, and GPU-side preprocessing.

Why this shape (see reports/02): `/dev/shm` is 64 MB here, so DataLoader workers cannot pass large float batches.
Images are decoded once into one uint8 array (1,006 x 1053 x 1916 x 3 = ~6.1 GB of 1 TiB RAM); the Dataset returns small
uint8 crops, and augmentation/normalisation run on the GPU. Statistics for normalisation come from *training* images only.
"""
from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

# Ruifrok & Johnston H&E(+residual) optical-density basis; rows are stain vectors. Used for mild stain jitter only.
_HED_ROWS = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]], dtype=np.float32)
_HED_ROWS = _HED_ROWS / np.linalg.norm(_HED_ROWS, axis=1, keepdims=True)


# ----------------------------------------------------------------------------------------------- store
def _decode(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))  # embedded ICC profile is deliberately ignored (policy in report 02)


@dataclass
class ImageStore:
    """All decoded images in one uint8 tensor [N, H, W, 3] plus aligned metadata."""

    images: torch.Tensor
    image_ids: list[str]
    labels: np.ndarray
    patient_ids: np.ndarray
    id_to_row: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id_to_row = {i: r for r, i in enumerate(self.image_ids)}

    def __len__(self) -> int:
        return len(self.image_ids)

    @property
    def hw(self) -> tuple[int, int]:
        return int(self.images.shape[1]), int(self.images.shape[2])

    def rows(self, image_ids: Sequence[str]) -> np.ndarray:
        return np.array([self.id_to_row[i] for i in image_ids], dtype=np.int64)

    @classmethod
    def from_manifest(cls, manifest: pd.DataFrame, images_root: Path, workers: int = 48) -> "ImageStore":
        man = manifest.reset_index(drop=True)
        first = _decode(images_root / man.path.iloc[0])
        out = np.empty((len(man), *first.shape), dtype=np.uint8)
        out[0] = first

        def work(i: int) -> None:
            a = _decode(images_root / man.path.iloc[i])
            if a.shape != first.shape:
                raise ValueError(f"{man.image_id.iloc[i]}: shape {a.shape} != {first.shape}")
            out[i] = a

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, range(1, len(man))))
        return cls(torch.from_numpy(out), man.image_id.tolist(), man.label.to_numpy(), man.patient_id.to_numpy())

    def save_npy(self, path: Path) -> None:
        np.save(path, self.images.numpy())

    @classmethod
    def from_npy(cls, path: Path, manifest: pd.DataFrame) -> "ImageStore":
        man = manifest.reset_index(drop=True)
        arr = np.load(path, mmap_mode="r")
        assert arr.shape[0] == len(man), "cache and manifest disagree on image count"
        return cls(torch.from_numpy(np.ascontiguousarray(arr)), man.image_id.tolist(), man.label.to_numpy(),
                   man.patient_id.to_numpy())


def channel_stats(manifest: pd.DataFrame) -> tuple[list[float], list[float]]:
    """Pooled per-channel mean/std (0-1 scale) from per-image statistics (equal pixel counts per image).

    Pass ONLY training rows. Uses E[x] and Var = mean(var_i + (mean_i - mu)^2).
    """
    mean, std = [], []
    for c in "rgb":
        m, s = manifest[f"mean_{c}"].to_numpy(np.float64), manifest[f"std_{c}"].to_numpy(np.float64)
        mu = m.mean()
        mean.append(mu / 255.0)
        std.append(float(np.sqrt((s**2 + (m - mu) ** 2).mean())) / 255.0)
    return mean, std


# ------------------------------------------------------------------------------------------- dataset
def balanced_weights(patient_ids: np.ndarray, balance: str = "none") -> np.ndarray:
    """Per-image sampling probabilities.

    'none': uniform over images. 'sqrt': a patient's total weight is proportional to sqrt(#images)
    (limits domination by patients such as #17 with 123 images). 'equal': every patient has equal total weight.
    """
    if balance not in ("none", "sqrt", "equal"):
        raise ValueError(f"unknown balance mode {balance!r}")
    _, inv, counts = np.unique(patient_ids, return_inverse=True, return_counts=True)
    n = counts[inv].astype(np.float64)
    w = {"none": np.ones_like(n), "sqrt": 1 / np.sqrt(n), "equal": 1 / n}[balance]
    return w / w.sum()


class PticDataset(Dataset):
    """Returns (uint8 CHW tensor, label, store row).

    mode='train': `samples_per_epoch` random square crops of side round(crop/scale). Which image each draw uses (with
      optional patient balancing) and where it is cropped are pure functions of (seed, epoch, draw index), so
      every draw is distinct and a run is reproducible. Call `set_epoch` before each epoch.
    mode='eval': the whole image, in the order of `rows`.
    """

    def __init__(self, store: ImageStore, rows: np.ndarray, mode: str = "train", crop: int = 256, scale: float = 1.0,
                 seed: int = 0, samples_per_epoch: int | None = None, balance: str = "none", source_margin: float = 1.0) -> None:
        assert mode in ("train", "eval")
        self.store, self.rows, self.mode, self.crop, self.scale, self.seed = store, np.asarray(rows), mode, crop, scale, seed
        # source_margin > 1 cuts a slightly larger region so GPU magnification jitter can zoom out without padding (step 2)
        self.side = int(math.ceil(round(crop / scale) * source_margin)) if source_margin != 1.0 else int(round(crop / scale))
        self.n = (samples_per_epoch or len(self.rows)) if mode == "train" else len(self.rows)
        self.weights = balanced_weights(store.patient_ids[self.rows], balance)
        H, W = store.hw
        if mode == "train" and (self.side > H or self.side > W):
            raise ValueError(f"crop side {self.side} larger than image {H}x{W}")
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
        if self.mode == "train":
            rng = np.random.default_rng([self.seed, epoch, 0xA11])
            self.plan = rng.choice(len(self.rows), size=self.n, replace=True, p=self.weights)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int, int]:
        if self.mode == "train":
            row = int(self.rows[self.plan[i]])
            H, W = self.store.hw
            rng = np.random.default_rng([self.seed, self.epoch, i])
            y, x = int(rng.integers(0, H - self.side + 1)), int(rng.integers(0, W - self.side + 1))
            img = self.store.images[row, y:y + self.side, x:x + self.side]
        else:
            row = int(self.rows[i])
            img = self.store.images[row]
        return img.permute(2, 0, 1).contiguous(), int(self.store.labels[row]), row


# ------------------------------------------------------------------------------------ GPU preprocessing
@dataclass
class AugConfig:
    flip: bool = True
    rot90: bool = True
    brightness: float = 0.10     # +/- fraction
    contrast: float = 0.10       # +/- fraction
    hed_sigma: float = 0.03      # stain-jitter strength; 0 disables. Mild by design.
    # Step-2 robustness options (Report 7 section 1.2), all off by default so earlier runs are reproduced exactly:
    saturation: float = 0.0      # saturation factor ~ U(1-s, 1+s)
    blur_sigma_max: float = 0.0  # Gaussian blur sigma ~ U(0, max) px on a share blur_p of the crops
    blur_p: float = 0.5
    scale_jitter: float = 0.0    # magnification jitter, zoom ~ U(1-j, 1+j). Small only (scanner calibration differences):
                                 # nuclear size is itself a PTC feature, so large zoom / RandomResizedCrop stay excluded.


def source_margin(aug: "AugConfig") -> float:
    """How much larger the training source crop must be for magnification jitter (1.0 when it is off)."""
    return 1.0 / (1.0 - aug.scale_jitter) if aug.scale_jitter > 0 else 1.0


class GPUPreprocess(torch.nn.Module):
    """uint8 [B,3,S,S] (on device) -> float32 normalised [B,3,crop,crop]; augmentation only if `train`."""

    def __init__(self, mean: Sequence[float], std: Sequence[float], crop: int = 256, scale: float = 1.0,
                 aug: AugConfig | None = None) -> None:
        super().__init__()
        self.crop, self.scale, self.aug = crop, scale, aug or AugConfig()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("hed", torch.from_numpy(_HED_ROWS))
        self.register_buffer("hed_inv", torch.from_numpy(np.linalg.inv(_HED_ROWS).astype(np.float32)))

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 1.0:
            return x
        size = (int(round(x.shape[-2] * self.scale)), int(round(x.shape[-1] * self.scale)))
        return F.interpolate(x, size=size, mode="area")

    def _hed_jitter(self, x: torch.Tensor, g: torch.Generator | None) -> torch.Tensor:
        B = x.shape[0]
        s = self.aug.hed_sigma
        alpha = 1 + (torch.rand(B, 3, 1, 1, device=x.device, generator=g) * 2 - 1) * s
        beta = (torch.rand(B, 3, 1, 1, device=x.device, generator=g) * 2 - 1) * s
        od = -torch.log(x.clamp(1 / 255, 1.0))                                  # optical density, [B,3,H,W]
        conc = torch.einsum("bchw,cd->bdhw", od, self.hed_inv)                  # stain concentrations
        conc = conc * alpha + beta
        od2 = torch.einsum("bdhw,dc->bchw", conc, self.hed).clamp_min(0)
        return torch.exp(-od2)

    def _scale_jitter(self, x: torch.Tensor, g: torch.Generator | None) -> torch.Tensor:
        """Per crop: zoom z ~ U(1-j, 1+j); take a random (S/z)-sided window of the larger source crop and resize it to S."""
        B, _, Hs, Ws = x.shape
        S = int(round(self.crop / self.scale))
        j = self.aug.scale_jitter
        z = 1 + (torch.rand(B, device=x.device, generator=g) * 2 - 1) * j
        u = torch.rand(B, 2, device=x.device, generator=g)
        out = []
        for b in range(B):
            L = min(Hs, Ws, int(round(S / float(z[b]))))
            oy, ox = int(u[b, 0] * (Hs - L)), int(u[b, 1] * (Ws - L))
            out.append(F.interpolate(x[b:b + 1, :, oy:oy + L, ox:ox + L], size=(S, S), mode="bilinear", align_corners=False, antialias=L > S))
        return torch.cat(out)

    def _blur(self, x: torch.Tensor, g: torch.Generator | None) -> torch.Tensor:
        a, (B, C, H, W) = self.aug, x.shape
        on = torch.rand(B, device=x.device, generator=g) < a.blur_p
        sig = (torch.rand(B, device=x.device, generator=g) * a.blur_sigma_max).clamp_min(1e-3)
        r = int(math.ceil(3 * a.blur_sigma_max))
        t = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
        k = torch.exp(-t[None] ** 2 / (2 * sig[:, None] ** 2)); k = k / k.sum(1, keepdim=True)
        delta = torch.zeros_like(k); delta[:, r] = 1
        k = torch.where(on[:, None], k, delta).repeat_interleave(C, 0)               # [B*C, 2r+1]
        y = x.reshape(1, B * C, H, W)
        y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="reflect"), k.view(B * C, 1, 1, -1), groups=B * C)
        y = F.conv2d(F.pad(y, (0, 0, r, r), mode="reflect"), k.view(B * C, 1, -1, 1), groups=B * C)
        return y.reshape(B, C, H, W)

    @torch.no_grad()
    def forward(self, x_u8: torch.Tensor, train: bool = False, generator: torch.Generator | None = None) -> torch.Tensor:
        x = x_u8.float() / 255.0
        if train and self.aug.scale_jitter > 0:
            x = self._scale_jitter(x, generator)
        x = self._resize(x)
        if train:
            a, B, dev = self.aug, x.shape[0], x.device
            if a.flip:
                m = torch.rand(B, device=dev, generator=generator) < 0.5
                x = torch.where(m.view(B, 1, 1, 1), x.flip(-1), x)
                m = torch.rand(B, device=dev, generator=generator) < 0.5
                x = torch.where(m.view(B, 1, 1, 1), x.flip(-2), x)
            if a.rot90 and x.shape[-1] == x.shape[-2]:
                k = torch.randint(0, 4, (B,), device=dev, generator=generator)
                x = torch.stack([torch.rot90(xi, int(ki), (-2, -1)) for xi, ki in zip(x, k.tolist())])
            if a.hed_sigma > 0:
                x = self._hed_jitter(x, generator)
            if a.saturation > 0:
                s_ = 1 + (torch.rand(B, 1, 1, 1, device=dev, generator=generator) * 2 - 1) * a.saturation
                lum = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
                x = lum + s_ * (x - lum)
            if a.brightness > 0 or a.contrast > 0:
                b = 1 + (torch.rand(B, 1, 1, 1, device=dev, generator=generator) * 2 - 1) * a.brightness
                c = 1 + (torch.rand(B, 1, 1, 1, device=dev, generator=generator) * 2 - 1) * a.contrast
                mu = x.mean(dim=(1, 2, 3), keepdim=True)
                x = ((x - mu) * c + mu) * b
            if a.blur_sigma_max > 0:
                x = self._blur(x.clamp(0, 1), generator)
            x = x.clamp(0, 1)
        return ((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last)


def tile_grid(h: int, w: int, tile: int) -> tuple[int, int]:
    """Number of non-overlapping tiles that fit (rows, cols). Remainder pixels at the border are not used."""
    return h // tile, w // tile


@torch.no_grad()
def tile_images(x_u8: torch.Tensor, pre: GPUPreprocess, tile: int) -> tuple[torch.Tensor, int]:
    """Whole images uint8 [B,3,H,W] -> normalised tiles [B*T,3,tile,tile] on a fixed grid (deterministic).

    Scale (if != 1) is applied to the whole image first, then it is tiled. Returns (tiles, T tiles per image).
    """
    x = pre(x_u8, train=False)                       # resize + normalise, [B,3,H',W']
    B, C, H, W = x.shape
    r, c = tile_grid(H, W, tile)
    if r == 0 or c == 0:
        raise ValueError(f"image {H}x{W} smaller than tile {tile}")
    x = x[:, :, : r * tile, : c * tile]
    x = x.reshape(B, C, r, tile, c, tile).permute(0, 2, 4, 1, 3, 5).reshape(B * r * c, C, tile, tile)
    return x.contiguous(memory_format=torch.channels_last), r * c
