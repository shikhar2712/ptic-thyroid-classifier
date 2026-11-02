"""Tiled whole-image inference, shared by validation, test evaluation and (later) serving.

One prediction per labelled image: tile the image on a fixed grid, average the logits over tiles, softmax.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

from src.data.dataset import GPUPreprocess, ImageStore, tile_images


@torch.no_grad()
def predict_uint8(model: nn.Module, imgs_u8: torch.Tensor, pre: GPUPreprocess, tile: int,
                  amp_dtype: torch.dtype | None) -> torch.Tensor:
    """imgs_u8: [B,3,H,W] uint8 on the model's device -> mean logits per image [B, num_classes] (float32)."""
    tiles, t = tile_images(imgs_u8, pre, tile)
    with torch.autocast(device_type=tiles.device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        logits = model(tiles).float()
    return logits.view(imgs_u8.shape[0], t, -1).mean(dim=1)


@torch.no_grad()
def predict_store(model: nn.Module, store: ImageStore, rows: np.ndarray, pre: GPUPreprocess, tile: int,
                  batch_images: int, device: torch.device, amp_dtype: torch.dtype | None) -> tuple[np.ndarray, np.ndarray]:
    """Returns (logits [N,C], prob of class 1 [N]) in the order of `rows`."""
    was_training = model.training
    model.eval()
    out = []
    for i in range(0, len(rows), batch_images):
        x = store.images[torch.as_tensor(rows[i:i + batch_images])].permute(0, 3, 1, 2)
        out.append(predict_uint8(model, x.to(device, non_blocking=True), pre, tile, amp_dtype).cpu())
    model.train(was_training)
    logits = torch.cat(out)
    return logits.numpy(), torch.softmax(logits, dim=1)[:, 1].numpy()


def predict_file(model: nn.Module, path: Path, pre: GPUPreprocess, tile: int, device: torch.device,
                 amp_dtype: torch.dtype | None) -> float:
    """Probability of PTC-like appearance for one image file (embedded ICC ignored, as in training)."""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    logits = predict_uint8(model.eval(), x, pre, tile, amp_dtype)
    return float(torch.softmax(logits, dim=1)[0, 1])
