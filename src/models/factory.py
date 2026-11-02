"""Model factory: build any candidate from a config dict (`name`, `in_channels`, `num_classes`, + extras)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.common import PROJECT_ROOT
from src.models.cnn import ResidualCNN, SmallCNN


def _resnet18(in_channels: int, num_classes: int, pretrained: bool, dropout: float = 0.0, **_: Any) -> nn.Module:
    import torchvision

    if pretrained:  # keep the 45 MB weight file inside the project, not in ~/.cache
        os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "artifacts" / "torch_cache"))
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    if in_channels != 3:
        old = m.conv1
        m.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        with torch.no_grad():
            if pretrained:  # average RGB filters, tile over the new channel count
                m.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1) * 3 / in_channels)
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(m.fc.in_features, num_classes))
    return m


def build_model(cfg: dict[str, Any]) -> nn.Module:
    name = cfg["name"]
    c, n = int(cfg.get("in_channels", 3)), int(cfg.get("num_classes", 2))
    extra = {k: v for k, v in cfg.items() if k not in ("name", "in_channels", "num_classes", "pretrained")}
    if name == "small_cnn":
        return SmallCNN(c, n, **{k: tuple(v) if isinstance(v, list) else v for k, v in extra.items()})
    if name == "residual_cnn":
        return ResidualCNN(c, n, **{k: tuple(v) if isinstance(v, list) else v for k, v in extra.items()})
    if name == "resnet18":
        return _resnet18(c, n, bool(cfg.get("pretrained", False)), **extra)
    raise KeyError(f"unknown model {name!r}; available: small_cnn, residual_cnn, resnet18")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
