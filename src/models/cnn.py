"""Custom CNNs trained from scratch. Both take [B, in_channels, H, W] of any H, W >= 32 (global pooling)."""
from __future__ import annotations

import torch
from torch import nn


def _conv_bn_relu(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class SmallCNN(nn.Module):
    """M1: plain 5-stage CNN, two 3x3 convs per stage, max-pool between stages, global average pool."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2, widths: tuple[int, ...] = (16, 32, 64, 96, 128),
                 dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        cin = in_channels
        for i, w in enumerate(widths):
            layers += [_conv_bn_relu(cin, w), _conv_bn_relu(w, w)]
            if i < len(widths) - 1:
                layers.append(nn.MaxPool2d(2))
            cin = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.bn2(self.conv2(self.act(self.bn1(self.conv1(x)))))
        return self.act(y + (x if self.short is None else self.short(x)))


class ResidualCNN(nn.Module):
    """M2: small ResNet-style network. A light stem keeps full resolution for the first stage (fine nuclear detail),
    then four residual stages with stride-2 downsampling. ~1-3 M parameters with the default widths."""

    def __init__(self, in_channels: int = 3, num_classes: int = 2, widths: tuple[int, ...] = (32, 64, 128, 192),
                 blocks: tuple[int, ...] = (1, 1, 2, 1), dropout: float = 0.2) -> None:
        super().__init__()
        assert len(widths) == len(blocks)
        self.stem = nn.Sequential(_conv_bn_relu(in_channels, widths[0] // 2), _conv_bn_relu(widths[0] // 2, widths[0], stride=2))
        stages: list[nn.Module] = []
        cin = widths[0]
        for i, (w, n) in enumerate(zip(widths, blocks)):
            for j in range(n):
                stages.append(BasicBlock(cin, w, stride=2 if (j == 0 and i > 0) else 1))
                cin = w
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stages(self.stem(x)))
