import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from src.common import PROJECT_ROOT
from src.data.dataset import AugConfig, GPUPreprocess, ImageStore, PticDataset, source_margin

OLD = PROJECT_ROOT / "reports" / "provenance" / "test" / "dataset.py"      # frozen copy from before the step-2 changes


def load_old():
    import sys
    spec = importlib.util.spec_from_file_location("old_dataset", OLD)
    m = importlib.util.module_from_spec(spec)
    sys.modules["old_dataset"] = m            # dataclasses need the module registered while it executes
    spec.loader.exec_module(m)
    return m


@pytest.mark.skipif(not OLD.exists(), reason="frozen pre-change copy missing")
def test_defaults_reproduce_the_previous_code_bit_for_bit():
    old = load_old()
    x = torch.randint(0, 256, (6, 3, 64, 64), dtype=torch.uint8)
    e1_aug = dict(flip=True, rot90=True, brightness=0.1, contrast=0.1, hed_sigma=0.03)          # E1/E2 checkpoint configs
    new_pre = GPUPreprocess([.5] * 3, [.25] * 3, 64, 1.0, AugConfig(**e1_aug))
    old_pre = old.GPUPreprocess([.5] * 3, [.25] * 3, 64, 1.0, old.AugConfig(**e1_aug))
    for train in (False, True):
        a = new_pre(x, train=train, generator=torch.Generator().manual_seed(7))
        b = old_pre(x, train=train, generator=torch.Generator().manual_seed(7))
        assert torch.equal(a, b), f"train={train}"
    rng = np.random.default_rng(0)
    store = ImageStore(torch.from_numpy(rng.integers(0, 256, (6, 96, 160, 3), dtype=np.uint8)), [f"i{k}" for k in range(6)], np.arange(6) % 2, np.arange(6) // 2)
    nd, od = PticDataset(store, np.arange(6), "train", 32, 1.0, 3, 12), old.PticDataset(store, np.arange(6), "train", 32, 1.0, 3, 12)
    assert all(torch.equal(nd[i][0], od[i][0]) for i in range(12))


def test_source_margin_and_crop_side():
    assert source_margin(AugConfig()) == 1.0
    assert abs(source_margin(AugConfig(scale_jitter=0.1)) - 1 / 0.9) < 1e-12
    rng = np.random.default_rng(0)
    store = ImageStore(torch.from_numpy(rng.integers(0, 256, (4, 400, 400, 3), dtype=np.uint8)), list("abcd"), np.array([0, 1, 0, 1]), np.arange(4))
    assert PticDataset(store, np.arange(4), "train", 256, 1.0, 0, 4, source_margin=1 / 0.9).side == 285
    assert PticDataset(store, np.arange(4), "train", 256, 1.0, 0, 4).side == 256


def test_new_augmentations_shapes_ranges_and_determinism():
    aug = AugConfig(hed_sigma=0.1, brightness=0.15, contrast=0.15, saturation=0.2, blur_sigma_max=1.0, blur_p=0.5, scale_jitter=0.1)
    pre = GPUPreprocess([0.] * 3, [1.] * 3, 64, 1.0, aug)
    x = torch.randint(0, 256, (8, 3, 72, 72), dtype=torch.uint8)                  # 72 = ceil(64 / 0.9)
    a = pre(x, train=True, generator=torch.Generator().manual_seed(1))
    b = pre(x, train=True, generator=torch.Generator().manual_seed(1))
    assert a.shape == (8, 3, 64, 64) and torch.equal(a, b) and torch.isfinite(a).all()
    assert a.min() >= 0 and a.max() <= 1                                         # identity normalisation: values stay in [0, 1]
    ev = pre(torch.randint(0, 256, (2, 3, 128, 192), dtype=torch.uint8), train=False)
    assert ev.shape == (2, 3, 128, 192)                                          # evaluation path untouched (no jitter, no resize)


def test_blur_and_saturation_do_what_they_say():
    x = torch.randint(0, 256, (4, 3, 64, 64), dtype=torch.uint8)
    blur = GPUPreprocess([0.] * 3, [1.] * 3, 64, 1.0, AugConfig(flip=False, rot90=False, brightness=0, contrast=0, hed_sigma=0, blur_sigma_max=1.0, blur_p=1.0))
    y = blur(x, train=True, generator=torch.Generator().manual_seed(0))
    assert (y[..., 1:] - y[..., :-1]).abs().mean() < (x.float() / 255)[..., 1:].sub((x.float() / 255)[..., :-1]).abs().mean()
    g = torch.full((2, 3, 32, 32), 128, dtype=torch.uint8)
    sat = GPUPreprocess([0.] * 3, [1.] * 3, 32, 1.0, AugConfig(flip=False, rot90=False, brightness=0, contrast=0, hed_sigma=0, saturation=0.5))
    assert torch.allclose(sat(g, train=True, generator=torch.Generator().manual_seed(0)), g.float() / 255, atol=1e-6)   # grey stays grey
