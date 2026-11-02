import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import (AugConfig, GPUPreprocess, ImageStore, PticDataset, balanced_weights, channel_stats,
                              tile_grid, tile_images)
from src.evaluation.metrics import binary_metrics, cluster_bootstrap
from src.models.cnn import ResidualCNN, SmallCNN
from src.models.colour_baseline import fit_predict
from src.models.factory import build_model, count_params
from src.training.trainer import make_scheduler
from src.training.utils import deep_update, select_amp_dtype

H, W = 96, 160  # small stand-in for 1053 x 1916


def fake_store(n=12, seed=0):
    rng = np.random.default_rng(seed)
    imgs = torch.from_numpy(rng.integers(0, 256, (n, H, W, 3), dtype=np.uint8))
    pats = np.repeat(np.arange(n // 3), 3)
    return ImageStore(imgs, [f"i{k}" for k in range(n)], np.array([k % 2 for k in range(n)]), pats)


def test_store_row_lookup():
    s = fake_store()
    assert s.hw == (H, W) and len(s) == 12 and s.rows(["i3", "i0"]).tolist() == [3, 0]


def test_train_dataset_shapes_dtypes_and_determinism():
    s = fake_store()
    ds = PticDataset(s, np.arange(12), "train", crop=32, scale=1.0, seed=5, samples_per_epoch=20)
    x, y, row = ds[0]
    assert x.shape == (3, 32, 32) and x.dtype == torch.uint8 and y in (0, 1) and 0 <= row < 12
    assert len(ds) == 20
    ds2 = PticDataset(s, np.arange(12), "train", crop=32, scale=1.0, seed=5, samples_per_epoch=20)
    assert torch.equal(ds[7][0], ds2[7][0])
    ds.set_epoch(1)
    assert not all(torch.equal(ds[i][0], ds2[i][0]) for i in range(20))  # crops change with the epoch


def test_scale_changes_source_crop_side_not_output_size():
    s = fake_store()
    ds = PticDataset(s, np.arange(12), "train", crop=32, scale=0.5, seed=0)
    x, _, _ = ds[0]
    assert x.shape == (3, 64, 64)                          # source region is 2x larger...
    pre = GPUPreprocess([0.5] * 3, [0.25] * 3, crop=32, scale=0.5)
    assert pre(x[None], train=False).shape == (1, 3, 32, 32)  # ...and is area-downsampled to the crop size


def test_eval_dataset_returns_whole_images_in_order():
    s = fake_store()
    ds = PticDataset(s, np.array([4, 2]), "eval")
    x, y, row = ds[0]
    assert x.shape == (3, H, W) and row == 4 and torch.equal(x, s.images[4].permute(2, 0, 1))


def test_crop_larger_than_image_is_rejected():
    with pytest.raises(ValueError):
        PticDataset(fake_store(), np.arange(12), "train", crop=H + 1)


def test_patient_balancing_weights():
    pats = np.array([1] * 100 + [2] * 4)
    w_none, w_sqrt, w_eq = (balanced_weights(pats, m) for m in ("none", "sqrt", "equal"))
    assert w_none[:100].sum() == pytest.approx(100 / 104)
    assert w_eq[:100].sum() == pytest.approx(0.5) and w_eq[100:].sum() == pytest.approx(0.5)
    assert w_sqrt[:100].sum() == pytest.approx(10 / 12)
    with pytest.raises(ValueError):
        balanced_weights(pats, "bogus")


def test_gpu_preprocess_normalises_and_keeps_validation_deterministic():
    x = torch.randint(0, 256, (6, 3, 32, 32), dtype=torch.uint8)
    pre = GPUPreprocess([0.5, 0.5, 0.5], [0.25, 0.25, 0.25], 32, 1.0)
    a, b = pre(x, train=False), pre(x, train=False)
    assert torch.equal(a, b) and a.dtype == torch.float32 and torch.isfinite(a).all()
    assert a.min() >= -2.0 - 1e-5 and a.max() <= 2.0 + 1e-5      # (0-0.5)/0.25 .. (1-0.5)/0.25
    t = pre(x, train=True, generator=torch.Generator().manual_seed(0))
    assert t.shape == a.shape and torch.isfinite(t).all()


def test_hed_jitter_is_identity_when_disabled_and_mild_when_enabled():
    x = torch.randint(10, 245, (4, 3, 32, 32), dtype=torch.uint8)
    off = GPUPreprocess([0] * 3, [1] * 3, 32, aug=AugConfig(flip=False, rot90=False, brightness=0, contrast=0, hed_sigma=0))
    assert torch.allclose(off(x, train=True), x.float() / 255, atol=1e-6)
    on = GPUPreprocess([0] * 3, [1] * 3, 32, aug=AugConfig(flip=False, rot90=False, brightness=0, contrast=0, hed_sigma=0.03))
    d = (on(x, train=True, generator=torch.Generator().manual_seed(1)) - x.float() / 255).abs().mean()
    assert 0 < d < 0.05


def test_tiling_shapes_and_order():
    assert tile_grid(1053, 1916, 256) == (4, 7)
    x = torch.arange(2 * 3 * 40 * 70, dtype=torch.float32).reshape(2, 3, 40, 70).to(torch.uint8)
    pre = GPUPreprocess([0] * 3, [1] * 3, 16, 1.0)
    tiles, t = tile_images(x, pre, 16)
    assert t == 2 * 4 and tiles.shape == (2 * 8, 3, 16, 16)
    ref = (x[0, :, :16, :16].float() / 255)
    assert torch.allclose(tiles[0], ref)                          # first tile = top-left of first image
    with pytest.raises(ValueError):
        tile_images(x[:, :, :10, :10], pre, 16)


def test_channel_stats_use_given_rows_only():
    m = pd.DataFrame({"mean_r": [100., 200.], "mean_g": [100., 200.], "mean_b": [100., 200.],
                      "std_r": [10., 10.], "std_g": [10., 10.], "std_b": [10., 10.]})
    mean, std = channel_stats(m)
    assert mean[0] == pytest.approx(150 / 255)
    assert std[0] == pytest.approx(np.sqrt(100 + 2500) / 255)   # within-image var + between-image var
    assert channel_stats(m.iloc[:1])[0][0] == pytest.approx(100 / 255)


@pytest.mark.parametrize("name", ["small_cnn", "residual_cnn", "resnet18"])
@pytest.mark.parametrize("in_ch,n_cls", [(3, 2), (1, 2), (3, 3)])
def test_models_forward_shapes(name, in_ch, n_cls):
    m = build_model({"name": name, "in_channels": in_ch, "num_classes": n_cls}).eval()
    y = m(torch.randn(2, in_ch, 64, 64))
    assert y.shape == (2, n_cls) and torch.isfinite(y).all()


def test_param_budgets_match_design_targets():
    assert count_params(SmallCNN()) < 600_000
    assert 500_000 < count_params(ResidualCNN()) < 4_000_000
    with pytest.raises(KeyError):
        build_model({"name": "nope"})


def test_models_handle_variable_input_size():
    for m in (SmallCNN().eval(), ResidualCNN().eval()):
        assert m(torch.randn(1, 3, 96, 160)).shape == (1, 2)


def test_scheduler_warmup_then_cosine():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    s = make_scheduler(opt, total_steps=100, warmup_steps=10)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"]); opt.step(); s.step()
    assert lrs[0] < lrs[9] <= 1.0 and lrs[-1] < 0.01 and max(lrs) == pytest.approx(1.0, abs=0.01)


def test_config_merge_and_amp_fallback():
    assert deep_update({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}
    assert select_amp_dtype("bf16", torch.device("cpu")) is None and select_amp_dtype("fp32", torch.device("cpu")) is None


def test_binary_metrics_known_values():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([.9, .8, .3, .1, .2, .6, .4, .1])
    m = binary_metrics(y, p, 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["sensitivity"] == pytest.approx(2 / 3) and m["specificity"] == pytest.approx(4 / 5)
    assert m["precision"] == pytest.approx(2 / 3) and m["accuracy"] == pytest.approx(6 / 8)
    assert m["auroc"] == pytest.approx(13 / 15)
    assert np.isnan(binary_metrics(np.ones(4), np.full(4, .7))["auroc"])


def test_cluster_bootstrap_resamples_patients_and_skips_degenerate():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(10), 10)
    y = (groups % 2).astype(int)
    p = np.clip(y * 0.8 + rng.normal(0, .1, 100) + 0.1, 0, 1)
    auc = lambda yy, pp: binary_metrics(yy, pp)["auroc"]  # noqa: E731
    ci = cluster_bootstrap(y, p, groups, auc, n_boot=200, seed=1)
    assert ci["lo"] > 0.9 and ci["hi"] <= 1.0 and ci["n_valid"] + ci["n_skipped"] == 200


def test_colour_baseline_uses_only_training_rows_for_scaling():
    rng = np.random.default_rng(0)
    def mk(n, shift):
        d = {c: rng.normal(150 + shift, 10, n) for c in ["mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b", "gray_std", "white_frac"]}
        return pd.DataFrame(d)
    tr = pd.concat([mk(50, 0).assign(label=0), mk(50, 30).assign(label=1)])
    te = pd.concat([mk(20, 0).assign(label=0), mk(20, 30).assign(label=1)])
    p = fit_predict(tr, te)
    assert p.shape == (40,) and p[:20].mean() < 0.5 < p[20:].mean()
