import numpy as np
import torch

from src.data.dataset import GPUPreprocess, tile_images
from src.explain.gradcam import cell_stats, gradcam_tiles, image_maps, spearman_rows, stitch, summary_stats
from src.models.cnn import ResidualCNN


def small_net(seed=0):
    torch.manual_seed(seed)
    net = ResidualCNN().eval()
    for m in net.modules():                       # non-trivial batch-norm statistics, as in a trained model
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.2, 0.2); m.running_var.uniform_(0.5, 2.0)
    return net


def test_gradcam_equals_cam_and_adds_up_to_the_logit_difference():
    net = small_net()
    x = torch.randn(3, 3, 64, 64)
    local, d = gradcam_tiles(net, x)
    z = net(x)
    lin = net.head[-1]
    assert local.shape == (3, 4, 4)
    assert torch.allclose(d, z[:, 1] - z[:, 0], atol=1e-5)                  # same score as the plain forward pass
    assert torch.allclose(local.mean(dim=(1, 2)) + lin.bias[1] - lin.bias[0], d, atol=1e-4)
    feats = net.stages(net.stem(x))
    assert torch.allclose(local, torch.einsum("k,nkhw->nhw", lin.weight[1] - lin.weight[0], feats), atol=1e-4)


def test_stitch_follows_the_tile_order_of_tile_images():
    rows, cols, tile = 2, 3, 32
    img = torch.zeros(1, 3, rows * tile + 5, cols * tile + 7, dtype=torch.uint8)  # remainder pixels are dropped, as in scoring
    for i in range(rows):
        for j in range(cols):
            img[..., i * tile:(i + 1) * tile, j * tile:(j + 1) * tile] = 10 * (i * cols + j + 1)
    pre = GPUPreprocess([0, 0, 0], [1, 1, 1], crop=tile)
    tiles, t = tile_images(img, pre, tile)
    per_tile = tiles.mean(dim=(1, 2, 3)).view(t, 1, 1).expand(t, 4, 4)
    m = stitch(per_tile, rows, cols)
    assert m.shape == (rows * 4, cols * 4)
    for i in range(rows):
        for j in range(cols):
            assert np.allclose(m[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4], 10 * (i * cols + j + 1) / 255, atol=1e-6)


def test_image_maps_average_back_to_each_models_image_score():
    net = small_net()
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (70, 100, 3), dtype=np.uint8)
    pre = GPUPreprocess([0.8, 0.7, 0.8], [0.1, 0.1, 0.1], crop=32)
    maps, diffs, cell = image_maps([(net, pre, 32)], arr, torch.device("cpu"))
    assert maps.shape == (1, 4, 6) and cell == 16                           # 2 x 3 squares of 32 px, total stride 16
    tiles, _ = tile_images(torch.from_numpy(arr).permute(2, 0, 1)[None], pre, 32)
    z = net(tiles).mean(dim=0)                                              # the scorer averages the logits over the squares
    lin = net.head[-1]
    assert abs(diffs[0] - float(z[1] - z[0])) < 1e-4
    assert abs(maps[0].mean() + float(lin.bias[1] - lin.bias[0]) - diffs[0]) < 1e-4


def test_cell_stats_white_and_hematoxylin():
    arr = np.zeros((16, 48, 3), np.uint8)
    arr[:, :16] = 250                      # white
    arr[:, 16:32] = (90, 40, 140)          # hematoxylin-like purple
    arr[:, 32:] = (240, 140, 190)          # eosin-like pink
    white, hema = cell_stats(arr, 16, (1, 3))
    assert white.tolist() == [[1.0, 0.0, 0.0]]
    assert hema[0, 1] > max(hema[0, 0], hema[0, 2])


def test_summary_stats_and_spearman():
    a = np.arange(12.0).reshape(1, 3, 4)
    assert np.allclose(spearman_rows(np.concatenate([a, a ** 2, -a]).reshape(3, -1)), [[1, 1, -1], [1, 1, -1], [-1, -1, 1]])
    maps = np.stack([np.ones((4, 4)), np.ones((4, 4))]); maps[:, 0, 0] = 3
    white = np.zeros((4, 4)); white[0, 0] = 1
    s = summary_stats(maps, white, np.ones((4, 4)) + white, 4)            # one 4 x 4 square: 12 ring cells, 4 interior
    assert s["white_area"] == 0.0625 and np.isclose(s["evidence_on_white"], 3 / 18, atol=1e-4)
    assert np.isclose(s["white_ratio"], (3 / 18) / 0.0625, atol=1e-3)
    assert np.isclose(s["border_ratio"], 14 / 12, atol=1e-4) and s["rho_hematoxylin"] == 1.0 and s["seed_agreement"] == 1.0
