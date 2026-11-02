import numpy as np
import torch

from src.evaluation.stress_test import CONDITIONS, perturb


def img(seed=0, h=64, w=96):
    rng = np.random.default_rng(seed)
    return torch.from_numpy(rng.integers(20, 236, (2, 3, h, w), dtype=np.uint8))


def maxdiff(a, b):
    return int((a.int() - b.int()).abs().max())


def test_neutral_parameters_are_identity_up_to_rounding():
    x = img()
    assert torch.equal(perturb(x, "none", None), x)
    for kind, param in [("stain", (None, 1.0)), ("stain", (0, 1.0)), ("brightness", 1.0), ("contrast", 1.0), ("saturation", 1.0),
                        ("gamma", 1.0), ("cast", (1.0, 1.0, 1.0)), ("hue", 0.0), ("scale", 1.0)]:
        assert maxdiff(perturb(x, kind, param), x) <= 1, (kind, param)


def test_shapes_types_and_determinism():
    x = img()
    for name, fam, kind, param, status in CONDITIONS:
        y = perturb(x, kind, param, seed=3)
        assert y.dtype == torch.uint8 and y.shape[:2] == x.shape[:2], name
        if kind != "scale":
            assert y.shape == x.shape, name
        assert torch.equal(y, perturb(x, kind, param, seed=3)), f"{name} not deterministic"
    assert perturb(x, "scale", 0.5).shape[-2:] == (32, 48) and perturb(x, "scale", 1.5).shape[-2:] == (96, 144)


def test_directions_make_sense():
    x = img()
    g = perturb(x, "gray", None)
    assert torch.equal(g[:, 0], g[:, 1]) and torch.equal(g[:, 1], g[:, 2])
    assert perturb(x, "stain", (None, 1.4)).float().mean() < x.float().mean() < perturb(x, "stain", (None, 0.7)).float().mean()  # dark < orig < pale
    assert perturb(x, "blur", 2.0).float().std() < x.float().std()
    assert maxdiff(perturb(x, "jpeg", 30), x) > 0


def test_status_labels_are_the_declared_ones():
    held = {n for n, f, k, p, s in CONDITIONS if s == "held_out"}
    assert {k for n, f, k, p, s in CONDITIONS if s == "held_out"} == {"jpeg", "cast", "gamma", "hue", "gray", "noise"}
    assert not ({k for n, f, k, p, s in CONDITIONS if s.startswith("trained")} & {"jpeg", "cast", "gamma", "hue", "gray", "noise"})
    assert len(held) == 13
