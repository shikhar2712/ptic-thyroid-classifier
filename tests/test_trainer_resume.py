import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import ImageStore
from src.training.trainer import fit
from src.training.utils import load_train_config

H, W, N = 96, 160, 24


def tiny_setup():
    rng = np.random.default_rng(0)
    store = ImageStore(torch.from_numpy(rng.integers(0, 256, (N, H, W, 3), dtype=np.uint8)), [f"i{k}" for k in range(N)],
                       np.arange(N) % 2, np.arange(N) // 3)                       # 8 patients x 3 images, both classes each
    man = pd.DataFrame({"image_id": store.image_ids, "patient_id": store.patient_ids, "label": store.labels,
                        **{f"{s}_{c}": 128.0 if s == "mean" else 40.0 for s in ("mean", "std") for c in "rgb"}})
    role = np.where(store.patient_ids <= 4, "train", np.where(store.patient_ids <= 6, "val", "test"))
    return store, man, pd.DataFrame({"image_id": store.image_ids, "role_fold0": role})


def cfg(epochs):
    return load_train_config("train_default.yaml", {"fold": 0, "model": {"name": "small_cnn", "widths": [4, 8]},
                                                    "data": {"crop": 32, "eval_tile": 32, "batch_size": 4, "crops_per_image": 1},
                                                    "optim": {"epochs": epochs}})


def test_checkpoint_resume_continues_and_refuses_changed_config(tmp_path):
    store, man, folds = tiny_setup()
    run = tmp_path / "run"
    fit(cfg(1), store, man, folds, run)                       # epoch 0
    before = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert before["state"]["epoch"] == 0
    fit(cfg(2), store, man, folds, run)                       # resumes at epoch 1 (regression: RNG state was moved to the GPU)
    after = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
    assert after["state"]["epoch"] == 1
    assert [json.loads(l)["epoch"] for l in open(run / "metrics.jsonl")] == [0, 1]
    assert any(not torch.equal(before["model"][k], after["model"][k]) for k in before["model"] if before["model"][k].dtype.is_floating_point)
    bad = cfg(3); bad["optim"]["lr"] = 0.5
    with pytest.raises(RuntimeError, match="different configuration"):
        fit(bad, store, man, folds, run)


class RecordingImages:
    """Stand-in for ImageStore.images that records every image row the training code indexes."""

    def __init__(self, t: torch.Tensor) -> None:
        self.t, self.accessed = t, set()

    @property
    def shape(self):
        return self.t.shape

    def __getitem__(self, idx):
        first = idx[0] if isinstance(idx, tuple) else idx
        if isinstance(first, torch.Tensor):
            self.accessed.update(first.flatten().tolist())
        elif isinstance(first, (int, np.integer)):
            self.accessed.add(int(first))
        else:
            raise AssertionError(f"unrecognised image index {idx!r}; extend RecordingImages")
        return self.t[idx]


def test_fit_never_reads_test_partition_images(tmp_path):
    """Records every image row `fit` touches (training draws AND validation scoring) and asserts none is a test row."""
    store, man, folds = tiny_setup()
    rec = RecordingImages(store.images)
    spy = ImageStore(rec, store.image_ids, store.labels, store.patient_ids)
    test_rows = set(np.where(folds.role_fold0 == "test")[0].tolist())
    allowed = set(range(N)) - test_rows
    fit(cfg(1), spy, man, folds, tmp_path / "run")
    assert rec.accessed, "spy saw no reads at all, so this test would prove nothing"
    assert rec.accessed.isdisjoint(test_rows), f"fit() read test rows: {sorted(rec.accessed & test_rows)}"
    assert rec.accessed <= allowed
    assert rec.accessed & set(np.where(folds.role_fold0 == "val")[0].tolist()), "validation images were never scored"
    assert rec.accessed & set(np.where(folds.role_fold0 == "train")[0].tolist()), "training images were never read"
