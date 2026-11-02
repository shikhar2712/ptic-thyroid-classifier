import numpy as np
import pandas as pd
import pytest

from src.data.splits import ROLES, assign_folds, check_assignment, criteria, search_seed, split_report


def synthetic_manifest(seed=0):
    """24 patients shaped like PTIC: 7 PTC-only, 13 non-only, 4 mixed; 1-4 WSIs each, ~25 images per WSI."""
    rng = np.random.default_rng(seed)
    kinds = ["ptc"] * 7 + ["non"] * 13 + ["mixed"] * 4
    rows = []
    for pid, kind in enumerate(kinds, start=1):
        for w in range(1, int(rng.integers(1, 4)) + 1):
            label = {"ptc": 1, "non": 0, "mixed": w % 2}[kind]
            for n in range(int(rng.integers(8, 28))):
                rows.append({"image_id": f"p{pid}_w{w}_{n}", "patient_id": pid, "wsi_key": f"{pid}_{w}", "label": label,
                             "sha256": f"h{pid}_{w}_{n}"})
    return pd.DataFrame(rows)


def test_no_patient_or_wsi_overlap_in_any_fold():
    df = synthetic_manifest()
    a = assign_folds(df, 5, seed=3)
    check_assignment(a, 5, hashes=df.sha256)  # raises AssertionError on any leakage
    for f in range(5):
        pats = {r: set(a.loc[a[f"role_fold{f}"] == r, "patient_id"]) for r in ROLES}
        assert not (pats["train"] & pats["val"]) and not (pats["train"] & pats["test"]) and not (pats["val"] & pats["test"])


def test_every_patient_is_test_exactly_once_and_images_all_assigned():
    df = synthetic_manifest()
    a = assign_folds(df, 5, seed=1)
    assert len(a) == len(df) and a.image_id.is_unique
    assert a.groupby("patient_id").test_fold.nunique().eq(1).all()
    assert sorted(a.test_fold.unique()) == [0, 1, 2, 3, 4]


def test_validation_comes_only_from_training_patients():
    df = synthetic_manifest()
    a = assign_folds(df, 5, seed=2)
    for f in range(5):
        test_p = set(a.loc[a.test_fold == f, "patient_id"])
        val_p = set(a.loc[a[f"role_fold{f}"] == "val", "patient_id"])
        assert not (val_p & test_p)


def test_split_is_deterministic_for_a_seed_and_differs_across_seeds():
    df = synthetic_manifest()
    a1, a2, a3 = assign_folds(df, 5, 4), assign_folds(df, 5, 4), assign_folds(df, 5, 5)
    assert a1.equals(a2) and not a1.test_fold.equals(a3.test_fold)


def test_leakage_assertion_actually_fires():
    df = synthetic_manifest()
    a = assign_folds(df, 5, 0)
    bad = a.copy()
    pid = bad.loc[bad.role_fold0 == "test", "patient_id"].iloc[0]
    idx = bad.index[(bad.patient_id == pid)][0]
    bad.loc[idx, "role_fold0"] = "train"  # one image of a test patient leaks into train
    with pytest.raises(AssertionError, match="overlap"):
        check_assignment(bad, 5)
    bad2 = a.copy()
    bad2.loc[bad2.index[bad2.patient_id == pid][0], "test_fold"] = (a.loc[a.patient_id == pid, "test_fold"].iloc[0] + 1) % 5
    with pytest.raises(AssertionError):
        check_assignment(bad2, 5)


def test_identical_files_straddling_partitions_are_counted():
    df = synthetic_manifest()
    a = assign_folds(df, 5, 0)
    h = df.sha256.copy()
    tr = a.index[a.role_fold0 == "train"][0]
    te = a.index[a.role_fold0 == "test"][0]
    h.loc[te] = h.loc[tr]
    soft = check_assignment(a, 5, hashes=h)
    assert soft["fold0_identical_files_across_partitions"] == 1


def test_seed_search_meets_declared_criteria():
    df = synthetic_manifest()
    seed, a, rep, info = search_seed(df, 5, seeds=range(60))
    assert info["qualifies"]
    te = rep[rep.role == "test"]
    assert (te.patients_with_ptc >= 2).all() and (te.patients_with_non >= 2).all()
    check_assignment(a, 5)


def test_report_counts_are_consistent():
    df = synthetic_manifest()
    a = assign_folds(df, 5, 0)
    rep = split_report(df, a, 5)
    for f in range(5):
        r = rep[rep.fold == f]
        assert r.images.sum() == len(df) and r.patients.sum() == df.patient_id.nunique()
        assert (r.images_ptc + r.images_non == r.images).all()
    c = criteria(rep, 5)
    assert set(c) >= {"qualifies", "test_class_presence", "max_test_size_deviation"}


def test_every_test_fold_has_at_least_two_patients_of_each_class_for_any_seed():
    """The property sklearn's StratifiedGroupKFold could not deliver on PTIC-shaped data (see deal_patients docstring)."""
    df = synthetic_manifest()
    for seed in range(40):
        rep = split_report(df, assign_folds(df, 5, seed), 5)
        te = rep[rep.role == "test"]
        assert (te.patients_with_ptc >= 2).all() and (te.patients_with_non >= 2).all(), seed
