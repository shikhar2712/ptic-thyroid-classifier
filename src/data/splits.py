"""Patient-grouped cross-validation splits with explicit leakage assertions.

Design (declared in reports/02_what_we_should_build.md before any model was trained):
  * outer 5-fold patient-grouped split (see `deal_patients`)  -> every patient is a test patient exactly once
  * per outer fold, validation patients are carved out of that fold's *training* patients only
  * the split seed is chosen by label/count-only criteria (never by model results)
Run:  python -I src/data/splits.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.common import get_logger, load_config, resolve, write_json  # noqa: E402

ROLES = ("train", "val", "test")


def _class_patients(df: pd.DataFrame, label: int) -> int:
    return int(df.loc[df.label == label, "patient_id"].nunique())


def patient_table(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("patient_id")
    return pd.DataFrame({"n_images": g.size(), "n_ptc": g.label.sum().astype(int),
                         "n_non": g.size() - g.label.sum().astype(int)}).reset_index()


def deal_patients(pt: pd.DataFrame, k: int, rng: np.random.Generator) -> dict[int, int]:
    """Assign whole patients to k groups so that every group gets a near-equal number of PTC-containing patients and of
    non-PTC-containing patients (mixed patients count for both), breaking ties by smaller total image count.

    Why not sklearn's StratifiedGroupKFold: it balances *image* class proportions, and one patient (#17) holds 123 of
    the 490 PTC images, so it effectively reserves a whole fold's PTC quota for that patient. On the real PTIC counts
    that left a fold with a single PTC patient in 1000/1000 seeds. Uses only patient IDs, counts and labels.
    """
    n_ptc, n_non, n_img = np.zeros(k, int), np.zeros(k, int), np.zeros(k, int)
    out: dict[int, int] = {}
    rows = pt.sort_values("patient_id").to_dict("records")
    ptc_pat = [r for r in rows if r["n_ptc"] > 0]
    non_only = [r for r in rows if r["n_ptc"] == 0]
    for r in [ptc_pat[i] for i in rng.permutation(len(ptc_pat))]:
        f = min(range(k), key=lambda g: (n_ptc[g], n_img[g], g))
        out[r["patient_id"]] = f
        n_ptc[f] += 1; n_non[f] += r["n_non"] > 0; n_img[f] += r["n_images"]
    for r in [non_only[i] for i in rng.permutation(len(non_only))]:
        f = min(range(k), key=lambda g: (n_non[g], n_img[g], g))
        out[r["patient_id"]] = f
        n_non[f] += 1; n_img[f] += r["n_images"]
    return out


def assign_folds(df: pd.DataFrame, k: int, seed: int, val_groups: int = 5) -> pd.DataFrame:
    """Return df[['image_id', ...]] + test_fold + role_fold{i} in {train, val, test} for every image."""
    out = df[["image_id", "patient_id", "wsi_key", "label"]].copy().reset_index(drop=True)
    pt = patient_table(out)
    test_of = deal_patients(pt, k, np.random.default_rng([seed, 0]))
    out["test_fold"] = out.patient_id.map(test_of).astype(int)
    for f in range(k):
        tr_pt = pt[pt.patient_id.map(test_of) != f]
        rng = np.random.default_rng([seed, 1000 + f])
        val_of = deal_patients(tr_pt, val_groups, rng)
        val_group = int(rng.integers(0, val_groups))
        val_patients = {p for p, g in val_of.items() if g == val_group}
        role = np.where(out.test_fold == f, "test", np.where(out.patient_id.isin(val_patients), "val", "train"))
        out[f"role_fold{f}"] = role
    return out


def split_report(df: pd.DataFrame, assign: pd.DataFrame, k: int) -> pd.DataFrame:
    """Patients / WSIs / images per fold, role and class."""
    rows = []
    for f in range(k):
        for role in ROLES:
            sub = assign[assign[f"role_fold{f}"] == role]
            rows.append({"fold": f, "role": role, "patients": sub.patient_id.nunique(), "wsis": sub.wsi_key.nunique(),
                         "images": len(sub), "images_ptc": int((sub.label == 1).sum()),
                         "images_non": int((sub.label == 0).sum()),
                         "patients_with_ptc": _class_patients(sub, 1), "patients_with_non": _class_patients(sub, 0)})
    return pd.DataFrame(rows)


def check_assignment(assign: pd.DataFrame, k: int, hashes: pd.Series | None = None) -> dict[str, Any]:
    """Hard leakage assertions (raise AssertionError) + soft metrics. `hashes` = sha256 aligned to assign rows."""
    soft: dict[str, Any] = {}
    assert assign.image_id.is_unique, "duplicate image_id rows"
    assert (assign.test_fold >= 0).all() and (assign.test_fold < k).all(), "image without a test fold"
    pt = assign.groupby("patient_id").test_fold.nunique()
    assert (pt == 1).all(), f"patients in >1 test fold: {pt[pt > 1].index.tolist()}"
    for f in range(k):
        sets = {r: set(assign.loc[assign[f"role_fold{f}"] == r, "patient_id"]) for r in ROLES}
        wsis = {r: set(assign.loc[assign[f"role_fold{f}"] == r, "wsi_key"]) for r in ROLES}
        for a in ROLES:
            assert sets[a], f"fold {f}: empty {a} partition"
            for b in ROLES:
                if a < b:
                    assert not (sets[a] & sets[b]), f"fold {f}: patient overlap {a}/{b}: {sets[a] & sets[b]}"
                    assert not (wsis[a] & wsis[b]), f"fold {f}: WSI overlap {a}/{b}"
        assert set(assign.loc[assign[f"role_fold{f}"] == "test", "image_id"]) == set(
            assign.loc[assign.test_fold == f, "image_id"]), f"fold {f}: test role != test_fold"
        if hashes is not None:  # byte-identical files must not straddle partitions
            h = pd.DataFrame({"h": hashes.values, "r": assign[f"role_fold{f}"].values})
            straddle = h.groupby("h").r.nunique()
            soft[f"fold{f}_identical_files_across_partitions"] = int((straddle > 1).sum())
    return soft


def criteria(report: pd.DataFrame, k: int) -> dict[str, Any]:
    """Pre-declared selection criteria (see report 02)."""
    te, va = report[report.role == "test"], report[report.role == "val"]
    frac = te.images_ptc / te.images
    ok_class = bool(((te.patients_with_ptc >= 2) & (te.patients_with_non >= 2)).all())
    ok_val_min = bool(((va.patients_with_ptc >= 1) & (va.patients_with_non >= 1)).all())
    ok_val_pref = bool(((va.patients_with_ptc >= 2) & (va.patients_with_non >= 2)).all())
    ok_frac = bool(((frac >= 0.25) & (frac <= 0.75)).all())
    dev = float((te.images - te.images.mean()).abs().max())
    return {"test_class_presence": ok_class, "val_class_presence_min": ok_val_min, "val_class_presence_preferred": ok_val_pref,
            "test_ptc_fraction_in_band": ok_frac, "max_test_size_deviation": dev,
            "qualifies": ok_class and ok_val_min and ok_frac}


def search_seed(df: pd.DataFrame, k: int = 5, seeds: range = range(1000)) -> tuple[int, pd.DataFrame, pd.DataFrame, dict]:
    """Choose the split seed using counts/labels only.

    Candidates must meet every hard criterion. Among them prefer seeds whose validation sets also have >=2
    patients per class ("preferred"), then the smallest test-size deviation, then the lowest seed.
    """
    cands: list[tuple[bool, float, int]] = []
    for s in seeds:
        c = criteria(split_report(df, assign_folds(df, k, s), k), k)
        if c["qualifies"]:
            cands.append((not c["val_class_presence_preferred"], c["max_test_size_deviation"], s))
    if not cands:
        raise RuntimeError(f"no seed in {seeds} satisfies the declared criteria; relax a criterion and document it")
    cands.sort()
    seed = cands[0][2]
    a = assign_folds(df, k, seed)
    rep = split_report(df, a, k)
    info = {"seed": seed, "seeds_tried": len(seeds), "seeds_qualifying": len(cands),
            "seeds_qualifying_with_preferred_val": int(sum(1 for c in cands if not c[0])), **criteria(rep, k)}
    return seed, a, rep, info


def main() -> int:
    cfg = load_config("data.yaml")
    log = get_logger("splits", resolve(cfg["paths"]["logs"]) / "splits.log")
    meta_dir, split_dir, rep_dir = resolve(cfg["paths"]["metadata"]), resolve(cfg["paths"]["splits"]), resolve(cfg["paths"]["reports"])
    man = pd.read_csv(meta_dir / "manifest.csv", keep_default_na=False, na_values=[])
    man["readable"] = man.readable.astype(str).str.lower().eq("true")
    usable = man[man.readable & (man.label >= 0)].reset_index(drop=True)
    log.info("usable images: %d of %d", len(usable), len(man))
    k = 5
    seed, assign, rep, info = search_seed(usable, k)
    soft = check_assignment(assign, k, hashes=usable.sha256)
    info["identical_files_across_partitions"] = soft
    split_dir.mkdir(parents=True, exist_ok=True)
    assign.to_csv(split_dir / "folds_v1.csv", index=False)
    rep.to_csv(split_dir / "folds_v1_summary.csv", index=False)
    pat = assign.groupby("patient_id").agg(test_fold=("test_fold", "first"), n_images=("image_id", "size")).reset_index()
    for f in range(k):
        pat[f"role_fold{f}"] = pat.patient_id.map(assign.groupby("patient_id")[f"role_fold{f}"].first())
    pat.to_csv(split_dir / "folds_v1_patients.csv", index=False)
    write_json(split_dir / "folds_v1_info.json", info)
    man = man.merge(assign[["image_id", "test_fold"]], on="image_id", how="left")
    man["test_fold"] = man.test_fold.fillna(-1).astype(int)
    man.to_csv(meta_dir / "manifest.csv", index=False)
    log.info("split seed %d | %s", seed, {k_: info[k_] for k_ in ("seeds_tried", "seeds_qualifying", "qualifies", "val_class_presence_preferred", "max_test_size_deviation")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
