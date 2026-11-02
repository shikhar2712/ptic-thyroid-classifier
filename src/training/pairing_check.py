"""Pre-launch check that two configs form a PAIRED comparison: same image draws, same crop coordinates (byte-identical source
crops), same augmentation random draws and same initial weights for a given seed/fold, so that only the declared difference
(here: pixel size) separates them. Replays trainer.fit's construction order up to the first training batch; trains nothing.

  python -I src/training/pairing_check.py --a e1_residual.yaml --b e2_lowres.yaml --seed 0 --fold 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.common import PROJECT_ROOT, load_config, resolve, write_json  # noqa: E402
from src.data.dataset import AugConfig, GPUPreprocess, ImageStore, PticDataset, channel_stats, source_margin  # noqa: E402
from src.models.factory import build_model  # noqa: E402
from src.training.trainer import fold_rows  # noqa: E402
from src.training.utils import load_train_config, seed_everything, select_device  # noqa: E402


def replay(cfg: dict, store: ImageStore, man: pd.DataFrame, folds: pd.DataFrame, device: torch.device) -> dict:
    """Same order of RNG-relevant steps as trainer.fit, up to and including the first augmented batch of epoch 0."""
    seed_everything(cfg["seed"], cfg["deterministic"])
    rng0 = torch.get_rng_state().clone()
    f, d = cfg["fold"], cfg["data"]
    tr = fold_rows(store, folds, f, "train")
    mean, std = channel_stats(man.set_index("image_id").loc[[store.image_ids[r] for r in tr]])
    pre = GPUPreprocess(mean, std, d["crop"], d["scale"], AugConfig(**cfg["aug"])).to(device)
    ds = PticDataset(store, tr, "train", d["crop"], d["scale"], cfg["seed"], len(tr) * d["crops_per_image"], d["balance"],
                     source_margin=source_margin(AugConfig(**cfg["aug"])))
    dl = DataLoader(ds, batch_size=d["batch_size"], shuffle=False, num_workers=0, pin_memory=device.type == "cuda", drop_last=True)
    untouched = bool(torch.equal(rng0, torch.get_rng_state()))          # constructions above must not consume the torch RNG
    model = build_model(cfg["model"])
    init = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ds.set_epoch(0)
    gen = torch.Generator(device=device); gen.manual_seed(cfg["seed"] * 100_003 + 0)
    x, y, rows = next(iter(dl))
    out = pre(x.to(device), train=True, generator=gen)
    return {"rows": rows.clone(), "crops_u8": x.clone(), "labels": y.clone(), "init": init, "gen_state_after_batch": gen.get_state().clone(),
            "rng_untouched_by_setup": untouched, "model_input_shape": tuple(out.shape), "source_side": ds.side,
            "steps_per_epoch": len(dl), "crops_per_epoch": len(ds)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="e1_residual.yaml"); ap.add_argument("--b", default="e2_lowres.yaml")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="check every seed in --seeds x fold in --folds (writes <tag>_pairing_check_all.json)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2]); ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--tag", default="e2")
    a = ap.parse_args()
    dcfg = load_config("data.yaml")
    man = pd.read_csv(resolve(dcfg["paths"]["metadata"]) / "manifest.csv", keep_default_na=False, na_values=[])
    folds = pd.read_csv(PROJECT_ROOT / "data/splits/folds_v1.csv")
    man = man[man.image_id.isin(folds.image_id)].reset_index(drop=True)
    store = ImageStore.from_manifest(man, resolve(dcfg["paths"]["raw_images"]))
    device = select_device("cuda")
    if a.all:   # loop over all requested combinations; keep only the booleans
        allres = []
        for sd in a.seeds:
            for fd in a.folds:
                ra, rb = (replay(load_train_config(c, {"seed": sd, "fold": fd}), store, man, folds, device) for c in (a.a, a.b))
                r = {"seed": sd, "fold": fd, "same_rows_drawn": bool(torch.equal(ra["rows"], rb["rows"])),
                     "byte_identical_source_crops": bool(torch.equal(ra["crops_u8"], rb["crops_u8"])),
                     "same_initial_weights": all(torch.equal(ra["init"][k], rb["init"][k]) for k in ra["init"]),
                     "same_augmentation_draws": bool(torch.equal(ra["gen_state_after_batch"], rb["gen_state_after_batch"])),
                     "rng_untouched": ra["rng_untouched_by_setup"] and rb["rng_untouched_by_setup"],
                     "same_steps_per_epoch": ra["steps_per_epoch"] == rb["steps_per_epoch"]}
                r["PAIRED"] = all(v for k, v in r.items() if k not in ("seed", "fold"))
                allres.append(r); print(r)
                del ra, rb
        write_json(PROJECT_ROOT / "reports" / f"{a.tag}_pairing_check_all.json", {"configs": [a.a, a.b], "checked": len(allres),
                   "all_paired": all(r["PAIRED"] for r in allres), "detail": allres,
                   "note": "run after the E2 runs; pairing is a property of code + configs + fold file, not of results"})
        print("ALL PAIRED:", all(r["PAIRED"] for r in allres), "| combinations:", len(allres))
        return 0 if all(r["PAIRED"] for r in allres) else 1
    ra, rb = (replay(load_train_config(c, {"seed": a.seed, "fold": a.fold}), store, man, folds, device) for c in (a.a, a.b))
    res = {"seed": a.seed, "fold": a.fold, "configs": [a.a, a.b],
           "same_rows_drawn": bool(torch.equal(ra["rows"], rb["rows"])),
           "same_labels": bool(torch.equal(ra["labels"], rb["labels"])),
           "byte_identical_source_crops": bool(torch.equal(ra["crops_u8"], rb["crops_u8"])),
           "same_initial_weights": bool(ra["init"].keys() == rb["init"].keys() and all(torch.equal(ra["init"][k], rb["init"][k]) for k in ra["init"])),
           "same_augmentation_draws": bool(torch.equal(ra["gen_state_after_batch"], rb["gen_state_after_batch"])),
           "torch_rng_untouched_by_setup": [ra["rng_untouched_by_setup"], rb["rng_untouched_by_setup"]],
           "source_side_px": [ra["source_side"], rb["source_side"]], "model_input_shape": [list(ra["model_input_shape"]), list(rb["model_input_shape"])],
           "steps_per_epoch": [ra["steps_per_epoch"], rb["steps_per_epoch"]], "crops_per_epoch": [ra["crops_per_epoch"], rb["crops_per_epoch"]],
           "note": "same_augmentation_draws compares the GPU generator state after the first batch: equal start state + equal state after => the same random numbers were drawn"}
    res["PAIRED"] = all(res[k] for k in ("same_rows_drawn", "same_labels", "byte_identical_source_crops", "same_initial_weights", "same_augmentation_draws")) \
        and all(res["torch_rng_untouched_by_setup"]) and res["steps_per_epoch"][0] == res["steps_per_epoch"][1]
    write_json(PROJECT_ROOT / "reports" / f"{a.tag}_pairing_check.json", res)
    for k, v in res.items():
        print(f"{k}: {v}")
    return 0 if res["PAIRED"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
