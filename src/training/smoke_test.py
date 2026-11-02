"""End-to-end smoke test: disk -> manifest -> partitions -> uint8 crops -> GPU preprocessing -> model -> checkpoint.

Not a training run: a handful of optimisation steps on tiny budgets. Every step is wrapped so that a failure is
recorded in the output instead of aborting the rest. Results go to reports/smoke_test_results.json.
Run:  python -I src/training/smoke_test.py
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root (works under `python -I`)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.common import PROJECT_ROOT, env_snapshot, get_logger, load_config, resolve, write_json  # noqa: E402
from src.data.dataset import AugConfig, GPUPreprocess, ImageStore, PticDataset, channel_stats, tile_images  # noqa: E402
from src.inference.predict import predict_store  # noqa: E402
from src.models import colour_baseline  # noqa: E402
from src.models.factory import build_model, count_params  # noqa: E402
from src.training.trainer import fit, fold_rows  # noqa: E402
from src.training.utils import load_train_config, select_amp_dtype, select_device, seed_everything  # noqa: E402

SURFACE, INK = "#fcfcfb", "#0b0b0b"
RESULTS: dict[str, Any] = {}
log = get_logger("smoke", PROJECT_ROOT / "logs" / "smoke_test.log")


def step(name: str) -> Callable:
    def deco(fn: Callable) -> Callable:
        def run(*a: Any, **k: Any) -> Any:
            t0 = time.time()
            try:
                out = fn(*a, **k)
                RESULTS[name] = {"status": "ok", "seconds": round(time.time() - t0, 2), **(out if isinstance(out, dict) else {})}
                log.info("[ok]   %s (%.1fs)", name, time.time() - t0)
                return out
            except Exception as exc:  # noqa: BLE001 - recorded, not hidden
                RESULTS[name] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()[-1500:]}
                log.error("[FAIL] %s: %r", name, exc)
                return None
        return run
    return deco


def batch_report(x_u8: torch.Tensor, xb: torch.Tensor, y: torch.Tensor, device: torch.device) -> dict[str, Any]:
    return {"uint8_shape": list(x_u8.shape), "uint8_dtype": str(x_u8.dtype), "float_shape": list(xb.shape),
            "float_dtype": str(xb.dtype), "device": str(xb.device), "device_type_ok": xb.device.type == device.type,
            "value_min": round(float(xb.min()), 3), "value_max": round(float(xb.max()), 3),
            "value_mean": round(float(xb.mean()), 3), "value_std": round(float(xb.std()), 3),
            "finite": bool(torch.isfinite(xb).all()), "labels": sorted(set(y.tolist())),
            "n_label1": int((y == 1).sum()), "n_label0": int((y == 0).sum())}


def main() -> int:
    dcfg = load_config("data.yaml")
    cfg = load_train_config("train_default.yaml", {"fold": 0})
    device, fold = select_device("cuda"), 0
    amp = select_amp_dtype(cfg["precision"], device)
    seed_everything(cfg["seed"])
    RESULTS["environment"] = env_snapshot()
    man = pd.read_csv(resolve(dcfg["paths"]["metadata"]) / "manifest.csv", keep_default_na=False, na_values=[])
    folds = pd.read_csv(PROJECT_ROOT / cfg["split_file"])
    man = man[man.image_id.isin(folds.image_id)].reset_index(drop=True)
    ctx: dict[str, Any] = {}

    @step("01_build_in_ram_store")
    def _store() -> dict:
        t0 = time.time()
        ctx["store"] = ImageStore.from_manifest(man, resolve(dcfg["paths"]["raw_images"]))
        s = ctx["store"]
        return {"images": len(s), "hw": list(s.hw), "gib_in_ram": round(s.images.numel() / 2**30, 2),
                "decode_seconds": round(time.time() - t0, 1), "labels_match_manifest": bool((s.labels == man.label.to_numpy()).all())}
    _store()
    if "store" not in ctx:
        write_json(PROJECT_ROOT / "reports" / "smoke_test_results.json", RESULTS)
        return 1
    store: ImageStore = ctx["store"]

    tr, va, te = (fold_rows(store, folds, fold, r) for r in ("train", "val", "test"))
    mean, std = channel_stats(man.set_index("image_id").loc[[store.image_ids[r] for r in tr]])  # TRAIN rows only
    pre = GPUPreprocess(mean, std, 256, 1.0, AugConfig(**cfg["aug"])).to(device)
    ctx.update(pre=pre, tr=tr, va=va, te=te, mean=mean, std=std)
    RESULTS["partitions_fold0"] = {r: {"images": int(len(rows)), "patients": int(np.unique(store.patient_ids[rows]).size),
                                       "images_ptc": int((store.labels[rows] == 1).sum())} for r, rows in (("train", tr), ("val", va), ("test", te))}
    RESULTS["train_channel_mean_std"] = {"mean": mean, "std": std}

    @step("02_batch_from_each_partition")
    def _batches() -> dict:
        out = {}
        ds = PticDataset(store, tr, "train", 256, 1.0, 0, len(tr) * 4, "none")
        dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0, pin_memory=device.type == "cuda", drop_last=True)
        x, y, _ = next(iter(dl))
        xb = pre(x.to(device), train=True, generator=torch.Generator(device=device).manual_seed(0))
        out["train_crops"] = batch_report(x, xb, y, device)
        ctx["preview"] = {"train": (xb[:6].detach().float().cpu(), y[:6].clone())}
        for name, rows in (("val", va), ("test", te)):  # test partition: shape/dtype/leak-free loading check only
            rng = np.random.default_rng(0)               # 4 images per class, so the batch check sees both labels
            pick = np.concatenate([rng.choice(np.where(store.labels[rows] == c)[0], 4, replace=False) for c in (0, 1)])
            dse = PticDataset(store, rows, "eval", 256, 1.0, 0)
            xe = torch.stack([dse[int(i)][0] for i in pick]); ye = torch.tensor([dse[int(i)][1] for i in pick])
            tiles, t = tile_images(xe.to(device), pre, 256)
            rep = batch_report(xe, tiles, ye, device)
            rep["tiles_per_image"] = t
            out[f"{name}_whole_images_tiled"] = rep
            ctx["preview"][name] = (tiles[::t][[0, 1, 2, 4, 5, 6]].detach().float().cpu(), ye[[0, 1, 2, 4, 5, 6]])  # 3 per class, first tile each
        return out
    _batches()

    @step("03_preview_figure")
    def _preview() -> dict:
        plt.rcParams.update({"figure.facecolor": SURFACE, "text.color": INK})
        fig, axes = plt.subplots(3, 6, figsize=(13, 7))
        m, s = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)
        for r, part in enumerate(("train", "val", "test")):
            x, y = ctx["preview"][part]
            for c in range(6):
                ax = axes[r, c]; ax.axis("off")
                img = x[c].float().cpu() * s + m  # undo normalisation for display only
                ax.imshow(img.clamp(0, 1).permute(1, 2, 0).numpy())
                label = "PTC-like" if int(y[c]) == 1 else "Non-PTC-like"
                ax.set_title(f"{part}  {label}", fontsize=9, color=INK, loc="left")
        fig.suptitle("Model inputs, fold 0: train = augmented random 256 px crops; val/test = deterministic tiles (colours restored for display)", x=0.01, ha="left", fontsize=11)
        fig.tight_layout(); out = PROJECT_ROOT / "reports" / "figures" / "preprocessed_preview.png"
        out.parent.mkdir(exist_ok=True, parents=True); fig.savefig(out, dpi=110); plt.close(fig)
        return {"figure": str(out.relative_to(PROJECT_ROOT))}
    _preview()

    def measure(mc: dict, bs: int, crop: int = 256) -> dict:
        """Train-step memory for one config. Everything is freed on return; `leak_gib` shows whether it was."""
        import gc
        gc.collect(); torch.cuda.empty_cache()
        before = torch.cuda.memory_allocated()
        seed_everything(0)
        m = build_model(mc).to(device).to(memory_format=torch.channels_last)
        opt = torch.optim.AdamW(m.parameters(), 1e-3)
        x = torch.randn(bs, 3, crop, crop, device=device).contiguous(memory_format=torch.channels_last)
        y = torch.randint(0, 2, (bs,), device=device)

        def one() -> torch.Tensor:
            with torch.autocast(device.type, dtype=amp, enabled=amp is not None):
                loss = F.cross_entropy(m(x).float(), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            return loss
        torch.cuda.reset_peak_memory_stats()
        for _ in range(2):
            one()                                          # warm-up (cudnn autotuning may allocate big workspaces)
        torch.cuda.synchronize(); peak_warm = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats(); ts = []
        for _ in range(3):
            t0 = time.time(); loss = one(); torch.cuda.synchronize(); ts.append(time.time() - t0)
        res = {"params": count_params(m), "batch": bs, "crop": crop, "loss_finite": bool(torch.isfinite(loss)),
               "peak_vram_gib_steady": round(torch.cuda.max_memory_allocated() / 2**30, 3),
               "peak_vram_gib_incl_warmup": round(peak_warm / 2**30, 3), "ms_per_step": round(1000 * float(np.median(ts)), 1)}
        del m, opt, x, y, loss
        gc.collect(); torch.cuda.empty_cache()
        res["leak_gib"] = round((torch.cuda.memory_allocated() - before) / 2**30, 4)
        return res

    @step("04_models_forward_backward_and_vram")
    def _models() -> dict:
        return {name: measure(mc, 64) for name, mc in (("small_cnn", {"name": "small_cnn"}), ("residual_cnn", {"name": "residual_cnn"}),
                                                        ("resnet18_scratch", {"name": "resnet18", "pretrained": False}))}
    _models()

    @step("05_batch_size_sweep_residual_cnn")
    def _sweep() -> dict:
        out = {}
        for bs in (32, 64, 128, 256, 512):
            try:
                out[str(bs)] = measure({"name": "residual_cnn"}, bs)
            except torch.OutOfMemoryError:
                import gc
                gc.collect(); torch.cuda.empty_cache(); out[str(bs)] = {"oom": True}
        return {"by_batch": out, "gpu_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)}
    _sweep()

    @step("06_bf16_vs_fp32_loss_parity")
    def _parity() -> dict:
        seed_everything(0)
        m = build_model({"name": "residual_cnn"}).to(device).eval()
        x = torch.randn(32, 3, 256, 256, device=device)
        with torch.no_grad():
            l32 = m(x).float()
            with torch.autocast(device.type, dtype=torch.bfloat16):
                l16 = m(x).float()
        return {"max_abs_logit_diff": round(float((l32 - l16).abs().max()), 5), "rel_diff": round(float((l32 - l16).abs().max() / l32.abs().max()), 5)}
    _parity()

    @step("07_checkpoint_save_and_resume")
    def _ckpt() -> dict:
        import shutil
        run = PROJECT_ROOT / "checkpoints" / "_smoke" / f"fold{fold}"
        shutil.rmtree(run.parent, ignore_errors=True)
        base = load_train_config("train_default.yaml", {"fold": fold, "model": {"name": "small_cnn"}, "data": {"batch_size": 32, "crops_per_image": 1},
                                                       "smoke": {"max_train_steps": 3, "max_val_images": 16}, "optim": {"epochs": 1}})
        r1 = fit(base, store, man, folds, run, log)
        ck1 = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        cfg2 = load_train_config("train_default.yaml", {"fold": fold, "model": {"name": "small_cnn"}, "data": {"batch_size": 32, "crops_per_image": 1},
                                                        "smoke": {"max_train_steps": 3, "max_val_images": 16}, "optim": {"epochs": 2}})
        r2 = fit(cfg2, store, man, folds, run, log)
        lines = [json.loads(l) for l in open(run / "metrics.jsonl")]
        ck2 = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        changed = any(not torch.equal(ck1["model"][k], ck2["model"][k]) for k in ck1["model"] if ck1["model"][k].dtype.is_floating_point)
        bad = load_train_config("train_default.yaml", {"fold": fold, "model": {"name": "small_cnn"}, "optim": {"lr": 0.5, "epochs": 2}})
        try:
            fit(bad, store, man, folds, run, log); mismatch_raises = False
        except RuntimeError:
            mismatch_raises = True
        shutil.rmtree(run.parent, ignore_errors=True)
        return {"epochs_logged": [l["epoch"] for l in lines], "resumed_from_epoch": ck1["state"]["epoch"] + 1, "weights_changed_after_resume": changed,
                "config_mismatch_refused": mismatch_raises, "files": ["last.pt", "best.pt", "metrics.jsonl", "run_info.json"],
                "val_metrics_epoch0": {k: lines[0][k] for k in ("val_loss", "val_auroc")}}
    _ckpt()

    @step("08_tiled_inference_on_validation_images")
    def _infer() -> dict:
        m = build_model({"name": "residual_cnn"}).to(device).to(memory_format=torch.channels_last)
        lg, p = predict_store(m, store, va[:16], pre, 256, 8, device, amp)
        return {"n": int(len(p)), "prob_min": round(float(p.min()), 4), "prob_max": round(float(p.max()), 4), "finite": bool(np.isfinite(lg).all()),
                "note": "untrained network; values are not meaningful, only the path is verified"}
    _infer()

    @step("09_dataloader_workers_under_64MB_shm")
    def _workers() -> dict:
        out: dict[str, Any] = {}
        ds = PticDataset(store, tr, "train", 256, 1.0, 0, 512, "none")
        for nw, bs in ((0, 64), (2, 64), (4, 64), (4, 16)):
            t0 = time.time()
            try:
                kw = {"timeout": 90} if nw else {}
                n = sum(1 for _ in zip(range(6), DataLoader(ds, batch_size=bs, num_workers=nw, drop_last=True, **kw)))
                out[f"workers={nw},batch={bs}"] = {"ok": True, "batches": n, "seconds": round(time.time() - t0, 2)}
            except Exception as exc:  # noqa: BLE001
                lines = [l.strip() for l in str(exc).splitlines() if l.strip()]
                out[f"workers={nw},batch={bs}"] = {"ok": False, "error": lines[-1][:240] if lines else repr(exc), "seconds": round(time.time() - t0, 2)}
        return {"results": out, "batch_mib_uint8_64x3x256x256": 12.0, "dev_shm_mib": 64}
    _workers()

    @step("10_colour_baseline_pipeline_check_VALIDATION_ONLY")
    def _b0() -> dict:
        from src.evaluation.metrics import binary_metrics
        m_ = man.set_index("image_id")
        a, b = m_.loc[[store.image_ids[r] for r in tr]], m_.loc[[store.image_ids[r] for r in va]]
        p = colour_baseline.fit_predict(a, b)
        met = binary_metrics(b.label.to_numpy(), p)
        return {"note": "pipeline check on fold-0 VALIDATION partition only; the outer test fold was not scored; not an experimental result",
                "val_images": int(len(b)), "val_patients": int(b.patient_id.nunique()), "val_auroc": round(met["auroc"], 4)}
    _b0()

    RESULTS["all_steps_ok"] = all(v.get("status") == "ok" for k, v in RESULTS.items() if k[:2].isdigit())
    write_json(PROJECT_ROOT / "reports" / "smoke_test_results.json", RESULTS)
    log.info("smoke test finished; all_steps_ok=%s", RESULTS["all_steps_ok"])
    return 0 if RESULTS["all_steps_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
