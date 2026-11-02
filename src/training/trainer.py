"""Training loop with checkpoint/resume and structured logging.

Leakage guard: `fit` reads ONLY the 'train' and 'val' roles of the chosen fold. The outer-test images are never
loaded here; they are evaluated by a separate, explicit call after the model is frozen.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common import env_snapshot, get_logger, write_json
from src.data.dataset import AugConfig, GPUPreprocess, ImageStore, PticDataset, channel_stats, source_margin
from src.evaluation.metrics import binary_metrics
from src.inference.predict import predict_store
from src.models.factory import build_model, count_params
from src.training.utils import rng_state, seed_everything, select_amp_dtype, select_device, set_rng_state

_VOLATILE = ("resume", "smoke", "output")  # may change between a run and its resume without invalidating it


def config_hash(cfg: dict[str, Any]) -> str:
    c = {k: v for k, v in cfg.items() if k not in _VOLATILE}
    c["optim"] = {k: v for k, v in c["optim"].items() if k != "epochs"}
    return hashlib.sha256(json.dumps(c, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def make_scheduler(opt: torch.optim.Optimizer, total_steps: int, warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def fold_rows(store: ImageStore, folds: pd.DataFrame, fold: int, role: str) -> np.ndarray:
    ids = folds.loc[folds[f"role_fold{fold}"] == role, "image_id"].tolist()
    return store.rows(ids)


def fit(cfg: dict[str, Any], store: ImageStore, manifest: pd.DataFrame, folds: pd.DataFrame, run_dir: Path,
        log=None) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    log = log or get_logger(f"train.{run_dir.name}", run_dir / "train.log")
    f = int(cfg["fold"])
    device = select_device("cuda")
    amp = select_amp_dtype(cfg["precision"], device)
    seed_everything(cfg["seed"], cfg["deterministic"])

    tr_rows, va_rows = fold_rows(store, folds, f, "train"), fold_rows(store, folds, f, "val")
    man = manifest.set_index("image_id")
    tr_ids = [store.image_ids[r] for r in tr_rows]
    mean, std = channel_stats(man.loc[tr_ids])             # data-dependent preprocessing: training images only
    d, a = cfg["data"], cfg["aug"]
    pre = GPUPreprocess(mean, std, d["crop"], d["scale"], AugConfig(**a)).to(device)
    ds = PticDataset(store, tr_rows, "train", d["crop"], d["scale"], cfg["seed"], len(tr_rows) * d["crops_per_image"], d["balance"],
                     source_margin=source_margin(AugConfig(**a)))
    dl = DataLoader(ds, batch_size=d["batch_size"], shuffle=False, num_workers=d["num_workers"], pin_memory=device.type == "cuda",
                    drop_last=True, timeout=120 if d["num_workers"] else 0)
    model = build_model(cfg["model"]).to(device).to(memory_format=torch.channels_last)
    o = cfg["optim"]
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=o["weight_decay"])
    steps_per_epoch = len(dl) if not cfg["smoke"]["max_train_steps"] else min(len(dl), cfg["smoke"]["max_train_steps"])
    sched = make_scheduler(opt, o["epochs"] * steps_per_epoch, o["warmup_epochs"] * steps_per_epoch)

    state = {"epoch": -1, "best": -1.0, "best_loss": float("inf"), "bad": 0}
    last_p, best_p = run_dir / "last.pt", run_dir / "best.pt"
    chash = config_hash(cfg)
    if cfg["resume"] and last_p.exists():
        ck = torch.load(last_p, map_location="cpu", weights_only=False)  # CPU: RNG states must be CPU ByteTensors; model/optimizer copy to device
        if ck["config_hash"] != chash:
            raise RuntimeError(f"checkpoint {last_p} was made with a different configuration ({ck['config_hash']} != {chash})")
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"]); sched.load_state_dict(ck["scheduler"])
        state.update(ck["state"]); set_rng_state(ck["rng"])
        log.info("resumed from %s at epoch %d", last_p, state["epoch"] + 1)

    write_json(run_dir / "run_info.json", {
        "config": cfg, "config_hash": chash, "env": env_snapshot(), "params": count_params(model), "amp_dtype": str(amp),
        "channel_mean": mean, "channel_std": std,
        "partition_sizes": {"train_images": len(tr_rows), "val_images": len(va_rows),
                            "train_patients": int(np.unique(store.patient_ids[tr_rows]).size),
                            "val_patients": int(np.unique(store.patient_ids[va_rows]).size)},
        "train_patient_ids": sorted(map(int, np.unique(store.patient_ids[tr_rows]))),
        "val_patient_ids": sorted(map(int, np.unique(store.patient_ids[va_rows])))})
    log.info("fold %d | model %s (%d params) | train %d img / %d patients | val %d img / %d patients | amp=%s | device=%s",
             f, cfg["model"]["name"], count_params(model), len(tr_rows), np.unique(store.patient_ids[tr_rows]).size,
             len(va_rows), np.unique(store.patient_ids[va_rows]).size, amp, device)

    max_val = cfg["smoke"]["max_val_images"]
    va_eval = va_rows if not max_val else va_rows[: int(max_val)]
    gen = torch.Generator(device=device)
    summary: dict[str, Any] = {}
    for epoch in range(state["epoch"] + 1, o["epochs"]):
        t0 = time.time()
        ds.set_epoch(epoch)
        gen.manual_seed(cfg["seed"] * 100_003 + epoch)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        losses = []
        for step, (x, y, _) in enumerate(dl):
            if cfg["smoke"]["max_train_steps"] and step >= cfg["smoke"]["max_train_steps"]:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            xb = pre(x, train=True, generator=gen)
            with torch.autocast(device_type=device.type, dtype=amp, enabled=amp is not None):
                logits = model(xb)
            loss = F.cross_entropy(logits.float(), y)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {epoch} step {step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if o["grad_clip"]:
                torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"])
            opt.step(); sched.step()
            losses.append(loss.item())
        logits_v, p_v = predict_store(model, store, va_eval, pre, d["eval_tile"], d["eval_batch_images"], device, amp)
        y_v = store.labels[va_eval]
        vm = binary_metrics(y_v, p_v)
        vloss = float(F.cross_entropy(torch.from_numpy(logits_v), torch.from_numpy(y_v).long()))
        rec = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": vloss, "val_auroc": vm["auroc"],
               "val_balanced_accuracy": vm["balanced_accuracy"], "val_sensitivity": vm["sensitivity"],
               "val_specificity": vm["specificity"], "lr": sched.get_last_lr()[0], "seconds": round(time.time() - t0, 1),
               "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3) if device.type == "cuda" else None}
        with open(run_dir / "metrics.jsonl", "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        log.info("epoch %d | train_loss %.4f | val_loss %.4f | val_auroc %s | bal_acc %.3f | %.1fs",
                 epoch, rec["train_loss"], vloss, f"{vm['auroc']:.4f}", vm["balanced_accuracy"], rec["seconds"])
        auroc = vm["auroc"] if np.isfinite(vm["auroc"]) else -1.0
        improved = auroc > state["best"] + 1e-6 or (abs(auroc - state["best"]) <= 1e-6 and vloss < state["best_loss"])
        if improved:
            state.update(best=auroc, best_loss=vloss, bad=0, best_epoch=epoch)
            _atomic_save({"model": model.state_dict(), "epoch": epoch, "val": rec, "config": cfg,
                          "channel_mean": mean, "channel_std": std}, best_p)
        else:
            state["bad"] += 1
        state["epoch"] = epoch
        _atomic_save({"model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sched.state_dict(),
                      "state": state, "rng": rng_state(), "config": cfg, "config_hash": chash,
                      "channel_mean": mean, "channel_std": std}, last_p)
        summary = {"last_epoch": epoch, "best_val_auroc": state["best"], "best_epoch": state.get("best_epoch"), "last": rec}
        if state["bad"] >= cfg["early_stopping"]["patience"]:
            log.info("early stopping at epoch %d (no val improvement for %d epochs)", epoch, state["bad"])
            break
    return summary
