"""Train the compact CoinRun autoencoder on Plan 2 manifests."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import yaml

from src.data.frame_dataset import (
    CoinRunFrameDataset,
    EpisodeGroupedSampler,
    manifest_from_split_dir,
)
from src.models.autoencoder import build_autoencoder
from src.models.losses import ReconstructionLoss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_val_loss: float,
    config: dict[str, Any],
    mean_frame: torch.Tensor,
    global_step: int,
) -> dict[str, Any]:
    return {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "global_step": global_step,
        "config": config,
        "train_mean_frame": mean_frame.cpu(),
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }


def evaluate_loss(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: ReconstructionLoss,
    device: torch.device,
) -> dict[str, float]:
    totals = {
        "loss": 0.0,
        "l1": 0.0,
        "ssim": 0.0,
        "coin_roi_l1": 0.0,
        "coin_pixel_l1": 0.0,
        "coin_mask_bce": 0.0,
    }
    count = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            values = criterion(model(images), images)
            size = len(images)
            for key in totals:
                totals[key] += float(values[key]) * size
            count += size
    return {key: value / count for key, value in totals.items()}


def train(args: argparse.Namespace) -> dict[str, Any]:
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    training = config["training"]
    seed = int(training["seed"])
    seed_everything(seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    train_manifest = manifest_from_split_dir(args.train_dir)
    val_manifest = manifest_from_split_dir(args.val_dir)
    marker = train_manifest.parent.parent / "DO_NOT_USE.json"
    if marker.exists() and not bool(training.get("allow_diagnostic_split", False)):
        raise RuntimeError(f"diagnostic split is blocked by {marker}")

    train_dataset = CoinRunFrameDataset(train_manifest)
    val_dataset = CoinRunFrameDataset(val_manifest)
    if args.subset_frames:
        train_dataset_for_loader = Subset(
            train_dataset, range(min(args.subset_frames, len(train_dataset)))
        )
        sampler = None
        shuffle = True
    elif train_dataset.lazy:
        train_dataset_for_loader = train_dataset
        sampler = EpisodeGroupedSampler(train_dataset, seed)
        shuffle = False
    else:
        train_dataset_for_loader = train_dataset
        generator = torch.Generator().manual_seed(seed)
        sampler = WeightedRandomSampler(
            train_dataset.sample_weights,
            num_samples=len(train_dataset),
            replacement=True,
            generator=generator,
        )
        shuffle = False
    batch_size = int(training["batch_size"])
    train_loader = DataLoader(
        train_dataset_for_loader,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle,
        num_workers=int(training["num_workers"]),
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_autoencoder(config).to(device)
    criterion = ReconstructionLoss(
        l1_weight=float(config["loss"]["l1_weight"]),
        ssim_weight=float(config["loss"]["ssim_weight"]),
        window_size=int(config["loss"]["ssim_window_size"]),
        coin_roi_weight=float(config["loss"].get("coin_roi_weight", 1.0)),
        coin_roi_loss_weight=float(config["loss"].get("coin_roi_loss_weight", 0.0)),
        coin_pixel_loss_weight=float(config["loss"].get("coin_pixel_loss_weight", 0.0)),
        coin_mask_loss_weight=float(config["loss"].get("coin_mask_loss_weight", 0.0)),
        coin_mask_negative_weight=float(config["loss"].get("coin_mask_negative_weight", 1.0)),
        coin_hard_negative_fraction=float(
            config["loss"].get("coin_hard_negative_fraction", 0.0)
        ),
        coin_roi_size=int(config["loss"].get("coin_roi_size", 7)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    use_amp = device.type == "cuda" and bool(training.get("amp", True))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model_state"])
        optimizer.load_state_dict(saved["optimizer_state"])
        scaler.load_state_dict(saved["scaler_state"])
        start_epoch = int(saved["epoch"]) + 1
        global_step = int(saved["global_step"])
        best_val_loss = float(saved["best_val_loss"])
        if args.reset_best:
            best_val_loss = float("inf")

    train_mean_frame = train_dataset.mean_frame()
    output = Path(args.out)
    checkpoints = output / "checkpoints"
    output.mkdir(parents=True, exist_ok=True)
    history_path = output / "history.json"
    history: list[dict[str, Any]] = []
    if args.resume and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
    no_improvement = 0
    started = time.perf_counter()
    stop = False
    for epoch in range(start_epoch, int(training["epochs"])):
        model.train()
        totals = {
            "loss": 0.0,
            "l1": 0.0,
            "ssim": 0.0,
            "coin_roi_l1": 0.0,
            "coin_pixel_l1": 0.0,
            "coin_mask_bce": 0.0,
        }
        seen = 0
        for batch in train_loader:
            images = batch["image"].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                values = criterion(model(images), images)
            scaler.scale(values["loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            scaler.step(optimizer)
            scaler.update()
            size = len(images)
            for key in totals:
                totals[key] += float(values[key].detach()) * size
            seen += size
            global_step += 1
            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break
        train_metrics = {key: value / seen for key, value in totals.items()}
        val_metrics = evaluate_loss(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": val_metrics,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(row)
        improved = val_metrics["loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["loss"]
            no_improvement = 0
        else:
            no_improvement += 1
        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            best_val_loss=best_val_loss,
            config=config,
            mean_frame=train_mean_frame,
            global_step=global_step,
        )
        _atomic_torch_save(payload, checkpoints / "last.pt")
        if improved:
            _atomic_torch_save(payload, checkpoints / "best.pt")
        history_path.write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(row, sort_keys=True), flush=True)
        if stop or no_improvement >= int(training["early_stopping_patience"]):
            break

    summary = {
        "device": str(device),
        "gpu_used": device.type == "cuda",
        "amp_used": use_amp,
        "train_frames": len(train_dataset),
        "val_frames": len(val_dataset),
        "epochs_completed": len(history),
        "global_steps": global_step,
        "best_val_loss": best_val_loss,
        "elapsed_seconds": time.perf_counter() - started,
        "diagnostic_split_acknowledged": marker.exists(),
        "checkpoint": (checkpoints / "best.pt").as_posix(),
    }
    (output / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-dir", required=True)
    parser.add_argument("--val-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--subset-frames", type=int)
    parser.add_argument("--reset-best", action="store_true")
    return parser.parse_args()


def main() -> None:
    summary = train(parse_args())
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
