"""Evaluate global and coin-localized reconstruction quality."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader
import yaml

from src.data.frame_dataset import CoinRunFrameDataset, manifest_from_split_dir
from src.models.autoencoder import build_autoencoder
from src.models.losses import ssim_per_image


def coin_mask(images: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    evaluation = config["evaluation"]
    red = images[:, 0] * 255.0
    green = images[:, 1] * 255.0
    blue = images[:, 2] * 255.0
    return (
        (red >= float(evaluation["coin_red_min"]))
        & (green >= float(evaluation["coin_green_min"]))
        & (blue <= float(evaluation["coin_blue_max"]))
    )


def _to_image(tensor: torch.Tensor, size: int = 256) -> Image.Image:
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.round(array * 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.NEAREST
    )


def _comparison_montage(
    samples: list[dict[str, Any]], mean_frame: torch.Tensor, output: Path
) -> None:
    cell = 256
    header = 28
    columns = ("original", "reconstruction", "absolute error", "mean baseline")
    canvas = Image.new("RGB", (cell * 4, header + (cell + header) * len(samples)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(columns):
        draw.text((column * cell + 5, 8), title, fill="black")
    for row, sample in enumerate(samples):
        y = header + row * (cell + header)
        target = sample["target"]
        reconstruction = sample["reconstruction"]
        error = (target - reconstruction).abs()
        for column, image in enumerate((target, reconstruction, error, mean_frame)):
            canvas.paste(_to_image(image, cell), (column * cell, y))
        draw.text(
            (5, y + cell + 7),
            f"{sample['episode_id']} frame={sample['frame_index']} coin={sample['coin']}",
            fill="black",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _coin_roi_montage(
    sample: dict[str, Any], bbox: list[int], mean_frame: torch.Tensor, output: Path
) -> None:
    x1, y1, x2, y2 = bbox
    margin = 5
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(64, x2 + margin), min(64, y2 + margin)
    target = sample["target"][:, y1:y2, x1:x2]
    reconstruction = sample["reconstruction"][:, y1:y2, x1:x2]
    baseline = mean_frame[:, y1:y2, x1:x2]
    error = (target - reconstruction).abs()
    titles = ("original ROI", "reconstruction ROI", "absolute error", "mean baseline ROI")
    cell = 256
    canvas = Image.new("RGB", (cell * 4, cell + 32), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (title, tensor) in enumerate(
        zip(titles, (target, reconstruction, error, baseline))
    ):
        draw.text((column * cell + 5, 8), title, fill="black")
        canvas.paste(_to_image(tensor, cell), (column * cell, 32))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def evaluate(
    checkpoint_path: Path,
    eval_dir: Path,
    output_path: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_autoencoder(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    mean_frame = checkpoint["train_mean_frame"].float()
    dataset = CoinRunFrameDataset(manifest_from_split_dir(eval_dir))
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)

    totals = {
        "model_mae": 0.0,
        "model_mse": 0.0,
        "model_ssim": 0.0,
        "baseline_mae": 0.0,
        "baseline_mse": 0.0,
        "baseline_ssim": 0.0,
    }
    count = 0
    coin_rows: list[dict[str, Any]] = []
    montage_candidates: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            target = batch["image"].to(device)
            reconstruction = model(target)
            baseline = mean_frame.to(device).unsqueeze(0).expand_as(target)
            model_error = reconstruction - target
            baseline_error = baseline - target
            size = len(target)
            totals["model_mae"] += float(model_error.abs().mean(dim=(1, 2, 3)).sum())
            totals["model_mse"] += float(model_error.square().mean(dim=(1, 2, 3)).sum())
            totals["model_ssim"] += float(ssim_per_image(reconstruction, target).sum())
            totals["baseline_mae"] += float(baseline_error.abs().mean(dim=(1, 2, 3)).sum())
            totals["baseline_mse"] += float(baseline_error.square().mean(dim=(1, 2, 3)).sum())
            totals["baseline_ssim"] += float(ssim_per_image(baseline, target).sum())
            masks = coin_mask(target, config)
            reconstructed_masks = coin_mask(reconstruction, config)
            minimum = int(config["evaluation"]["coin_min_pixels"])
            padding = int(config["evaluation"]["coin_roi_padding"])
            for index in range(size):
                has_coin = int(masks[index].sum()) >= minimum
                candidate = {
                    "target": target[index].cpu(),
                    "reconstruction": reconstruction[index].cpu(),
                    "episode_id": batch["episode_id"][index],
                    "frame_index": int(batch["frame_index"][index]),
                    "coin": has_coin,
                }
                if has_coin or len(montage_candidates) < 5:
                    montage_candidates.append(candidate)
                if not has_coin:
                    continue
                ys, xs = torch.where(masks[index])
                x1 = max(0, int(xs.min()) - padding)
                y1 = max(0, int(ys.min()) - padding)
                x2 = min(64, int(xs.max()) + padding + 1)
                y2 = min(64, int(ys.max()) + padding + 1)
                target_roi = target[index, :, y1:y2, x1:x2]
                reconstruction_roi = reconstruction[index, :, y1:y2, x1:x2]
                baseline_roi = baseline[index, :, y1:y2, x1:x2]
                pixel_mask = masks[index].unsqueeze(0).expand(3, -1, -1)
                model_coin_mae = (reconstruction[index] - target[index]).abs()[pixel_mask].mean()
                baseline_coin_mae = (baseline[index] - target[index]).abs()[pixel_mask].mean()
                recall = (reconstructed_masks[index] & masks[index]).sum() / masks[index].sum()
                coin_rows.append(
                    {
                        "episode_id": batch["episode_id"][index],
                        "frame_index": int(batch["frame_index"][index]),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "coin_pixels": int(masks[index].sum()),
                        "model_roi_mae": float((reconstruction_roi - target_roi).abs().mean()),
                        "baseline_roi_mae": float((baseline_roi - target_roi).abs().mean()),
                        "model_coin_pixel_mae": float(model_coin_mae),
                        "baseline_coin_pixel_mae": float(baseline_coin_mae),
                        "coin_color_recall": float(recall),
                    }
                )
            count += size

    averages = {key: value / count for key, value in totals.items()}
    averages["model_psnr_db"] = 10.0 * math.log10(1.0 / averages["model_mse"])
    averages["baseline_psnr_db"] = 10.0 * math.log10(1.0 / averages["baseline_mse"])
    coin_summary: dict[str, Any] = {"frames": len(coin_rows), "rows": coin_rows}
    for key in (
        "model_roi_mae",
        "baseline_roi_mae",
        "model_coin_pixel_mae",
        "baseline_coin_pixel_mae",
        "coin_color_recall",
    ):
        coin_summary[f"mean_{key}"] = (
            float(np.mean([row[key] for row in coin_rows])) if coin_rows else None
        )
    model_roi = coin_summary.get("mean_model_roi_mae")
    baseline_roi = coin_summary.get("mean_baseline_roi_mae")
    coin_summary["roi_mae_improvement_fraction"] = (
        1.0 - model_roi / baseline_roi
        if model_roi is not None and baseline_roi and baseline_roi > 0
        else None
    )
    acceptance = {
        "psnr_beats_baseline_by_3db": averages["model_psnr_db"] >= averages["baseline_psnr_db"] + 3.0,
        "ssim_beats_baseline_by_0_10": averages["model_ssim"] >= averages["baseline_ssim"] + 0.10,
        "coin_roi_mae_improves_by_25_percent": (
            coin_summary["roi_mae_improvement_fraction"] is not None
            and coin_summary["roi_mae_improvement_fraction"] >= 0.25
        ),
        "coin_color_recall_at_least_90_percent": (
            coin_summary.get("mean_coin_color_recall") is not None
            and coin_summary["mean_coin_color_recall"] >= 0.90
        ),
        "coin_evidence_has_at_least_10_frames": len(coin_rows) >= 10,
    }
    unique_seeds = sorted({int(episode["level_seed"]) for episode in dataset.episodes})
    limitations = [
        "Coin regions are detected heuristically from yellow pixels in the ground-truth frame."
    ]
    if len(coin_rows) < 10:
        limitations.append(
            "Validation contains fewer than 10 coin-visible frames, so localized metrics are fragile."
        )
    if len(unique_seeds) == 1:
        limitations.append(
            f"The evaluation manifest contains only level seed {unique_seeds[0]}."
        )
    report = {
        "checkpoint": checkpoint_path.as_posix(),
        "eval_manifest": manifest_from_split_dir(eval_dir).as_posix(),
        "device": str(device),
        "frames": count,
        "global": averages,
        "coin": coin_summary,
        "acceptance": acceptance,
        "go": all(acceptance.values()),
        "evaluation_seed_count": len(unique_seeds),
        "limitations": limitations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    selected = montage_candidates[:5]
    coin_candidates = [sample for sample in montage_candidates if sample["coin"]]
    if coin_candidates and not any(sample["coin"] for sample in selected):
        selected[-1] = coin_candidates[0]
    _comparison_montage(selected, mean_frame, output_path.parent / "figures" / "reconstructions.png")
    if coin_rows and coin_candidates:
        _coin_roi_montage(
            coin_candidates[0],
            coin_rows[0]["bbox_xyxy"],
            mean_frame,
            output_path.parent / "figures" / "coin_roi_comparison.png",
        )
    with (output_path.parent / "coin_rois.jsonl").open("w", encoding="utf-8") as stream:
        for row in coin_rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    report = evaluate(Path(args.checkpoint), Path(args.eval_dir), Path(args.out), device)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
