"""Independent, reproducible review of autoencoder reconstructions.

This module intentionally does not import the implementation's metric or frame-dataset code.
It reads the manifest directly, recomputes PSNR/SSIM and localized coin metrics, and selects
its own deterministic visual sample.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.models.autoencoder import build_autoencoder


def yellow_object_mask(images: torch.Tensor, config: dict[str, Any]) -> torch.Tensor:
    """Detect the known yellow CoinRun coin pixels from normalized NCHW frames."""
    limits = config["evaluation"]
    rgb = images * 255.0
    return (
        (rgb[:, 0] >= float(limits["coin_red_min"]))
        & (rgb[:, 1] >= float(limits["coin_green_min"]))
        & (rgb[:, 2] <= float(limits["coin_blue_max"]))
    )


def independent_ssim(
    prediction: torch.Tensor, target: torch.Tensor, window_size: int = 7
) -> torch.Tensor:
    """Recompute SSIM directly from its local statistics, returning one value per image."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("SSIM expects equal NCHW tensors")
    if window_size % 2 != 1:
        raise ValueError("SSIM window size must be odd")
    pad = window_size // 2
    mean_x = F.avg_pool2d(prediction, window_size, 1, pad)
    mean_y = F.avg_pool2d(target, window_size, 1, pad)
    variance_x = F.avg_pool2d(prediction.square(), window_size, 1, pad) - mean_x.square()
    variance_y = F.avg_pool2d(target.square(), window_size, 1, pad) - mean_y.square()
    covariance = F.avg_pool2d(prediction * target, window_size, 1, pad) - mean_x * mean_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        ((mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2)).clamp_min(
            1e-12
        )
    )
    return score.mean(dim=(1, 2, 3))


class ManifestFrames(Dataset[dict[str, Any]]):
    """Minimal independent manifest reader with a small episode cache."""

    def __init__(self, split: str | Path, cache_episodes: int = 8) -> None:
        split = Path(split)
        self.manifest = split / "manifest.jsonl" if split.is_dir() else split
        records = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise ValueError(f"empty manifest: {self.manifest}")
        self.episodes: list[dict[str, Any]] = []
        self.index: list[tuple[int, int]] = []
        for episode_index, record in enumerate(records):
            path = Path(record["path"])
            if path.suffix == ".h5":
                with h5py.File(path, "r") as archive:
                    length = int(archive["observations"].shape[0])
            elif path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as archive:
                    length = int(archive["episode_length"])
            else:
                raise ValueError(f"unsupported episode: {path}")
            if int(record.get("frames", length)) != length:
                raise ValueError(f"manifest length mismatch: {path}")
            self.episodes.append(
                {
                    "episode_id": str(record["episode_id"]),
                    "level_seed": int(record["level_seed"]),
                    "path": path,
                    "length": length,
                }
            )
            self.index.extend((episode_index, frame_index) for frame_index in range(length))
        self._cache_limit = cache_episodes
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.index)

    def _episode(self, index: int) -> np.ndarray:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        metadata = self.episodes[index]
        path: Path = metadata["path"]
        if path.suffix == ".h5":
            with h5py.File(path, "r") as archive:
                frames = archive["observations"][...]
        else:
            with np.load(path, allow_pickle=False) as archive:
                frames = archive["observations"]
        if frames.dtype != np.uint8 or frames.shape != (metadata["length"], 64, 64, 3):
            raise ValueError(f"invalid observation payload: {path}")
        self._cache[index] = frames
        while len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)
        return frames

    def __getitem__(self, item: int) -> dict[str, Any]:
        episode_index, frame_index = self.index[item]
        metadata = self.episodes[episode_index]
        frame = self._episode(episode_index)[frame_index]
        return {
            "image": torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div_(255.0),
            "episode_id": metadata["episode_id"],
            "frame_index": frame_index,
            "level_seed": metadata["level_seed"],
        }


def _stable_score(episode_id: str, frame_index: int) -> int:
    payload = f"{episode_id}:{frame_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _reservoir_add(
    heap: list[tuple[int, int, dict[str, Any]]], candidate: dict[str, Any], limit: int
) -> None:
    score = _stable_score(candidate["episode_id"], candidate["frame_index"])
    serial = int(candidate["global_index"])
    entry = (-score, -serial, candidate)
    if len(heap) < limit:
        heapq.heappush(heap, entry)
    elif entry > heap[0]:
        heapq.heapreplace(heap, entry)


def _embedding(image: torch.Tensor) -> torch.Tensor:
    return F.adaptive_avg_pool2d(image.unsqueeze(0), (8, 8)).flatten().float()


def choose_diverse(candidates: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Deterministic farthest-first selection over low-resolution RGB appearance."""
    if count <= 0 or not candidates:
        return []
    ordered = sorted(candidates, key=lambda row: (row["episode_id"], row["frame_index"]))
    embeddings = torch.stack([_embedding(row["target"]) for row in ordered])
    selected = [int(torch.argmax(embeddings.var(dim=1)))]
    distances = torch.cdist(embeddings, embeddings[selected]).squeeze(1)
    while len(selected) < min(count, len(ordered)):
        distances[selected] = -1
        next_index = int(torch.argmax(distances))
        selected.append(next_index)
        distances = torch.minimum(
            distances, torch.cdist(embeddings, embeddings[next_index : next_index + 1]).squeeze(1)
        )
    return [ordered[index] for index in selected]


def _pil(tensor: torch.Tensor, size: int = 192) -> Image.Image:
    array = tensor.clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray(np.rint(array * 255).astype(np.uint8)).resize(
        (size, size), Image.Resampling.NEAREST
    )


def save_montage(rows: list[dict[str, Any]], output: Path, *, show_coin: bool) -> None:
    columns = ["original", "reconstruction", "absolute error"]
    if show_coin:
        columns.append("coin ROI: original / reconstruction")
    cell, header, label = 192, 26, 24
    canvas = Image.new("RGB", (cell * len(columns), header + (cell + label) * len(rows)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(columns):
        draw.text((column * cell + 4, 7), title, fill="black")
    for row_index, row in enumerate(rows):
        y = header + row_index * (cell + label)
        original = row["target"]
        reconstruction = row["reconstruction"]
        images = [original, reconstruction, (original - reconstruction).abs()]
        if show_coin:
            x1, y1, x2, y2 = row["bbox_xyxy"]
            margin = 4
            x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
            x2, y2 = min(64, x2 + margin), min(64, y2 + margin)
            left = _pil(original[:, y1:y2, x1:x2], cell // 2)
            right = _pil(reconstruction[:, y1:y2, x1:x2], cell // 2)
            joined = Image.new("RGB", (cell, cell), "white")
            joined.paste(left, (0, cell // 4))
            joined.paste(right, (cell // 2, cell // 4))
            images.append(joined)
        for column, image in enumerate(images):
            rendered = image if isinstance(image, Image.Image) else _pil(image, cell)
            canvas.paste(rendered, (column * cell, y))
        draw.text(
            (4, y + cell + 5),
            f"{row['episode_id']} frame={row['frame_index']}",
            fill="black",
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def _reported_metrics(checkpoint: Path) -> tuple[Path, dict[str, Any] | None]:
    metrics_path = checkpoint.parent.parent / "metrics.json"
    if not metrics_path.is_file():
        return metrics_path, None
    return metrics_path, json.loads(metrics_path.read_text(encoding="utf-8"))


def review(
    checkpoint_path: Path,
    eval_dir: Path,
    n_samples: int,
    output: Path,
    device: torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    model = build_autoencoder(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    dataset = ManifestFrames(eval_dir)
    loader = DataLoader(dataset, batch_size=int(config["training"]["batch_size"]), shuffle=False)
    totals = {"absolute_error": 0.0, "squared_error": 0.0, "ssim": 0.0}
    count = 0
    coin_count = 0
    coin_roi_mae: list[float] = []
    coin_pixel_mae: list[float] = []
    coin_recall: list[float] = []
    coin_heap: list[tuple[int, int, dict[str, Any]]] = []
    context_heap: list[tuple[int, int, dict[str, Any]]] = []
    minimum_pixels = int(config["evaluation"]["coin_min_pixels"])
    padding = int(config["evaluation"]["coin_roi_padding"])
    with torch.no_grad():
        for batch in loader:
            target = batch["image"].to(device)
            reconstruction = model(target)
            difference = reconstruction - target
            size = len(target)
            totals["absolute_error"] += float(difference.abs().mean(dim=(1, 2, 3)).sum())
            totals["squared_error"] += float(difference.square().mean(dim=(1, 2, 3)).sum())
            totals["ssim"] += float(
                independent_ssim(
                    reconstruction,
                    target,
                    int(config["loss"]["ssim_window_size"]),
                ).sum()
            )
            masks = yellow_object_mask(target, config)
            reconstruction_masks = yellow_object_mask(reconstruction, config)
            for index in range(size):
                candidate = {
                    "episode_id": str(batch["episode_id"][index]),
                    "frame_index": int(batch["frame_index"][index]),
                    "level_seed": int(batch["level_seed"][index]),
                    "global_index": count + index,
                    "target": target[index].cpu(),
                    "reconstruction": reconstruction[index].cpu(),
                }
                _reservoir_add(context_heap, candidate, 512)
                if int(masks[index].sum()) < minimum_pixels:
                    continue
                ys, xs = torch.where(masks[index])
                x1 = max(0, int(xs.min()) - padding)
                y1 = max(0, int(ys.min()) - padding)
                x2 = min(64, int(xs.max()) + padding + 1)
                y2 = min(64, int(ys.max()) + padding + 1)
                pixel_mask = masks[index].unsqueeze(0).expand(3, -1, -1)
                coin_count += 1
                coin_roi_mae.append(
                    float((reconstruction[index, :, y1:y2, x1:x2] - target[index, :, y1:y2, x1:x2]).abs().mean())
                )
                coin_pixel_mae.append(float(difference[index].abs()[pixel_mask].mean()))
                coin_recall.append(
                    float((reconstruction_masks[index] & masks[index]).sum() / masks[index].sum())
                )
                candidate["bbox_xyxy"] = [x1, y1, x2, y2]
                _reservoir_add(coin_heap, candidate, 512)
            count += size

    coin_candidates = [entry[2] for entry in coin_heap]
    context_candidates = [entry[2] for entry in context_heap]
    selected_coin = choose_diverse(coin_candidates, n_samples)
    selected_context = choose_diverse(context_candidates, n_samples)
    mse = totals["squared_error"] / count
    independent = {
        "frames": count,
        "mae": totals["absolute_error"] / count,
        "mse": mse,
        "psnr_db": 10.0 * math.log10(1.0 / mse),
        "ssim": totals["ssim"] / count,
        "coin_visible_frames": coin_count,
        "mean_coin_roi_mae": float(np.mean(coin_roi_mae)) if coin_roi_mae else None,
        "mean_coin_pixel_mae": float(np.mean(coin_pixel_mae)) if coin_pixel_mae else None,
        "mean_coin_color_recall": float(np.mean(coin_recall)) if coin_recall else None,
    }
    metrics_path, reported = _reported_metrics(checkpoint_path)
    comparison: dict[str, Any] = {"metrics_path": metrics_path.as_posix(), "available": reported is not None}
    if reported is not None:
        pairs = {
            "psnr_db": reported["global"]["model_psnr_db"],
            "ssim": reported["global"]["model_ssim"],
            "mean_coin_roi_mae": reported["coin"]["mean_model_roi_mae"],
            "mean_coin_color_recall": reported["coin"]["mean_coin_color_recall"],
        }
        comparison["differences"] = {
            key: (None if independent[key] is None or value is None else independent[key] - value)
            for key, value in pairs.items()
        }
        comparison["reproduced_within_1e_6"] = all(
            difference is None or abs(difference) <= 1e-6
            for difference in comparison["differences"].values()
        )
    output.mkdir(parents=True, exist_ok=True)
    if selected_coin:
        save_montage(selected_coin, output / "coin_samples.png", show_coin=True)
    save_montage(selected_context, output / "context_samples.png", show_coin=False)
    sample_rows = []
    for row in selected_coin:
        mask = yellow_object_mask(row["target"].unsqueeze(0), config)[0]
        reconstructed_mask = yellow_object_mask(row["reconstruction"].unsqueeze(0), config)[0]
        x1, y1, x2, y2 = row["bbox_xyxy"]
        sample_rows.append(
            {
                "episode_id": row["episode_id"],
                "frame_index": row["frame_index"],
                "level_seed": row["level_seed"],
                "bbox_xyxy": row["bbox_xyxy"],
                "coin_pixels": int(mask.sum()),
                "coin_color_recall": float((reconstructed_mask & mask).sum() / mask.sum()),
                "coin_roi_mae": float(
                    (row["reconstruction"][:, y1:y2, x1:x2] - row["target"][:, y1:y2, x1:x2]).abs().mean()
                ),
            }
        )
    (output / "selected_coin_samples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in sample_rows), encoding="utf-8"
    )
    context_rows = [
        {
            "episode_id": row["episode_id"],
            "frame_index": row["frame_index"],
            "level_seed": row["level_seed"],
        }
        for row in selected_context
    ]
    (output / "selected_context_samples.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in context_rows),
        encoding="utf-8",
    )
    report = {
        "checkpoint": checkpoint_path.as_posix(),
        "checkpoint_global_step": int(checkpoint.get("global_step", -1)),
        "eval_manifest": dataset.manifest.as_posix(),
        "device": str(device),
        "requested_coin_samples": n_samples,
        "selected_coin_samples": len(selected_coin),
        "coin_sample_requirement_met": len(selected_coin) >= n_samples,
        "independent_metrics": independent,
        "reported_comparison": comparison,
        "visual_evidence": {
            "coin_samples": (output / "coin_samples.png").as_posix(),
            "context_samples": (output / "context_samples.png").as_posix(),
        },
        "automatic_flags": {
            "insufficient_coin_frames": coin_count < n_samples,
            "systematic_coin_color_loss": (
                independent["mean_coin_color_recall"] is not None
                and independent["mean_coin_color_recall"] < 0.5
            ),
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval-dir", required=True, type=Path)
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_samples < 10:
        raise ValueError("the independent review requires at least 10 samples")
    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    report = review(args.checkpoint, args.eval_dir, args.n_samples, args.out, device)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
