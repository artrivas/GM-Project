"""Frame dataset for legacy HDF5 and active external NPZ manifests."""

from __future__ import annotations

import argparse
import json
import random
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class CoinRunFrameDataset(Dataset[dict[str, Any]]):
    """Expose frames without forcing the 163k-frame dataset into memory."""

    def __init__(
        self,
        manifest: str | Path,
        *,
        eager_frame_limit: int = 20_000,
        cache_episodes: int = 8,
    ) -> None:
        self.manifest = Path(manifest)
        if not self.manifest.is_file():
            raise FileNotFoundError(f"split manifest does not exist: {self.manifest}")
        records = [
            json.loads(line)
            for line in self.manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise ValueError(f"empty split manifest: {self.manifest}")

        self.episodes: list[dict[str, Any]] = []
        self.entries: list[tuple[int, int]] = []
        self.episode_ranges: list[range] = []
        weights: list[float] = []
        position = 0
        for episode_index, record in enumerate(records):
            path = Path(record["path"])
            if not path.is_file():
                raise FileNotFoundError(path)
            if path.suffix == ".h5":
                with h5py.File(path, "r") as episode:
                    shape = episode["observations"].shape
                    dtype = episode["observations"].dtype
                    episode_id = str(episode.attrs["episode_id"])
                storage_format = "hdf5_v1"
            elif path.suffix == ".npz":
                with np.load(path, allow_pickle=False) as episode:
                    length = int(episode["episode_length"])
                    shape = (length, 64, 64, 3)
                    dtype = np.dtype(np.uint8)
                episode_id = str(record["episode_id"])
                storage_format = "npz_external_v1"
            else:
                raise ValueError(f"unsupported episode format: {path}")
            if dtype != np.uint8 or shape[1:] != (64, 64, 3):
                raise ValueError(
                    f"expected uint8 observations shaped (T,64,64,3), got {dtype} {shape} in {path}"
                )
            if episode_id != str(record["episode_id"]):
                raise ValueError(f"manifest/episode ID mismatch: {path}")
            length = int(shape[0])
            if int(record.get("frames", length)) != length:
                raise ValueError(f"manifest frame count mismatch: {path}")
            group_size = max(1, int(record.get("duplicate_group_size", 1)))
            self.episodes.append(
                {
                    "episode_id": episode_id,
                    "path": path,
                    "length": length,
                    "level_seed": int(record["level_seed"]),
                    "storage_format": storage_format,
                }
            )
            self.entries.extend((episode_index, frame_index) for frame_index in range(length))
            self.episode_ranges.append(range(position, position + length))
            weights.extend([1.0 / group_size] * length)
            position += length

        self.sample_weights = torch.tensor(weights, dtype=torch.double)
        self._cache_limit = max(1, cache_episodes)
        self._episode_cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self.frames: np.ndarray | None = None
        if len(self.entries) <= eager_frame_limit:
            self.frames = np.concatenate(
                [self._read_episode(index) for index in range(len(self.episodes))], axis=0
            )
            self._episode_cache.clear()

    @property
    def lazy(self) -> bool:
        return self.frames is None

    @property
    def storage_formats(self) -> list[str]:
        return sorted({episode["storage_format"] for episode in self.episodes})

    def __len__(self) -> int:
        return len(self.entries)

    def _read_episode(self, episode_index: int) -> np.ndarray:
        cached = self._episode_cache.get(episode_index)
        if cached is not None:
            self._episode_cache.move_to_end(episode_index)
            return cached
        episode = self.episodes[episode_index]
        path: Path = episode["path"]
        if path.suffix == ".h5":
            with h5py.File(path, "r") as archive:
                observations = archive["observations"][...]
        else:
            with np.load(path, allow_pickle=False) as archive:
                observations = archive["observations"]
        if observations.dtype != np.uint8 or observations.shape != (
            episode["length"],
            64,
            64,
            3,
        ):
            raise ValueError(f"observation payload changed after manifest audit: {path}")
        self._episode_cache[episode_index] = observations
        self._episode_cache.move_to_end(episode_index)
        while len(self._episode_cache) > self._cache_limit:
            self._episode_cache.popitem(last=False)
        return observations

    def __getitem__(self, index: int) -> dict[str, Any]:
        episode_index, frame_index = self.entries[index]
        episode = self.episodes[episode_index]
        if self.frames is not None:
            frame_array = self.frames[index]
        else:
            frame_array = self._read_episode(episode_index)[frame_index]
        frame = torch.from_numpy(frame_array.copy()).permute(2, 0, 1)
        return {
            "image": frame.to(dtype=torch.float32).div_(255.0),
            "episode_id": episode["episode_id"],
            "frame_index": frame_index,
            "path": episode["path"].as_posix(),
            "level_seed": episode["level_seed"],
        }

    def mean_frame(self) -> torch.Tensor:
        if self.frames is not None:
            mean = self.frames.astype(np.float64).mean(axis=0)
        else:
            total = np.zeros((64, 64, 3), dtype=np.float64)
            count = 0
            for episode_index in range(len(self.episodes)):
                frames = self._read_episode(episode_index)
                total += frames.sum(axis=0, dtype=np.float64)
                count += len(frames)
            mean = total / count
        return torch.from_numpy(mean).permute(2, 0, 1).float().div_(255.0)

    def sampled_pixel_range(self, maximum_frames: int = 256) -> tuple[int, int]:
        indices = np.linspace(0, len(self) - 1, min(maximum_frames, len(self)), dtype=int)
        minimum, maximum = 255, 0
        for index in indices:
            image = self[int(index)]["image"]
            minimum = min(minimum, int(round(float(image.min()) * 255)))
            maximum = max(maximum, int(round(float(image.max()) * 255)))
        return minimum, maximum


class EpisodeGroupedSampler(Sampler[int]):
    """Shuffle episodes while keeping their frames contiguous for NPZ cache locality."""

    def __init__(self, dataset: CoinRunFrameDataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = seed
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        order = list(range(len(self.dataset.episode_ranges)))
        random.Random(self.seed + self.epoch).shuffle(order)
        self.epoch += 1
        for episode_index in order:
            yield from self.dataset.episode_ranges[episode_index]

    def __len__(self) -> int:
        return len(self.dataset)


def manifest_from_split_dir(path: str | Path) -> Path:
    split_dir = Path(path)
    manifest = split_dir / "manifest.jsonl" if split_dir.is_dir() else split_dir
    if not manifest.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--audit-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = CoinRunFrameDataset(args.train_manifest)
    val = CoinRunFrameDataset(args.val_manifest)
    minimum, maximum = train.sampled_pixel_range()
    report = {
        "train_frames": len(train),
        "val_frames": len(val),
        "train_episodes": len(train.episodes),
        "val_episodes": len(val.episodes),
        "storage_formats": train.storage_formats,
        "lazy_loading": train.lazy,
        "dtype_after_normalization": str(train[0]["image"].dtype),
        "shape": list(train[0]["image"].shape),
        "sampled_pixel_minimum": minimum,
        "sampled_pixel_maximum": maximum,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
