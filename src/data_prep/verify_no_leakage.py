"""Verify split manifests directly, without using the Plan 2 splitter."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py


SPLITS = ("train", "val", "test")
FINGERPRINT_DATASETS = (
    "observations",
    "next_observations",
    "next_observation_valid",
    "actions",
    "rewards",
    "dones",
    "truncateds",
    "events",
)


def _fingerprint(episode: h5py.File) -> str:
    digest = hashlib.sha256()
    for name in FINGERPRINT_DATASETS:
        array = episode[name][...]
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def verify(splits_dir: Path) -> dict[str, Any]:
    ids: dict[str, set[str]] = {}
    fingerprints: dict[str, set[str]] = {}
    counts: dict[str, dict[str, int]] = {}
    errors: list[str] = []

    for split in SPLITS:
        manifest = splits_dir / split / "manifest.jsonl"
        if not manifest.is_file():
            errors.append(f"missing_manifest:{manifest}")
            ids[split] = set()
            fingerprints[split] = set()
            counts[split] = {}
            continue
        split_ids: list[str] = []
        split_fingerprints: list[str] = []
        event_by_fingerprint: dict[str, tuple[bool, bool, bool]] = {}
        metrics = Counter()
        for line_number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            item = json.loads(line)
            path = Path(item["path"])
            if not path.is_file():
                errors.append(f"missing_episode:{split}:{line_number}:{path}")
                continue
            with h5py.File(path, "r") as episode:
                manifest_id = str(item["episode_id"])
                hdf5_id = str(episode.attrs["episode_id"])
                if manifest_id != hdf5_id:
                    errors.append(f"id_mismatch:{split}:{manifest_id}:{hdf5_id}")
                split_ids.append(hdf5_id)
                fingerprint = _fingerprint(episode)
                split_fingerprints.append(fingerprint)
                event_by_fingerprint[fingerprint] = (
                    bool(episode.attrs.get("win", 0)),
                    bool(episode.attrs.get("death", 0)),
                    bool(episode.attrs.get("truncated", 0)),
                )
                metrics.update(
                    episodes=1,
                    frames=int(episode["observations"].shape[0]),
                    victories=int(episode.attrs.get("win", 0)),
                    deaths=int(episode.attrs.get("death", 0)),
                    truncated=int(episode.attrs.get("truncated", 0)),
                )
        if len(split_ids) != len(set(split_ids)):
            errors.append(f"duplicate_episode_id_within:{split}")
        ids[split] = set(split_ids)
        fingerprints[split] = set(split_fingerprints)
        unique_events = Counter()
        for won, died, truncated in event_by_fingerprint.values():
            unique_events.update(
                unique_victories=int(won),
                unique_deaths=int(died),
                unique_truncated=int(truncated),
            )
        counts[split] = {
            **dict(metrics),
            **dict(unique_events),
            "unique_content_trajectories": len(event_by_fingerprint),
        }

    id_overlap: dict[str, list[str]] = {}
    fingerprint_overlap: dict[str, list[str]] = {}
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            pair = f"{left}-{right}"
            id_overlap[pair] = sorted(ids[left] & ids[right])
            fingerprint_overlap[pair] = sorted(
                fingerprints[left] & fingerprints[right]
            )
            if id_overlap[pair]:
                errors.append(f"episode_id_leakage:{pair}")
            if fingerprint_overlap[pair]:
                errors.append(f"content_fingerprint_leakage:{pair}")

    return {
        "splits_dir": splits_dir.as_posix(),
        "counts_reopened_from_hdf5": counts,
        "episode_id_overlap": id_overlap,
        "content_fingerprint_overlap": fingerprint_overlap,
        "unique_content_trajectories_global": len(
            set().union(*fingerprints.values())
        ),
        "no_leakage": not errors,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = verify(Path(args.splits_dir))
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["no_leakage"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
