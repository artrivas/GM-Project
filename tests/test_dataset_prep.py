from __future__ import annotations

import json
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data_collection.episode_writer import EpisodeWriter
from src.data_prep.common import inspect_episode, normalize_glob, scan_dataset
from src.data_prep.split_dataset import (
    SPLIT_NAMES,
    build_split_summary,
    split_records,
    validate_splits,
)


def _record(
    episode_id: str,
    fingerprint: str,
    event: str = "truncated",
    frames: int = 10,
    policy: str = "random",
) -> dict:
    return {
        "path": f"data/raw/test/{episode_id}.h5",
        "episode_id": episode_id,
        "level_seed": 0,
        "source_policy": policy,
        "distribution_mode": "easy",
        "terminal_event": event,
        "frames": frames,
        "content_fingerprint": fingerprint,
    }


def _episode(path: Path, episode_id: str, *, offset: int = 0) -> Path:
    metadata = {
        "episode_id": episode_id,
        "level_seed": 0,
        "source_policy": "random",
        "distribution_mode": "easy",
        "procgen_version": "0.10.7",
        "collector_git_commit": "test",
        "collection_id": "pytest",
    }
    frames = [
        np.full((64, 64, 3), offset + index, dtype=np.uint8)
        for index in range(3)
    ]
    writer = EpisodeWriter(path, metadata, initial_state=b"state")
    writer.append(
        observation=frames[0],
        action=1,
        reward=0.0,
        done=False,
        truncated=False,
        event="continue",
        next_observation=frames[1],
        timestamp_ns=1,
    )
    writer.append(
        observation=frames[1],
        action=7,
        reward=1.0,
        done=True,
        truncated=False,
        event="victory",
        next_observation=frames[2],
        timestamp_ns=2,
    )
    return writer.finalize()


def test_escaped_underscore_in_supplied_command_is_normalized():
    assert normalize_glob(r"data/raw/*/episode\_*.h5") == "data/raw/*/episode_*.h5"


def test_content_duplicates_receive_the_same_fingerprint(tmp_path: Path):
    first, errors_first = inspect_episode(_episode(tmp_path / "a.h5", "a"))
    second, errors_second = inspect_episode(_episode(tmp_path / "b.h5", "b"))
    assert errors_first == errors_second == []
    assert first is not None and second is not None
    assert first["file_sha256"] != second["file_sha256"]
    assert first["content_fingerprint"] == second["content_fingerprint"]


def test_broken_frame_sequence_is_rejected(tmp_path: Path):
    path = _episode(tmp_path / "broken.h5", "broken")
    with h5py.File(path, "r+") as episode:
        episode["frame_indices"][1] = 9
    record, errors = inspect_episode(path)
    assert record is None
    assert "frame indices are not contiguous" in errors[0]


def test_scan_excludes_smoke_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    path = _episode(tmp_path / "data/raw/smoke_random/episode_000.h5", "smoke")
    assert path.exists()
    scan = scan_dataset(["data/raw/*/episode_*.h5"])
    assert scan["eligible"] == []
    assert scan["excluded"][0]["reasons"] == ["non_dataset_smoke_collection"]


def test_no_episode_or_duplicate_content_leakage_and_frame_conservation():
    records = [
        _record("duplicate-a", "same", frames=4),
        _record("duplicate-b", "same", frames=4),
        *[_record(f"other-{index}", f"fp-{index}", frames=10 + index) for index in range(8)],
    ]
    splits = split_records(records, seed=42)
    checks = validate_splits(splits, records)
    assert checks["no_episode_id_overlap"]
    assert checks["no_content_fingerprint_overlap"]
    assert checks["frame_conservation"]
    assert checks["episode_conservation"]


def test_split_is_deterministic_for_a_seed():
    records = [_record(str(index), f"fp-{index}") for index in range(12)]
    first = split_records(records, seed=123)
    second = split_records(list(reversed(records)), seed=123)
    first_ids = {name: [item["episode_id"] for item in first[name]] for name in SPLIT_NAMES}
    second_ids = {name: [item["episode_id"] for item in second[name]] for name in SPLIT_NAMES}
    assert first_ids == second_ids


def test_unique_rare_events_are_distributed_when_enough_exist():
    records = []
    for index in range(3):
        records.append(_record(f"win-{index}", f"win-fp-{index}", "victory"))
        records.append(_record(f"fail-{index}", f"fail-fp-{index}", "terminal_failure"))
    records.append(_record("human", "human-fp", policy="human"))
    splits = split_records(records, seed=7)
    summary = build_split_summary(
        splits, records, strategy="by_episode_within_seed", seed=7
    )
    assert all(summary["rare_event_gate"].values())
    assert summary["accepted_for_training"]


def test_rare_event_gate_fails_for_one_unique_event_of_each_kind():
    records = [
        _record("win-a", "win", "victory"),
        _record("win-b", "win", "victory"),
        _record("fail-a", "fail", "terminal_failure"),
        _record("fail-b", "fail", "terminal_failure"),
        _record("human", "human", policy="human"),
        *[_record(f"other-{index}", f"other-{index}") for index in range(6)],
    ]
    splits = split_records(records, seed=7)
    summary = build_split_summary(
        splits, records, strategy="by_episode_within_seed", seed=7
    )
    assert summary["checks"]["no_content_fingerprint_overlap"]
    assert not summary["accepted_for_training"]
    assert not all(summary["rare_event_gate"].values())


def test_real_split_artifacts_are_conservative_when_present():
    summary_path = Path("data/splits/summary.json")
    if not summary_path.exists():
        pytest.skip("real Plan 2 split has not been generated")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["checks"]["no_episode_id_overlap"]
    assert summary["checks"]["no_content_fingerprint_overlap"]
    assert summary["checks"]["frame_conservation"]
    assert summary["checks"]["episode_conservation"]
    for name in SPLIT_NAMES:
        assert (Path("data/splits") / name / "manifest.jsonl").is_file()


def test_no_leakage():
    """Independent audit: reopen every manifest target and compare IDs/content."""
    split_ids: dict[str, set[str]] = {}
    split_fingerprints: dict[str, set[str]] = {}
    datasets = (
        "observations",
        "next_observations",
        "next_observation_valid",
        "actions",
        "rewards",
        "dones",
        "truncateds",
        "events",
    )
    for split in ("train", "val", "test"):
        manifest = Path("data/splits") / split / "manifest.jsonl"
        assert manifest.is_file(), f"missing split manifest: {manifest}"
        ids: list[str] = []
        fingerprints: list[str] = []
        for line in manifest.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            with h5py.File(item["path"], "r") as episode:
                episode_id = str(episode.attrs["episode_id"])
                assert item["episode_id"] == episode_id
                ids.append(episode_id)
                digest = hashlib.sha256()
                for name in datasets:
                    array = episode[name][...]
                    digest.update(name.encode("utf-8"))
                    digest.update(str(array.dtype).encode("ascii"))
                    digest.update(str(array.shape).encode("ascii"))
                    digest.update(array.tobytes())
                fingerprints.append(digest.hexdigest())
        assert len(ids) == len(set(ids)), f"duplicate episode ID inside {split}"
        split_ids[split] = set(ids)
        split_fingerprints[split] = set(fingerprints)

    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not split_ids[left] & split_ids[right]
        assert not split_fingerprints[left] & split_fingerprints[right]
