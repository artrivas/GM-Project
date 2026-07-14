"""Shared inspection primitives for Plan 2."""

from __future__ import annotations

import glob
import hashlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from src.data_collection.episode_writer import EVENT_NAMES, validate_episode


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


def normalize_glob(pattern: str) -> str:
    """Accept the escaped underscore present in the supplied phase commands."""
    return pattern.replace("\\_", "_")


def discover_paths(patterns: Iterable[str]) -> tuple[list[Path], list[dict[str, str]]]:
    paths: set[Path] = set()
    normalizations: list[dict[str, str]] = []
    for original in patterns:
        normalized = normalize_glob(original)
        normalizations.append({"original": original, "normalized": normalized})
        paths.update(Path(item) for item in glob.glob(normalized, recursive=True))
    return sorted(path.resolve() for path in paths if path.is_file()), normalizations


def _json_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def content_fingerprint(episode: h5py.File) -> str:
    digest = hashlib.sha256()
    for name in FINGERPRINT_DATASETS:
        dataset = episode[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(dataset.dtype).encode("ascii"))
        digest.update(np.asarray(dataset.shape, dtype=np.int64).tobytes())
        digest.update(dataset[...].tobytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_episode(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        validate_episode(path)
    except Exception as exc:  # quality report must preserve the exact rejection reason
        return None, [f"schema_validation: {type(exc).__name__}: {exc}"]

    with h5py.File(path, "r") as episode:
        observations = episode["observations"][...]
        next_observations = episode["next_observations"][...]
        next_valid = episode["next_observation_valid"][...]
        dones = episode["dones"][...]
        truncateds = episode["truncateds"][...]
        events = episode["events"][...]
        actions = episode["actions"][...]

        if np.any(dones[:-1]) or np.any(truncateds[:-1]):
            errors.append("terminal_or_truncated_marker_before_last_transition")
        if bool(dones[-1]) == bool(truncateds[-1]):
            errors.append("last_transition_must_be_exactly_terminal_or_truncated")
        if len(observations) > 1 and not np.array_equal(
            next_observations[:-1], observations[1:]
        ):
            errors.append("next_observation_chain_mismatch")
        if not np.array_equal(next_valid, np.logical_not(dones)):
            errors.append("next_observation_valid_mismatch")
        final_event = EVENT_NAMES.get(int(events[-1]), "unknown")
        if final_event != str(_json_value(episode.attrs["terminal_event"])):
            errors.append("terminal_event_attr_mismatch")

        attrs = {key: _json_value(episode.attrs[key]) for key in episode.attrs}
        try:
            relative_path = path.relative_to(Path.cwd()).as_posix()
        except ValueError:
            relative_path = path.as_posix()
        record = {
            "path": relative_path,
            "episode_id": str(attrs["episode_id"]),
            "level_seed": int(attrs["level_seed"]),
            "source_policy": str(attrs["source_policy"]),
            "distribution_mode": str(attrs["distribution_mode"]),
            "collection_id": str(attrs.get("collection_id", "")),
            "terminal_event": str(attrs["terminal_event"]),
            "win": int(attrs.get("win", 0)),
            "death_proxy": int(attrs.get("death", 0)),
            "truncated": int(attrs.get("truncated", 0)),
            "frames": int(len(observations)),
            "content_fingerprint": content_fingerprint(episode),
            "file_sha256": file_sha256(path),
            "human_confirmed": bool(attrs.get("human_confirmed", False)),
            "human_non_noop_actions": int(attrs.get("human_non_noop_actions", 0)),
            "action_histogram": {
                str(action): int(count)
                for action, count in sorted(Counter(map(int, actions)).items())
            },
        }
    return record, errors


def exclusion_reasons(record: dict[str, Any], excluded_policies: set[str]) -> list[str]:
    reasons: list[str] = []
    if record["source_policy"] in excluded_policies:
        reasons.append(f"excluded_source_policy:{record['source_policy']}")
    if record["collection_id"].startswith("smoke_") or Path(record["path"]).parent.name.startswith("smoke_"):
        reasons.append("non_dataset_smoke_collection")
    if record["source_policy"] == "human" and not (
        record["human_confirmed"] and record["human_non_noop_actions"] > 0
    ):
        reasons.append("invalid_or_idle_human_session")
    return reasons


def scan_dataset(
    patterns: Iterable[str],
    *,
    excluded_policies: set[str] | None = None,
) -> dict[str, Any]:
    excluded_policies = excluded_policies or {"gui_smoke_idle"}
    paths, normalizations = discover_paths(patterns)
    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    broken: list[dict[str, Any]] = []

    for path in paths:
        record, errors = inspect_episode(path)
        if errors:
            broken.append({"path": path.as_posix(), "reasons": errors})
            continue
        assert record is not None
        reasons = exclusion_reasons(record, excluded_policies)
        if reasons:
            excluded.append({"path": record["path"], "reasons": reasons, "record": record})
        else:
            eligible.append(record)

    episode_ids = Counter(record["episode_id"] for record in eligible)
    duplicate_ids = sorted(key for key, count in episode_ids.items() if count > 1)
    if duplicate_ids:
        retained: list[dict[str, Any]] = []
        for record in eligible:
            if record["episode_id"] in duplicate_ids:
                broken.append(
                    {
                        "path": record["path"],
                        "reasons": [f"duplicate_episode_id:{record['episode_id']}"],
                    }
                )
            else:
                retained.append(record)
        eligible = retained

    groups: dict[str, list[dict[str, Any]]] = {}
    for record in eligible:
        groups.setdefault(record["content_fingerprint"], []).append(record)
    for fingerprint, records in groups.items():
        for record in records:
            record["duplicate_group_size"] = len(records)
            record["duplicate_group_fingerprint"] = fingerprint

    return {
        "patterns": normalizations,
        "scanned_paths": len(paths),
        "eligible": sorted(eligible, key=lambda item: item["path"]),
        "excluded": excluded,
        "broken": broken,
        "duplicate_episode_ids": duplicate_ids,
        "fingerprint_groups": groups,
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    import json
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, path)
