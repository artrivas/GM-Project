"""Promote the indexed external CoinRun delivery into leakage-safe active manifests."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _parse_team_list(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def _smoke_sample(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    remaining = sorted(records, key=lambda item: item["episode_id"])
    selected: list[dict[str, Any]] = []
    team_counts: Counter[str] = Counter()
    seen_modes: set[str] = set()
    seen_outcomes: set[str] = set()
    seen_sessions: set[str] = set()
    while remaining and len(selected) < count:
        def score(record: dict[str, Any]) -> tuple[int, int, int, int, int, str]:
            return (
                -team_counts[record["team"]],
                int(record["distribution_mode"] not in seen_modes),
                int(record["outcome"] not in seen_outcomes),
                int(record["group_id"] not in seen_sessions),
                -int(record["frames"]),
                record["episode_id"],
            )

        chosen = max(remaining, key=score)
        remaining.remove(chosen)
        selected.append(chosen)
        team_counts[chosen["team"]] += 1
        seen_modes.add(chosen["distribution_mode"])
        seen_outcomes.add(chosen["outcome"])
        seen_sessions.add(chosen["group_id"])
    return sorted(selected, key=lambda item: item["episode_id"])


def _stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(records),
        "frames": sum(int(record["frames"]) for record in records),
        "teams": sorted({record["team"] for record in records}),
        "sessions": len({(record["team"], record["session_directory"]) for record in records}),
        "unique_seeds": len({int(record["level_seed"]) for record in records}),
        "by_mode": dict(sorted(Counter(record["distribution_mode"] for record in records).items())),
        "by_outcome": dict(sorted(Counter(record["outcome"] for record in records).items())),
    }


def promote(
    source_manifest: Path,
    original_root: Path,
    output: Path,
    team_assignment: dict[str, set[str]],
    smoke_episodes: int,
) -> dict[str, Any]:
    team_sets = list(team_assignment.values())
    if any(team_sets[i] & team_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("team assignments overlap")
    indexed = _read_jsonl(source_manifest)
    indexed_teams = {record["team"] for record in indexed}
    assigned_teams = set().union(*team_sets)
    if indexed_teams != assigned_teams:
        raise ValueError(
            f"team assignment must cover the delivery exactly: indexed={indexed_teams} assigned={assigned_teams}"
        )

    splits: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLITS}
    for record in indexed:
        split = next(name for name, teams in team_assignment.items() if record["team"] in teams)
        path = original_root / record["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        promoted = {
            "episode_id": record["external_episode_id"],
            "path": path.as_posix(),
            "storage_format": "npz_external_v1",
            "frames": int(record["frames"]),
            "level_seed": int(record["level_seed"]),
            "source_policy": "unknown_external",
            "distribution_mode": record["distribution_mode"],
            "outcome": record["outcome"],
            "completed": bool(record["completed"]),
            "truncated": bool(record["truncated"]),
            "team": record["team"],
            "session_directory": record["session_directory"],
            "group_id": f"{record['team']}::{record['session_directory']}",
            "content_fingerprint": record["content_fingerprint"],
            "file_sha256": record["file_sha256"],
            "duplicate_group_size": 1,
        }
        splits[split].append(promoted)

    for split in SPLITS:
        records = sorted(splits[split], key=lambda item: item["episode_id"])
        _write_jsonl(output / "splits" / split / "manifest.jsonl", records)
        _write_jsonl(
            output / "smoke" / split / "manifest.jsonl",
            _smoke_sample(records, smoke_episodes),
        )

    team_overlap = {
        f"{left}-{right}": sorted(team_assignment[left] & team_assignment[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    seed_sets = {
        split: {record["level_seed"] for record in records}
        for split, records in splits.items()
    }
    seed_overlap = {
        f"{left}-{right}": sorted(seed_sets[left] & seed_sets[right])
        for index, left in enumerate(SPLITS)
        for right in SPLITS[index + 1 :]
    }
    summary = {
        "schema_version": 1,
        "dataset_name": "coinrun_teams_v1",
        "active_for": ["autoencoder", "visual_reconstruction"],
        "not_yet_active_for": ["rssm", "action_conditioned_dynamics"],
        "split_strategy": "hold_out_entire_teams",
        "splits": {split: _stats(records) for split, records in splits.items()},
        "smoke": {
            split: _stats(_read_jsonl(output / "smoke" / split / "manifest.jsonl"))
            for split in SPLITS
        },
        "checks": {
            "team_overlap": team_overlap,
            "seed_overlap": seed_overlap,
            "no_team_leakage": not any(team_overlap.values()),
            "no_seed_leakage": not any(seed_overlap.values()),
            "episode_conservation": sum(len(records) for records in splits.values()) == len(indexed),
            "frame_conservation": sum(
                int(record["frames"]) for records in splits.values() for record in records
            )
            == sum(int(record["frames"]) for record in indexed),
        },
        "limitations": [
            "Team identifiers are treated as provenance groups, not automatically as human labels.",
            "Action/frame temporal alignment must be audited before RSSM training.",
            "Smoke manifests are diagnostic samples and must never replace full evaluation manifests.",
        ],
    }
    _write_json(output / "summary.json", summary)
    _write_json(
        output / "ACTIVE_DATASET.json",
        {
            "dataset": "coinrun_teams_v1",
            "active_for_autoencoder": True,
            "active_splits": "splits/",
            "smoke_splits": "smoke/",
            "legacy_splits_untouched": "data/splits/",
            "dynamics_ready": False,
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--original-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-teams", default="Team1,Team2,Team3,Team4")
    parser.add_argument("--val-teams", default="Team5")
    parser.add_argument("--test-teams", default="Team6")
    parser.add_argument("--smoke-episodes", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assignment = {
        "train": _parse_team_list(args.train_teams),
        "val": _parse_team_list(args.val_teams),
        "test": _parse_team_list(args.test_teams),
    }
    summary = promote(
        Path(args.source_manifest),
        Path(args.original_root),
        Path(args.out),
        assignment,
        args.smoke_episodes,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
