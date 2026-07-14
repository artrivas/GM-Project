"""Inventory an external CoinRun NPZ delivery without importing it into data/raw."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REQUIRED_KEYS = {
    "observations",
    "actions",
    "rewards",
    "level_seed",
    "completed",
    "truncated",
    "episode_length",
    "total_reward",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_fingerprint(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(REQUIRED_KEYS):
        array = arrays[name]
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


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


def inventory(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    root = root.resolve()
    manifests: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    by_team: Counter[str] = Counter()
    by_mode: Counter[str] = Counter()
    by_outcome: Counter[str] = Counter()
    action_counts: Counter[int] = Counter()
    seeds: Counter[int] = Counter()
    content_groups: dict[str, list[str]] = defaultdict(list)
    total_frames = 0
    pixel_min = 255
    pixel_max = 0

    session_dirs = sorted(path.parent for path in root.glob("Team*/session*/session.json"))
    for session_dir in session_dirs:
        team = session_dir.parent.name
        session_file = session_dir / "session.json"
        try:
            session = json.loads(session_file.read_text(encoding="utf-8"))
        except Exception as exc:
            quarantine.append(
                {"type": "invalid_session_json", "path": session_file.as_posix(), "error": str(exc)}
            )
            continue
        mode = str(session.get("env_kwargs", {}).get("distribution_mode", "unknown"))
        declared_entries = {str(item["file"]): item for item in session.get("episodes", [])}
        actual_files = {path.name: path for path in sorted(session_dir.glob("episode_*.npz"))}
        missing_files = sorted(set(declared_entries) - set(actual_files))
        undeclared_files = sorted(set(actual_files) - set(declared_entries))
        for name in missing_files:
            quarantine.append(
                {
                    "type": "missing_declared_episode_file",
                    "session": session_dir.relative_to(root).as_posix(),
                    "file": name,
                }
            )
        for name in undeclared_files:
            quarantine.append(
                {
                    "type": "undeclared_episode_file",
                    "session": session_dir.relative_to(root).as_posix(),
                    "file": name,
                }
            )

        valid_in_session = 0
        frames_in_session = 0
        wins_in_session = 0
        for name, path in actual_files.items():
            external_id = f"{team.lower()}__{session_dir.name}__{path.stem}"
            errors: list[str] = []
            try:
                with np.load(path, allow_pickle=False) as archive:
                    if set(archive.files) != REQUIRED_KEYS:
                        errors.append(
                            f"keys_mismatch: expected={sorted(REQUIRED_KEYS)} actual={sorted(archive.files)}"
                        )
                    if errors:
                        raise ValueError(errors[0])
                    arrays = {key: archive[key] for key in archive.files}
            except Exception as exc:
                quarantine.append(
                    {
                        "type": "invalid_npz",
                        "path": path.relative_to(root).as_posix(),
                        "external_episode_id": external_id,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue

            observations = arrays["observations"]
            actions = arrays["actions"]
            rewards = arrays["rewards"]
            length = int(arrays["episode_length"])
            seed = int(arrays["level_seed"])
            completed = bool(arrays["completed"])
            truncated = bool(arrays["truncated"])
            total_reward = float(arrays["total_reward"])
            if observations.dtype != np.uint8 or observations.shape != (length, 64, 64, 3):
                errors.append(f"invalid_observations:{observations.dtype}:{observations.shape}")
            if actions.shape != (length,) or rewards.shape != (length,):
                errors.append(
                    f"unaligned_transitions:observations={len(observations)} actions={actions.shape} rewards={rewards.shape}"
                )
            if length <= 0:
                errors.append("empty_episode")
            if actions.size and (int(actions.min()) < 0 or int(actions.max()) >= 15):
                errors.append("action_out_of_range")
            if not np.isclose(float(rewards.sum()), total_reward, atol=1e-5):
                errors.append(f"reward_sum_mismatch:{float(rewards.sum())}:{total_reward}")
            declared = declared_entries.get(name)
            if declared:
                comparisons = {
                    "length": length,
                    "level_seed": seed,
                    "completed": completed,
                    "truncated": truncated,
                }
                for key, measured in comparisons.items():
                    if declared.get(key) != measured:
                        errors.append(f"session_metadata_mismatch:{key}:{declared.get(key)}:{measured}")
                if not np.isclose(float(declared.get("total_reward", 0.0)), total_reward, atol=1e-5):
                    errors.append(
                        f"session_metadata_mismatch:total_reward:{declared.get('total_reward')}:{total_reward}"
                    )
            if errors:
                quarantine.append(
                    {
                        "type": "episode_validation_failure",
                        "path": path.relative_to(root).as_posix(),
                        "external_episode_id": external_id,
                        "errors": errors,
                    }
                )
                continue

            fingerprint = _content_fingerprint(arrays)
            content_groups[fingerprint].append(external_id)
            relative_path = path.relative_to(root).as_posix()
            outcome = "victory" if completed else "truncated" if truncated else "non_win_terminal"
            record = {
                "external_episode_id": external_id,
                "path": relative_path,
                "team": team,
                "session_directory": session_dir.name,
                "session_id": str(session.get("session_id", session_dir.name)),
                "distribution_mode": mode,
                "level_seed": seed,
                "frames": length,
                "total_reward": total_reward,
                "completed": completed,
                "truncated": truncated,
                "outcome": outcome,
                "source_policy": "unknown_external",
                "file_sha256": _file_sha256(path),
                "content_fingerprint": fingerprint,
                "ingestion_status": "isolated_needs_metadata_review",
            }
            manifests.append(record)
            valid_in_session += 1
            frames_in_session += length
            wins_in_session += int(completed)
            total_frames += length
            by_team[team] += 1
            by_mode[mode] += 1
            by_outcome[outcome] += 1
            seeds[seed] += 1
            action_counts.update(map(int, actions))
            pixel_min = min(pixel_min, int(observations.min()))
            pixel_max = max(pixel_max, int(observations.max()))

        declared_summary = session.get("summary", {})
        session_warnings: list[str] = []
        if int(declared_summary.get("episodes", len(declared_entries))) != len(actual_files):
            session_warnings.append(
                f"summary_episode_count_mismatch:{declared_summary.get('episodes')}:{len(actual_files)}"
            )
        sessions.append(
            {
                "team": team,
                "session": session_dir.name,
                "distribution_mode": mode,
                "declared_episodes": len(declared_entries),
                "physical_episode_files": len(actual_files),
                "valid_episodes": valid_in_session,
                "frames": frames_in_session,
                "victories": wins_in_session,
                "missing_declared_files": missing_files,
                "undeclared_files": undeclared_files,
                "warnings": session_warnings,
            }
        )

    duplicate_groups = [
        {"content_fingerprint": key, "size": len(values), "episode_ids": sorted(values)}
        for key, values in content_groups.items()
        if len(values) > 1
    ]
    inventory_report = {
        "schema_version": 1,
        "dataset_role": "isolated_external_delivery_not_active_training_data",
        "root": root.as_posix(),
        "summary": {
            "teams": len({item["team"] for item in sessions}),
            "sessions": len(sessions),
            "physical_npz_files": len(list(root.glob("Team*/session*/episode_*.npz"))),
            "valid_episodes": len(manifests),
            "quarantine_records": len(quarantine),
            "frames": total_frames,
            "unique_level_seeds": len(seeds),
            "duplicate_content_groups": len(duplicate_groups),
            "pixel_min": pixel_min if manifests else None,
            "pixel_max": pixel_max if manifests else None,
        },
        "by_team": dict(sorted(by_team.items())),
        "by_distribution_mode": dict(sorted(by_mode.items())),
        "by_outcome": dict(sorted(by_outcome.items())),
        "action_histogram": {str(key): action_counts.get(key, 0) for key in range(15)},
        "sessions": sessions,
        "duplicate_groups": duplicate_groups,
        "limitations": [
            "source_policy is absent; records are labeled unknown_external, not human.",
            "initial_state, next_observations, dones and per-transition episode metadata are absent.",
            "num_levels=0 and many procedural seeds do not match the current fixed-seed scope.",
            "Non-completed episodes are called non_win_terminal; death versus fall is not inferable.",
            "This inventory does not authorize inclusion in data/raw or regeneration of active splits.",
        ],
    }
    return inventory_report, sorted(manifests, key=lambda item: item["external_episode_id"]), quarantine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, manifest, quarantine = inventory(Path(args.root))
    output = Path(args.out)
    _write_json(output / "inventory.json", report)
    _write_jsonl(output / "manifest.jsonl", manifest)
    _write_jsonl(output / "quarantine.jsonl", quarantine)
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
