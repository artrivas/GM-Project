"""Independent HDF5 recount for the Plan 2 dataset audit.

This module intentionally does not import the Plan 2 audit or split implementation.
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
from typing import Any

import h5py


def _normalise_pattern(pattern: str) -> str:
    return pattern.replace("\\_", "_")


def _is_eligible(path: Path, episode: h5py.File) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    policy = str(episode.attrs.get("source_policy", ""))
    if path.parent.name.startswith("smoke_"):
        reasons.append("smoke_directory")
    if policy == "gui_smoke_idle":
        reasons.append("gui_smoke_idle_policy")
    if policy == "human" and not (
        bool(episode.attrs.get("human_confirmed", False))
        and int(episode.attrs.get("human_non_noop_actions", 0)) > 0
    ):
        reasons.append("invalid_human_session")
    return not reasons, reasons


def recount(pattern: str, report_path: Path | None = None) -> dict[str, Any]:
    normalised = _normalise_pattern(pattern)
    paths = sorted(Path(item) for item in glob.glob(normalised))
    raw = Counter()
    eligible = Counter()
    seed_counts: Counter[int] = Counter()
    exclusions: list[dict[str, Any]] = []
    inconsistencies: list[dict[str, Any]] = []
    truncated_without_done: list[str] = []
    truncated_with_explicit_marker = 0

    for path in paths:
        with h5py.File(path, "r") as episode:
            frames = int(episode["observations"].shape[0])
            win = int(episode.attrs.get("win", 0))
            death = int(episode.attrs.get("death", 0))
            raw.update(episodes=1, frames=frames, victories=win, deaths=death)
            accepted, reasons = _is_eligible(path, episode)
            if not accepted:
                exclusions.append(
                    {"path": path.as_posix(), "frames": frames, "reasons": reasons}
                )
                continue

            episode_id = str(episode.attrs["episode_id"])
            truncated_attr = int(episode.attrs.get("truncated", 0))
            final_event = int(episode["events"][-1])
            done_last = bool(episode["dones"][-1])
            truncated_last = bool(episode["truncateds"][-1])
            expected_flags = (final_event == 1, final_event == 2, final_event == 3)
            actual_flags = (bool(win), bool(death), bool(truncated_attr))
            if expected_flags != actual_flags:
                inconsistencies.append(
                    {
                        "episode_id": episode_id,
                        "event_code": final_event,
                        "attribute_flags": actual_flags,
                    }
                )
            if truncated_attr and not done_last:
                truncated_without_done.append(episode_id)
                if truncated_last and final_event == 3:
                    truncated_with_explicit_marker += 1

            eligible.update(
                episodes=1,
                frames=frames,
                victories=win,
                deaths=death,
                truncated=truncated_attr,
            )
            seed_counts[int(episode.attrs["level_seed"])] += 1

    result: dict[str, Any] = {
        "input_pattern": {"original": pattern, "normalised": normalised},
        "raw_all_files": dict(raw),
        "eligible": dict(eligible),
        "excluded": exclusions,
        "seed_episode_counts": {str(key): value for key, value in sorted(seed_counts.items())},
        "terminal_metadata_inconsistencies": inconsistencies,
        "truncation_check": {
            "truncated_without_done": len(truncated_without_done),
            "episode_ids": truncated_without_done,
            "also_marked_truncated_true_and_event_3": truncated_with_explicit_marker,
        },
    }

    if report_path is not None and report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        claimed = {
            "episodes": int(report["summary"]["eligible_episodes"]),
            "frames": int(report["summary"]["frames"]),
            "victories": int(report["summary"]["physical_victories"]),
            "deaths": int(report["summary"]["physical_terminal_failures"]),
            "seed_episode_counts": {
                str(key): int(value) for key, value in report["by_seed"].items()
            },
        }
        measured = {
            "episodes": eligible["episodes"],
            "frames": eligible["frames"],
            "victories": eligible["victories"],
            "deaths": eligible["deaths"],
            "seed_episode_counts": result["seed_episode_counts"],
        }
        discrepancies = {
            key: {"claimed": claimed[key], "measured": measured[key]}
            for key in claimed
            if claimed[key] != measured[key]
        }
        result["report_comparison"] = {
            "report": report_path.as_posix(),
            "claimed": claimed,
            "measured": measured,
            "matches": not discrepancies,
            "discrepancies": discrepancies,
        }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument(
        "--report",
        default="experiments/EXP-002-dataset-audit/quality_report.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = recount(args.path, Path(args.report))
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("report_comparison", {}).get("discrepancies"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
