"""Create deterministic, leakage-safe diagnostic splits of CoinRun episodes."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .common import scan_dataset, write_jsonl


SPLIT_NAMES = ("train", "val", "test")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def _groups(records: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["content_fingerprint"], []).append(record)
    return [sorted(group, key=lambda item: item["episode_id"]) for group in grouped.values()]


def split_records(
    records: list[dict[str, Any]],
    *,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 20260714,
) -> dict[str, list[dict[str, Any]]]:
    """Assign whole content-fingerprint groups and spread unique rare events first."""
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("ratios must contain three positive values")
    ratio_sum = sum(ratios)
    normalized = tuple(value / ratio_sum for value in ratios)
    groups = _groups(records)
    rng = random.Random(seed)

    buckets: dict[str, list[list[dict[str, Any]]]] = {
        "victory": [],
        "terminal_failure": [],
        "other": [],
    }
    for group in groups:
        events = {record["terminal_event"] for record in group}
        if len(events) != 1:
            raise ValueError("a content-fingerprint group contains conflicting terminal events")
        event = next(iter(events))
        buckets[event if event in buckets else "other"].append(group)
    for values in buckets.values():
        values.sort(key=lambda group: group[0]["content_fingerprint"])
        rng.shuffle(values)

    assigned_groups: dict[str, list[list[dict[str, Any]]]] = {
        name: [] for name in SPLIT_NAMES
    }
    frames = Counter({name: 0 for name in SPLIT_NAMES})

    def assign(group: list[dict[str, Any]], split_name: str) -> None:
        assigned_groups[split_name].append(group)
        frames[split_name] += sum(record["frames"] for record in group)

    # Each event category independently starts at train. If there are at least three
    # unique examples, every split receives one before frame balancing begins.
    for event in ("victory", "terminal_failure"):
        for index, group in enumerate(buckets[event]):
            if index < len(SPLIT_NAMES):
                assign(group, SPLIT_NAMES[index])
            else:
                buckets["other"].append(group)

    total_frames = sum(int(record["frames"]) for record in records)
    targets = {
        name: total_frames * normalized[index]
        for index, name in enumerate(SPLIT_NAMES)
    }
    remaining = buckets["other"]
    remaining.sort(
        key=lambda group: (
            -sum(record["frames"] for record in group),
            group[0]["content_fingerprint"],
        )
    )
    for group in remaining:
        # Lowest target fill ratio wins; split order resolves exact ties.
        destination = min(
            SPLIT_NAMES,
            key=lambda name: (frames[name] / targets[name], SPLIT_NAMES.index(name)),
        )
        assign(group, destination)

    result: dict[str, list[dict[str, Any]]] = {}
    for name in SPLIT_NAMES:
        result[name] = sorted(
            (record for group in assigned_groups[name] for record in group),
            key=lambda item: item["episode_id"],
        )
    return result


def validate_splits(
    splits: dict[str, list[dict[str, Any]]],
    input_records: list[dict[str, Any]],
) -> dict[str, Any]:
    episode_sets = {
        name: {record["episode_id"] for record in splits[name]}
        for name in SPLIT_NAMES
    }
    fingerprint_sets = {
        name: {record["content_fingerprint"] for record in splits[name]}
        for name in SPLIT_NAMES
    }
    episode_overlap = {
        f"{left}-{right}": sorted(episode_sets[left] & episode_sets[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    }
    fingerprint_overlap = {
        f"{left}-{right}": sorted(fingerprint_sets[left] & fingerprint_sets[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    }
    input_frames = sum(int(record["frames"]) for record in input_records)
    output_frames = sum(
        int(record["frames"])
        for name in SPLIT_NAMES
        for record in splits[name]
    )
    input_ids = {record["episode_id"] for record in input_records}
    output_ids = set().union(*episode_sets.values())
    return {
        "no_episode_id_overlap": not any(episode_overlap.values()),
        "no_content_fingerprint_overlap": not any(fingerprint_overlap.values()),
        "episode_id_overlap": episode_overlap,
        "content_fingerprint_overlap": fingerprint_overlap,
        "frame_conservation": input_frames == output_frames,
        "input_frames": input_frames,
        "output_frames": output_frames,
        "episode_conservation": input_ids == output_ids,
        "input_episodes": len(input_ids),
        "output_episodes": len(output_ids),
    }


def _split_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    physical = Counter(record["terminal_event"] for record in records)
    by_fingerprint: dict[str, dict[str, Any]] = {}
    for record in records:
        by_fingerprint.setdefault(record["content_fingerprint"], record)
    unique = Counter(record["terminal_event"] for record in by_fingerprint.values())
    return {
        "episodes": len(records),
        "frames": sum(int(record["frames"]) for record in records),
        "unique_content_trajectories": len(by_fingerprint),
        "physical_victories": physical.get("victory", 0),
        "physical_terminal_failures": physical.get("terminal_failure", 0),
        "physical_truncated": physical.get("truncated", 0),
        "unique_victories": unique.get("victory", 0),
        "unique_terminal_failures": unique.get("terminal_failure", 0),
        "unique_truncated": unique.get("truncated", 0),
        "level_seeds": sorted({int(record["level_seed"]) for record in records}),
        "source_policies": sorted({record["source_policy"] for record in records}),
    }


def build_split_summary(
    splits: dict[str, list[dict[str, Any]]],
    input_records: list[dict[str, Any]],
    *,
    strategy: str,
    seed: int,
) -> dict[str, Any]:
    checks = validate_splits(splits, input_records)
    stats = {name: _split_stats(splits[name]) for name in SPLIT_NAMES}
    rare_event_gate = {
        name: (
            stats[name]["unique_victories"] >= 1
            and stats[name]["unique_terminal_failures"] >= 1
        )
        for name in SPLIT_NAMES
    }
    human_present = any(
        record["source_policy"] == "human" for record in input_records
    )
    violations: list[str] = []
    if not checks["no_episode_id_overlap"]:
        violations.append("episode_id_leakage")
    if not checks["no_content_fingerprint_overlap"]:
        violations.append("content_duplicate_leakage")
    if not checks["frame_conservation"] or not checks["episode_conservation"]:
        violations.append("input_output_conservation_failure")
    for name, passed in rare_event_gate.items():
        if not passed:
            violations.append(f"{name}_missing_unique_victory_or_terminal_failure")
    if not human_present:
        violations.append("upstream_plan1_has_no_valid_human_episode")
    accepted = not violations
    return {
        "schema_version": 1,
        "strategy_requested": strategy,
        "strategy_effective": "by_episode_within_seed_grouped_by_content_fingerprint",
        "assignment_seed": seed,
        "accepted_for_training": accepted,
        "status": "GO" if accepted else "NO-GO_DIAGNOSTIC_SPLIT_ONLY",
        "checks": checks,
        "rare_event_gate": rare_event_gate,
        "valid_human_episode_present": human_present,
        "splits": stats,
        "violations": violations,
        "limitations": [
            "All eligible episodes use one level seed, so this is an episode split within one seed.",
            "Exact duplicate trajectories are kept together and count once for rare-event gates.",
            "A diagnostic manifest is written even on No-Go so the failure is reproducible.",
        ],
    }


def write_splits(
    output_dir: Path,
    splits: dict[str, list[dict[str, Any]]],
    summary: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        write_jsonl(output_dir / name / "manifest.jsonl", splits[name])
    summary_path = output_dir / "summary.json"
    temporary = summary_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)
    marker = output_dir / "DO_NOT_USE.json"
    if summary["accepted_for_training"]:
        marker.unlink(missing_ok=True)
    else:
        marker.write_text(
            json.dumps(
                {
                    "accepted_for_training": False,
                    "reason": "strict quality gates failed; see summary.json",
                    "violations": summary["violations"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--path", action="append", dest="patterns")
    source.add_argument("--manifest")
    parser.add_argument(
        "--strategy",
        choices=("by_episode_within_seed", "by_episode_within_seed_grouped"),
        default="by_episode_within_seed",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--strict-rare-events", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.manifest:
        records = load_manifest(Path(args.manifest))
    else:
        scan = scan_dataset(args.patterns, excluded_policies={"gui_smoke_idle"})
        if scan["broken"]:
            raise SystemExit(
                "refusing to split broken episodes: " + json.dumps(scan["broken"], sort_keys=True)
            )
        records = scan["eligible"]
    if not records:
        raise SystemExit("no eligible episodes found")

    splits = split_records(records, seed=args.seed)
    summary = build_split_summary(
        splits,
        records,
        strategy=args.strategy,
        seed=args.seed,
    )
    write_splits(Path(args.out), splits, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.strict_rare_events and not summary["accepted_for_training"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
