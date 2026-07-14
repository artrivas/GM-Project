"""Audit raw CoinRun episodes and generate machine-readable quality evidence."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .common import scan_dataset, write_jsonl


def _horizontal_bars(
    labels: list[str], values: list[int], title: str, output: Path
) -> None:
    width = 900
    row_height = 28
    height = 70 + row_height * max(1, len(labels))
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((12, 12), title, fill="black", font=font)
    maximum = max(values, default=1) or 1
    label_width = 190
    for index, (label, value) in enumerate(zip(labels, values)):
        y = 48 + index * row_height
        draw.text((12, y + 5), label[:28], fill="black", font=font)
        bar_width = int((width - label_width - 80) * value / maximum)
        draw.rectangle((label_width, y, label_width + bar_width, y + 18), fill="#4472C4")
        draw.text((label_width + bar_width + 8, y + 5), str(value), fill="black", font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def _length_bins(lengths: list[int]) -> dict[str, int]:
    bins = {"0-99": 0, "100-199": 0, "200-299": 0, "300+": 0}
    for length in lengths:
        if length < 100:
            bins["0-99"] += 1
        elif length < 200:
            bins["100-199"] += 1
        elif length < 300:
            bins["200-299"] += 1
        else:
            bins["300+"] += 1
    return bins


def _visual_sample(
    records: list[dict[str, Any]], output: Path, sample_count: int, seed: int
) -> list[dict[str, Any]]:
    if not records or sample_count <= 0:
        return []
    rng = random.Random(seed)
    selected = rng.sample(records, min(sample_count, len(records)))
    panels_per_row = 4
    panel_size = 192
    header = 38
    row_height = panel_size + header
    montage = Image.new("RGB", (panel_size * panels_per_row, row_height * len(selected)), "white")
    draw = ImageDraw.Draw(montage)
    evidence: list[dict[str, Any]] = []
    for row, record in enumerate(selected):
        with h5py.File(record["path"], "r") as episode:
            frames = episode["observations"]
            indices = sorted(rng.sample(range(len(frames)), min(panels_per_row, len(frames))))
            for column, frame_index in enumerate(indices):
                frame = Image.fromarray(frames[frame_index]).resize(
                    (panel_size, panel_size), Image.Resampling.NEAREST
                )
                montage.paste(frame, (column * panel_size, row * row_height + header))
                draw.text(
                    (column * panel_size + 5, row * row_height + 20),
                    f"frame {frame_index}",
                    fill="black",
                )
        draw.text(
            (5, row * row_height + 3),
            f"{record['path']} | {record['terminal_event']}",
            fill="black",
        )
        evidence.append(
            {
                "path": record["path"],
                "episode_id": record["episode_id"],
                "frame_indices": indices,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    montage.save(output)
    return evidence


def build_quality_report(
    scan: dict[str, Any],
    *,
    output_dir: Path,
    sample_episodes: int,
    sample_seed: int,
) -> dict[str, Any]:
    records = scan["eligible"]
    action_counts: Counter[int] = Counter()
    lengths: list[int] = []
    by_policy: Counter[str] = Counter()
    by_seed: Counter[str] = Counter()
    by_mode: Counter[str] = Counter()
    by_event: Counter[str] = Counter()
    rare_by_policy_mode: Counter[str] = Counter()
    for record in records:
        action_counts.update(
            {int(key): value for key, value in record["action_histogram"].items()}
        )
        lengths.append(record["frames"])
        by_policy[record["source_policy"]] += 1
        by_seed[str(record["level_seed"])] += 1
        by_mode[record["distribution_mode"]] += 1
        by_event[record["terminal_event"]] += 1
        if record["terminal_event"] in {"victory", "terminal_failure"}:
            rare_by_policy_mode[
                f"{record['source_policy']}/{record['distribution_mode']}/{record['terminal_event']}"
            ] += 1

    fingerprint_groups = scan["fingerprint_groups"]
    duplicate_groups = [
        {
            "content_fingerprint": fingerprint,
            "size": len(group),
            "paths": [record["path"] for record in group],
            "terminal_event": group[0]["terminal_event"],
        }
        for fingerprint, group in sorted(fingerprint_groups.items())
        if len(group) > 1
    ]
    unique_events = Counter(
        group[0]["terminal_event"] for group in fingerprint_groups.values()
    )
    human_episodes = sum(record["source_policy"] == "human" for record in records)

    figures = output_dir / "figures"
    _horizontal_bars(
        [str(action) for action in range(15)],
        [action_counts.get(action, 0) for action in range(15)],
        "Distribución de acciones",
        figures / "action_histogram.png",
    )
    length_bins = _length_bins(lengths)
    _horizontal_bars(
        list(length_bins),
        list(length_bins.values()),
        "Longitud de episodio (transiciones)",
        figures / "episode_length_histogram.png",
    )
    _horizontal_bars(
        list(sorted(rare_by_policy_mode)),
        [rare_by_policy_mode[key] for key in sorted(rare_by_policy_mode)],
        "Eventos raros por política/modo",
        figures / "rare_events_by_policy_mode.png",
    )
    visual_evidence = _visual_sample(
        records,
        output_dir / "samples" / "random_episode_sample.png",
        sample_episodes,
        sample_seed,
    )

    frames = sum(lengths)
    unique_victories = unique_events.get("victory", 0)
    unique_failures = unique_events.get("terminal_failure", 0)
    gates = {
        "structural_integrity": len(scan["broken"]) == 0,
        "metadata_complete": all(
            record["episode_id"]
            and record["source_policy"]
            and isinstance(record["level_seed"], int)
            for record in records
        ),
        "all_15_actions_present": len(action_counts) == 15,
        "human_episode_present": human_episodes > 0,
        "at_least_3_unique_victories": unique_victories >= 3,
        "at_least_3_unique_terminal_failures": unique_failures >= 3,
    }
    gates["training_split_ready"] = all(gates.values())
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_patterns": scan["patterns"],
        "summary": {
            "scanned_episode_files": scan["scanned_paths"],
            "eligible_episodes": len(records),
            "excluded_episodes": len(scan["excluded"]),
            "broken_episodes": len(scan["broken"]),
            "frames": frames,
            "unique_content_trajectories": len(fingerprint_groups),
            "duplicate_groups": len(duplicate_groups),
            "episodes_in_duplicate_groups": sum(item["size"] for item in duplicate_groups),
            "physical_victories": by_event.get("victory", 0),
            "physical_terminal_failures": by_event.get("terminal_failure", 0),
            "unique_victories": unique_victories,
            "unique_terminal_failures": unique_failures,
            "human_episodes": human_episodes,
        },
        "action_histogram": {str(action): action_counts.get(action, 0) for action in range(15)},
        "action_coverage": len(action_counts),
        "episode_lengths": {
            "minimum": min(lengths) if lengths else 0,
            "maximum": max(lengths) if lengths else 0,
            "mean": float(np.mean(lengths)) if lengths else 0.0,
            "median": float(np.median(lengths)) if lengths else 0.0,
            "bins": length_bins,
        },
        "by_policy": dict(sorted(by_policy.items())),
        "by_seed": dict(sorted(by_seed.items())),
        "by_distribution_mode": dict(sorted(by_mode.items())),
        "by_terminal_event": dict(sorted(by_event.items())),
        "duplicate_groups": duplicate_groups,
        "excluded": scan["excluded"],
        "broken": scan["broken"],
        "visual_sample": {
            "selection_seed": sample_seed,
            "episodes": visual_evidence,
            "file": "samples/random_episode_sample.png",
        },
        "quality_gates": gates,
        "limitations": [
            "Only one level seed is present; splits cannot measure cross-seed generalization.",
            "Physical rare-event counts are inflated by exact duplicate scripted trajectories.",
            "No valid human episode is present.",
            "Death is a documented proxy for non-win terminal; death and fall are not separable.",
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", action="append", required=True, dest="patterns")
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--exclude-source-policy",
        action="append",
        default=["gui_smoke_idle"],
    )
    parser.add_argument("--sample-episodes", type=int, default=2)
    parser.add_argument("--sample-seed", type=int, default=20260714)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.out)
    scan = scan_dataset(
        args.patterns,
        excluded_policies=set(args.exclude_source_policy),
    )
    report = build_quality_report(
        scan,
        output_dir=output.parent,
        sample_episodes=args.sample_episodes,
        sample_seed=args.sample_seed,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest_records = []
    for record in scan["eligible"]:
        manifest_records.append(
            {key: value for key, value in record.items() if key != "action_histogram"}
        )
    write_jsonl(output.parent / "episodes.jsonl", manifest_records)
    write_jsonl(
        output.parent / "rejected_episodes.jsonl",
        [*scan["excluded"], *scan["broken"]],
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
