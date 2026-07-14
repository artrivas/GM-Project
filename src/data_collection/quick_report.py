"""Quantitative coverage report for raw per-episode HDF5 files."""

from __future__ import annotations

import argparse
import glob
import json
from collections import Counter

import h5py

from .episode_writer import validate_episode


def quick_report(path_glob: str) -> dict[str, object]:
    paths = sorted(glob.glob(path_glob))
    counters: Counter[str] = Counter()
    by_policy: Counter[str] = Counter()
    by_mode: Counter[str] = Counter()
    metadata_complete = 0
    for path in paths:
        validate_episode(path)
        with h5py.File(path, "r") as episode:
            counters["episodes"] += 1
            counters["frames"] += int(episode["observations"].shape[0])
            counters["deaths"] += int(episode.attrs.get("death", 0))
            counters["wins"] += int(episode.attrs.get("win", 0))
            counters["truncated"] += int(episode.attrs.get("truncated", 0))
            if "level_seed" in episode.attrs and "source_policy" in episode.attrs:
                metadata_complete += 1
            by_policy[str(episode.attrs["source_policy"])] += 1
            by_mode[str(episode.attrs["distribution_mode"])] += 1
    report: dict[str, object] = dict(counters)
    report.setdefault("episodes", 0)
    report.setdefault("frames", 0)
    report.setdefault("deaths", 0)
    report.setdefault("wins", 0)
    report.setdefault("truncated", 0)
    report["metadata_complete"] = metadata_complete
    report["metadata_percent"] = (
        100.0 * metadata_complete / int(report["episodes"])
        if report["episodes"]
        else 0.0
    )
    report["by_policy"] = dict(sorted(by_policy.items()))
    report["by_mode"] = dict(sorted(by_mode.items()))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, dest="path_glob")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = quick_report(args.path_glob)
    print(
        "episodios={episodes} frames={frames} muertes_proxy={deaths} "
        "victorias={wins} truncados={truncated} metadata={metadata_percent:.1f}%".format(
            **report
        )
    )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
