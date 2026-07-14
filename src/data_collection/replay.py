"""Validate an HDF5 episode offline and optionally against Procgen state replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .collector import advance_collection_step, create_coinrun_env
from .episode_writer import load_episode, validate_episode


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_with_procgen(path: str | Path) -> dict[str, int | str]:
    episode = load_episode(path)
    arrays = episode["arrays"]
    attrs = episode["attrs"]
    env = create_coinrun_env(
        level_seed=int(attrs["level_seed"]),
        distribution_mode=str(attrs["distribution_mode"]),
    )
    compared_frames = 0
    try:
        env.callmethod("set_state", [arrays["initial_state"].tobytes()])
        for index, action in enumerate(arrays["actions"]):
            _, current_observation, _ = env.observe()
            np.testing.assert_array_equal(
                current_observation["rgb"][0], arrays["observations"][index]
            )
            result = advance_collection_step(
                env, int(action), purpose="replay_verification"
            )
            np.testing.assert_allclose(
                result.reward,
                float(arrays["rewards"][index]),
                rtol=0.0,
                atol=1e-6,
            )
            if result.done != bool(arrays["dones"][index]):
                raise AssertionError(f"done mismatch at transition {index}")
            if bool(arrays["next_observation_valid"][index]):
                np.testing.assert_array_equal(
                    result.next_observation,
                    arrays["next_observations"][index],
                )
                compared_frames += 1
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return {
        "episode": str(path),
        "transitions_verified": len(arrays["actions"]),
        "next_frames_verified": compared_frames,
        "simulator_match": 1,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument("--verify-simulator", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = validate_episode(args.episode)
    summary["sha256"] = file_sha256(args.episode)
    print("offline=" + json.dumps(summary, sort_keys=True))
    if args.verify_simulator:
        print(
            "simulator="
            + json.dumps(verify_with_procgen(args.episode), sort_keys=True)
        )


if __name__ == "__main__":
    main()
