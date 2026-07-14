"""Command-line entry point for automatic CoinRun collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .collector import collect_episodes
from .policies import make_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--policy", required=True, choices=["random", "sticky", "scripted", "trained"]
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--distribution-mode", choices=["hard", "easy"])
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--rng-seed", type=int)
    parser.add_argument("--sticky-probability", type=float)
    parser.add_argument("--collection-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    distribution_mode = (
        args.distribution_mode or config["environment"]["distribution_mode"]
    )
    max_steps = args.max_steps or config["pilot"]["max_steps_per_episode"]
    rng_seed = args.rng_seed
    if rng_seed is None:
        rng_seed = int(config["pilot"]["policy_rng_seed"]) + int(args.seed)
    sticky_probability = args.sticky_probability
    if sticky_probability is None:
        sticky_probability = config["policies"]["sticky_default_for_pilot"]
    output_dir = Path(args.out)
    collection_id = args.collection_id or output_dir.name

    policy = make_policy(
        args.policy,
        rng_seed=rng_seed,
        action_count=int(config["environment"]["action_count"]),
        sticky_probability=float(sticky_probability),
    )
    results = collect_episodes(
        policy=policy,
        level_seed=args.seed,
        episodes=args.episodes,
        output_dir=output_dir,
        distribution_mode=distribution_mode,
        max_steps=max_steps,
        collection_id=collection_id,
    )
    for result in results:
        print(json.dumps(result, sort_keys=True))
    aggregate = {
        "episodes": len(results),
        "transitions": sum(item["transitions"] for item in results),
        "wins": sum(item["win"] for item in results),
        "deaths": sum(item["death"] for item in results),
        "truncated": sum(item["truncated"] for item in results),
        "policy": args.policy,
        "level_seed": args.seed,
        "distribution_mode": distribution_mode,
    }
    print("aggregate=" + json.dumps(aggregate, sort_keys=True))


if __name__ == "__main__":
    main()
