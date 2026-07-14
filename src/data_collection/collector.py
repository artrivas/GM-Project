"""Core collection loop; the only project module allowed to advance Procgen."""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from procgen import ProcgenGym3Env

from .episode_writer import EpisodeWriter, validate_episode
from .policies import Policy


SIMULATOR_TRANSITION_ALLOWED = "data_collection_or_replay_verification_only"
ALLOWED_TRANSITION_PURPOSES = {"data_collection", "smoke_test", "replay_verification"}


@dataclass(frozen=True)
class StepResult:
    reward: float
    next_observation: np.ndarray
    done: bool
    info: dict[str, Any]


def current_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def create_coinrun_env(
    *,
    level_seed: int,
    distribution_mode: str,
    render_mode: str | None = None,
) -> ProcgenGym3Env:
    kwargs: dict[str, Any] = {
        "num": 1,
        "env_name": "coinrun",
        "start_level": int(level_seed),
        "num_levels": 1,
        "distribution_mode": distribution_mode,
        "use_sequential_levels": False,
        "num_threads": 0,
    }
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return ProcgenGym3Env(**kwargs)


def advance_collection_step(
    env: ProcgenGym3Env,
    action: int,
    *,
    purpose: str,
) -> StepResult:
    """Advance Procgen exactly once under an explicit, audited purpose marker."""
    if purpose not in ALLOWED_TRANSITION_PURPOSES:
        raise PermissionError(
            f"Procgen transition forbidden for purpose={purpose!r}; "
            f"marker={SIMULATOR_TRANSITION_ALLOWED}"
        )
    if not 0 <= int(action) < 15:
        raise ValueError(f"invalid action: {action}")
    env.act(np.asarray([action], dtype=np.int32))
    reward, observation, first = env.observe()
    return StepResult(
        reward=float(reward[0]),
        next_observation=observation["rgb"][0].copy(),
        done=bool(first[0]),
        info=dict(env.get_info()[0]),
    )


def classify_event(*, done: bool, truncated: bool, info: dict[str, Any]) -> str:
    if truncated:
        return "truncated"
    if not done:
        return "continue"
    if bool(info.get("prev_level_complete", False)):
        return "victory"
    return "terminal_failure"


def collect_episodes(
    *,
    policy: Policy,
    level_seed: int,
    episodes: int,
    output_dir: str | Path,
    distribution_mode: str,
    max_steps: int,
    collection_id: str,
) -> list[dict[str, Any]]:
    if episodes <= 0 or max_steps <= 0:
        raise ValueError("episodes and max_steps must be positive")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for episode_index in range(episodes):
        env = create_coinrun_env(
            level_seed=level_seed,
            distribution_mode=distribution_mode,
        )
        writer: EpisodeWriter | None = None
        try:
            _, observation_batch, first = env.observe()
            if not bool(first[0]):
                raise RuntimeError("new Procgen environment did not start at an episode boundary")
            observation = observation_batch["rgb"][0].copy()
            initial_state = env.callmethod("get_state")[0]
            episode_id = f"{collection_id}-{policy.name}-s{level_seed}-{episode_index:03d}"
            path = output_dir / f"episode_{episode_index:03d}.h5"
            writer = EpisodeWriter(
                path,
                metadata={
                    "episode_id": episode_id,
                    "level_seed": int(level_seed),
                    "source_policy": policy.name,
                    "distribution_mode": distribution_mode,
                    "procgen_version": "0.10.7",
                    "collector_git_commit": current_git_commit(),
                    "collection_id": collection_id,
                    "policy_rng_seed": getattr(policy, "rng_seed", -1),
                    "sticky_probability": getattr(policy, "repeat_probability", -1.0),
                    "action_count": 15,
                    "observation_shape": "64,64,3",
                },
                initial_state=initial_state,
            )
            policy.reset()

            for step_index in range(max_steps):
                action = policy.act(observation)
                result = advance_collection_step(
                    env, action, purpose="data_collection"
                )
                truncated = step_index + 1 == max_steps and not result.done
                event = classify_event(
                    done=result.done,
                    truncated=truncated,
                    info=result.info,
                )
                writer.append(
                    observation=observation,
                    action=action,
                    reward=result.reward,
                    done=result.done,
                    truncated=truncated,
                    event=event,
                    next_observation=result.next_observation,
                    timestamp_ns=time.time_ns(),
                )
                observation = result.next_observation
                if result.done or truncated:
                    break

            final_path = writer.finalize()
            writer = None
            results.append(validate_episode(final_path))
        except Exception:
            if writer is not None:
                writer.abort()
            raise
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
    return results
