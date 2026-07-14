"""Human keyboard capture reusing Procgen/Gym3's official interactive loop."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import yaml
from gym3 import Interactive
from procgen.interactive import ProcgenInteractive

from .collector import (
    advance_collection_step,
    classify_event,
    create_coinrun_env,
    current_git_commit,
)
from .episode_writer import EpisodeWriter


class KeyboardCapture(ProcgenInteractive):
    """Keep official rendering/key mapping while recording every transition."""

    def __init__(
        self,
        *,
        output_dir: Path,
        level_seed: int,
        distribution_mode: str,
        session_id: str,
        max_minutes: float,
        tps: int,
    ) -> None:
        self._capture_env = create_coinrun_env(
            level_seed=level_seed,
            distribution_mode=distribution_mode,
        )
        super().__init__(
            self._capture_env,
            ob_key="rgb",
            width=768,
            height=768,
            tps=tps,
        )
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._level_seed = level_seed
        self._distribution_mode = distribution_mode
        self._session_id = session_id
        self._max_seconds = max_minutes * 60.0
        self._started_at = time.monotonic()
        self._writer: EpisodeWriter | None = None
        self._episode_file_index = self._next_file_index()
        self._non_noop_actions = 0
        self.completed_paths: list[Path] = []

    def _next_file_index(self) -> int:
        existing = []
        for path in self._output_dir.glob("episode_*.h5"):
            try:
                existing.append(int(path.stem.split("_")[-1]))
            except ValueError:
                continue
        return max(existing, default=-1) + 1

    def _start_writer(self) -> None:
        initial_state = self._capture_env.callmethod("get_state")[0]
        episode_id = (
            f"{self._session_id}-human-s{self._level_seed}-"
            f"{self._episode_file_index:03d}"
        )
        path = self._output_dir / f"episode_{self._episode_file_index:03d}.h5"
        self._writer = EpisodeWriter(
            path,
            metadata={
                "episode_id": episode_id,
                "level_seed": self._level_seed,
                "source_policy": "human",
                "distribution_mode": self._distribution_mode,
                "procgen_version": "0.10.7",
                "collector_git_commit": current_git_commit(),
                "collection_id": self._session_id,
                "input_device": "keyboard",
                "human_confirmed": True,
                "action_count": 15,
                "observation_shape": "64,64,3",
            },
            initial_state=initial_state,
        )
        self._non_noop_actions = 0

    def _publish_writer(self, *, truncate: bool) -> None:
        if self._writer is None:
            return
        if self._writer.length == 0:
            self._writer.abort()
        else:
            if truncate:
                self._writer.mark_last_truncated()
            self._writer.set_metadata_attr(
                "human_non_noop_actions", self._non_noop_actions
            )
            idle_smoke = self._non_noop_actions == 0
            if idle_smoke:
                self._writer.relabel_source_policy("gui_smoke_idle")
                self._writer.set_metadata_attr("human_confirmed", False)
            path = self._writer.finalize()
            if idle_smoke:
                quarantine = self._output_dir.parent / "gui_smoke_idle"
                quarantine.mkdir(parents=True, exist_ok=True)
                destination = quarantine / path.name
                if destination.exists():
                    destination = quarantine / (
                        f"{self._session_id}-{path.stem}{path.suffix}"
                    )
                path.replace(destination)
                path = destination
                print("idle GUI smoke quarantined; it does not count as human data")
            self.completed_paths.append(path)
            print(f"saved_human_episode={path}")
            self._episode_file_index += 1
        self._writer = None

    def _act(self, action: np.ndarray) -> bool:
        _, observation_batch, _ = self._capture_env.observe()
        observation = observation_batch["rgb"][0].copy()
        if self._writer is None:
            self._start_writer()
        action_value = int(np.asarray(action).reshape(-1)[0])
        if action_value != 4:
            self._non_noop_actions += 1
        result = advance_collection_step(
            self._capture_env,
            action_value,
            purpose="data_collection",
        )
        event = classify_event(done=result.done, truncated=False, info=result.info)
        assert self._writer is not None
        self._writer.append(
            observation=observation,
            action=action_value,
            reward=result.reward,
            done=result.done,
            truncated=False,
            event=event,
            next_observation=result.next_observation,
            timestamp_ns=time.time_ns(),
        )

        self._last_rew = result.reward
        self._last_ob = {"rgb": result.next_observation[None, ...]}
        self._last_ac = action
        info = copy.copy(result.info)
        for key in list(info):
            if isinstance(info[key], np.ndarray):
                del info[key]
        self._episode_return += result.reward
        self._steps += 1
        self._episode_steps += 1
        self._last_info = {
            "episode_steps": self._episode_steps,
            "episode_return": self._episode_return,
            "non_noop_actions": self._non_noop_actions,
            **info,
        }
        if result.done:
            self._publish_writer(truncate=False)
        return result.done

    def _update(self, dt, keys_clicked, keys_pressed):
        # F1 state load is intentionally disabled while recording: it would splice a trajectory.
        filtered_clicked = [key for key in keys_clicked if key != "F1"]
        filtered_pressed = [key for key in keys_pressed if key != "F1"]
        Interactive._update(self, dt, filtered_clicked, filtered_pressed)
        if time.monotonic() - self._started_at >= self._max_seconds:
            self._renderer._should_close = True

    def run_capture(self) -> list[Path]:
        try:
            self.run()
        finally:
            self._publish_writer(truncate=True)
            close = getattr(self._capture_env, "close", None)
            if callable(close):
                close()
        return self.completed_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--out")
    parser.add_argument("--distribution-mode", choices=["hard", "easy"])
    parser.add_argument("--max-minutes", type=float)
    parser.add_argument("--tps", type=int)
    parser.add_argument("--session-id")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    seed = args.seed if args.seed is not None else int(config["seeds"]["primary"])
    output = Path(args.out or config["human"]["output"])
    mode = args.distribution_mode or config["environment"]["distribution_mode"]
    max_minutes = args.max_minutes or float(config["human"]["pilot_minutes"])
    tps = args.tps or int(config["human"]["frames_per_second"])
    session_id = args.session_id or time.strftime("human-%Y%m%dT%H%M%S")
    print("Controls: arrow keys; ESC closes and safely finalizes the session.")
    capture = KeyboardCapture(
        output_dir=output,
        level_seed=seed,
        distribution_mode=mode,
        session_id=session_id,
        max_minutes=max_minutes,
        tps=tps,
    )
    paths = capture.run_capture()
    print(f"human_episodes_saved={len(paths)}")


if __name__ == "__main__":
    main()
