"""Atomic, per-episode HDF5 storage for CoinRun transitions."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


SCHEMA_VERSION = "1.0"
EVENT_CODES = {
    "continue": 0,
    "victory": 1,
    "terminal_failure": 2,
    "truncated": 3,
}
EVENT_NAMES = {value: key for key, value in EVENT_CODES.items()}
REQUIRED_METADATA = (
    "episode_id",
    "level_seed",
    "source_policy",
    "distribution_mode",
    "procgen_version",
    "collector_git_commit",
)
TRANSITION_DATASETS = (
    "observations",
    "next_observations",
    "next_observation_valid",
    "actions",
    "rewards",
    "dones",
    "truncateds",
    "events",
    "frame_indices",
    "timestamps_ns",
    "level_seeds",
    "source_policies",
    "episode_ids",
)


def _coerce_attr(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, bytes, bool, int, float, np.number)):
        return value
    return str(value)


class EpisodeWriter:
    """Incrementally write one episode and publish it with an atomic rename."""

    def __init__(
        self,
        final_path: str | Path,
        metadata: Mapping[str, Any],
        initial_state: bytes,
    ) -> None:
        missing = [key for key in REQUIRED_METADATA if key not in metadata]
        if missing:
            raise ValueError(f"missing episode metadata: {missing}")

        self.final_path = Path(final_path)
        if self.final_path.suffix != ".h5":
            raise ValueError("episode path must use the .h5 suffix")
        self.final_path.parent.mkdir(parents=True, exist_ok=True)
        self.partial_path = self.final_path.with_suffix(".h5.partial")
        if self.final_path.exists() or self.partial_path.exists():
            raise FileExistsError(f"refusing to overwrite episode: {self.final_path}")

        self.metadata = dict(metadata)
        self._file = h5py.File(self.partial_path, "w")
        self._closed = False
        self._length = 0
        self._file.attrs["schema_version"] = SCHEMA_VERSION
        self._file.attrs["complete"] = False
        self._file.attrs["created_at_ns"] = time.time_ns()
        for key, value in self.metadata.items():
            self._file.attrs[key] = _coerce_attr(value)

        self._file.create_dataset(
            "initial_state",
            data=np.frombuffer(initial_state, dtype=np.uint8),
            dtype=np.uint8,
        )
        frame_args = {
            "shape": (0, 64, 64, 3),
            "maxshape": (None, 64, 64, 3),
            "chunks": (1, 64, 64, 3),
            "dtype": np.uint8,
            "compression": "lzf",
        }
        self._file.create_dataset("observations", **frame_args)
        self._file.create_dataset("next_observations", **frame_args)
        self._create_vector("next_observation_valid", np.bool_)
        self._create_vector("actions", np.uint8)
        self._create_vector("rewards", np.float32)
        self._create_vector("dones", np.bool_)
        self._create_vector("truncateds", np.bool_)
        self._create_vector("events", np.uint8)
        self._create_vector("frame_indices", np.int64)
        self._create_vector("timestamps_ns", np.int64)
        self._create_vector("level_seeds", np.int64)
        string_dtype = h5py.string_dtype(encoding="utf-8")
        self._create_vector("source_policies", string_dtype)
        self._create_vector("episode_ids", string_dtype)

    @property
    def length(self) -> int:
        return self._length

    def _create_vector(self, name: str, dtype: Any) -> None:
        self._file.create_dataset(
            name,
            shape=(0,),
            maxshape=(None,),
            chunks=(256,),
            dtype=dtype,
        )

    def _append_value(self, name: str, value: Any) -> None:
        dataset = self._file[name]
        dataset.resize((self._length + 1, *dataset.shape[1:]))
        dataset[self._length] = value

    def append(
        self,
        *,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        truncated: bool,
        event: str,
        next_observation: np.ndarray,
        timestamp_ns: int | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("cannot append to a closed episode")
        if event not in EVENT_CODES:
            raise ValueError(f"unknown event: {event}")
        if not 0 <= int(action) < 15:
            raise ValueError(f"invalid Procgen action: {action}")
        observation = np.asarray(observation, dtype=np.uint8)
        next_observation = np.asarray(next_observation, dtype=np.uint8)
        if observation.shape != (64, 64, 3) or next_observation.shape != (64, 64, 3):
            raise ValueError("CoinRun observations must have shape (64, 64, 3)")
        if done and truncated:
            raise ValueError("a transition cannot be terminal and truncated simultaneously")

        next_valid = not done
        safe_next = next_observation if next_valid else np.zeros_like(next_observation)
        values = {
            "observations": observation,
            "next_observations": safe_next,
            "next_observation_valid": next_valid,
            "actions": int(action),
            "rewards": float(reward),
            "dones": bool(done),
            "truncateds": bool(truncated),
            "events": EVENT_CODES[event],
            "frame_indices": self._length,
            "timestamps_ns": timestamp_ns if timestamp_ns is not None else time.time_ns(),
            "level_seeds": int(self.metadata["level_seed"]),
            "source_policies": str(self.metadata["source_policy"]),
            "episode_ids": str(self.metadata["episode_id"]),
        }
        for name, value in values.items():
            self._append_value(name, value)
        self._length += 1

    def finalize(self) -> Path:
        if self._closed:
            raise RuntimeError("episode is already closed")
        if self._length == 0:
            raise ValueError("refusing to publish an empty episode")

        final_event = EVENT_NAMES[int(self._file["events"][-1])]
        won = final_event == "victory"
        terminal_failure = final_event == "terminal_failure"
        self._file.attrs["transition_count"] = self._length
        self._file.attrs["terminal_event"] = final_event
        self._file.attrs["prev_level_complete"] = int(won)
        self._file.attrs["win"] = int(won)
        self._file.attrs["terminal_failure"] = int(terminal_failure)
        self._file.attrs["death"] = int(terminal_failure)
        self._file.attrs["death_definition"] = (
            "non-win terminal proxy; Procgen does not distinguish death from fall"
        )
        self._file.attrs["fall"] = -1
        self._file.attrs["truncated"] = int(final_event == "truncated")
        self._file.attrs["complete"] = True
        self._file.attrs["closed_at_ns"] = time.time_ns()
        self._file.flush()
        self._file.close()
        self._closed = True
        os.replace(self.partial_path, self.final_path)
        return self.final_path

    def set_metadata_attr(self, key: str, value: Any) -> None:
        if self._closed:
            raise RuntimeError("cannot update metadata on a closed episode")
        self._file.attrs[key] = _coerce_attr(value)

    def relabel_source_policy(self, source_policy: str) -> None:
        if self._closed:
            raise RuntimeError("cannot relabel a closed episode")
        self.metadata["source_policy"] = source_policy
        self._file.attrs["source_policy"] = source_policy
        if self._length:
            self._file["source_policies"][:] = source_policy

    def mark_last_truncated(self) -> None:
        if self._closed or self._length == 0:
            raise RuntimeError("cannot truncate an empty or closed episode")
        if bool(self._file["dones"][-1]):
            raise RuntimeError("a terminal episode cannot also be truncated")
        self._file["truncateds"][-1] = True
        self._file["events"][-1] = EVENT_CODES["truncated"]

    def abort(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True
        self.partial_path.unlink(missing_ok=True)


def load_episode(path: str | Path) -> dict[str, Any]:
    """Load an episode into memory for tests and offline replay."""
    with h5py.File(path, "r") as episode:
        arrays = {name: episode[name][...] for name in TRANSITION_DATASETS}
        arrays["initial_state"] = episode["initial_state"][...]
        attrs = {key: episode.attrs[key] for key in episode.attrs}
    return {"arrays": arrays, "attrs": attrs}


def validate_episode(path: str | Path) -> dict[str, Any]:
    """Validate schema, alignment, metadata coverage and completion."""
    with h5py.File(path, "r") as episode:
        missing_attrs = [key for key in REQUIRED_METADATA if key not in episode.attrs]
        if missing_attrs:
            raise ValueError(f"missing required attrs: {missing_attrs}")
        if not bool(episode.attrs.get("complete", False)):
            raise ValueError("episode was not finalized")
        lengths = {name: len(episode[name]) for name in TRANSITION_DATASETS}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"transition datasets are misaligned: {lengths}")
        length = next(iter(lengths.values()))
        if length == 0:
            raise ValueError("episode contains no transitions")
        if episode["observations"].shape[1:] != (64, 64, 3):
            raise ValueError("invalid observation shape")
        if np.any(episode["actions"][...] >= 15):
            raise ValueError("episode contains invalid actions")
        expected_indices = np.arange(length, dtype=np.int64)
        if not np.array_equal(episode["frame_indices"][...], expected_indices):
            raise ValueError("frame indices are not contiguous")
        if not np.all(episode["level_seeds"][...] == int(episode.attrs["level_seed"])):
            raise ValueError("per-transition level_seed does not match episode metadata")
        source_values = episode["source_policies"].asstr()[...]
        if not np.all(source_values == str(episode.attrs["source_policy"])):
            raise ValueError("per-transition source_policy does not match episode metadata")
        episode_values = episode["episode_ids"].asstr()[...]
        if not np.all(episode_values == str(episode.attrs["episode_id"])):
            raise ValueError("per-transition episode_id does not match episode metadata")
        done = episode["dones"][...]
        next_valid = episode["next_observation_valid"][...]
        if np.any(done & next_valid):
            raise ValueError("terminal transitions must not expose reset frames as next frames")
        return {
            "path": str(Path(path)),
            "episode_id": str(episode.attrs["episode_id"]),
            "transitions": length,
            "level_seed": int(episode.attrs["level_seed"]),
            "source_policy": str(episode.attrs["source_policy"]),
            "win": int(episode.attrs.get("win", 0)),
            "death": int(episode.attrs.get("death", 0)),
            "truncated": int(episode.attrs.get("truncated", 0)),
            "terminal_event": str(episode.attrs["terminal_event"]),
        }
