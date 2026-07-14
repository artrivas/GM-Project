"""Deterministic policy primitives for CoinRun data collection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


class Policy(Protocol):
    name: str

    def reset(self) -> None: ...

    def act(self, observation: np.ndarray) -> int: ...


@dataclass
class RandomPolicy:
    rng_seed: int
    action_count: int = 15
    name: str = "random"
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.rng_seed)

    def reset(self) -> None:
        return None

    def act(self, observation: np.ndarray) -> int:
        del observation
        return int(self._rng.integers(0, self.action_count))


@dataclass
class StickyPolicy:
    rng_seed: int
    repeat_probability: float
    action_count: int = 15
    name: str = "sticky"
    _rng: np.random.Generator = field(init=False, repr=False)
    _previous_action: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not 0.0 <= self.repeat_probability <= 1.0:
            raise ValueError("repeat_probability must be in [0, 1]")
        self._rng = np.random.default_rng(self.rng_seed)

    def reset(self) -> None:
        self._previous_action = None

    def act(self, observation: np.ndarray) -> int:
        del observation
        should_repeat = (
            self._previous_action is not None
            and self._rng.random() < self.repeat_probability
        )
        if not should_repeat:
            self._previous_action = int(self._rng.integers(0, self.action_count))
        return int(self._previous_action)


@dataclass
class ScriptedHopPolicy:
    """Measured pilot heuristic: short right+jump bursts followed by right movement."""

    jump_steps: int = 4
    cycle_steps: int = 12
    right_action: int = 7
    right_jump_action: int = 8
    name: str = "scripted"
    _step: int = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        self._step = 0

    def act(self, observation: np.ndarray) -> int:
        del observation
        action = (
            self.right_jump_action
            if self._step % self.cycle_steps < self.jump_steps
            else self.right_action
        )
        self._step += 1
        return action


def make_policy(
    name: str,
    *,
    rng_seed: int,
    action_count: int = 15,
    sticky_probability: float = 0.7,
) -> Policy:
    if name == "random":
        return RandomPolicy(rng_seed=rng_seed, action_count=action_count)
    if name == "sticky":
        return StickyPolicy(
            rng_seed=rng_seed,
            repeat_probability=sticky_probability,
            action_count=action_count,
        )
    if name == "scripted":
        return ScriptedHopPolicy()
    if name == "trained":
        raise NotImplementedError(
            "trained policy is optional and requires an explicitly configured checkpoint"
        )
    raise ValueError(f"unknown policy: {name}")
