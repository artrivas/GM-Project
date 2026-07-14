"""Plan 0 environment checks.

Procgen transitions are forbidden during learned-model rollouts. This module performs
exactly one simulator transition in one explicitly named smoke test; every other test
is read-only with respect to the environment.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "project.yaml"


def _close_if_supported(env: object) -> None:
    close = getattr(env, "close", None)
    if callable(close):
        close()


def test_python_version_is_supported_by_procgen_wheels() -> None:
    assert (3, 7) <= sys.version_info[:2] <= (3, 10), (
        "Procgen 0.10.7 only publishes wheels for CPython 3.7-3.10; "
        f"current interpreter is {sys.version.split()[0]}"
    )


def test_required_imports() -> None:
    for module_name in ("torch", "procgen", "numpy", "yaml", "pytest", "gym3"):
        importlib.import_module(module_name)


def test_project_config_round_trip(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    output = tmp_path / "round_trip.yaml"
    output.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
    assert yaml.safe_load(output.read_text(encoding="utf-8")) == config


def test_project_config_safety_guards() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["procgen"]["version"] == "0.10.7"
    assert config["procgen"]["num_levels"] == 1
    assert config["seed_progression"]["stage_1_fixed"] == [0]
    assert not config["runtime_policy"][
        "allow_procgen_transition_during_world_model_rollout"
    ]


def test_fixed_seed_initial_frame_is_deterministic() -> None:
    from procgen import ProcgenGym3Env

    kwargs = {
        "num": 1,
        "env_name": "coinrun",
        "start_level": 0,
        "num_levels": 1,
        "distribution_mode": "hard",
        "num_threads": 0,
    }
    first_env = ProcgenGym3Env(**kwargs)
    second_env = ProcgenGym3Env(**kwargs)
    try:
        _, first_observation, _ = first_env.observe()
        _, second_observation, _ = second_env.observe()
        np.testing.assert_array_equal(
            first_observation["rgb"], second_observation["rgb"]
        )
    finally:
        _close_if_supported(first_env)
        _close_if_supported(second_env)


def test_procgen_coinrun_single_permitted_transition() -> None:
    """Advance Procgen once, only to verify the installation in Plan 0.

    The native Gym3 API calls this operation ``act`` rather than ``step``.
    It must never be reused to render a learned-model rollout.
    """
    from procgen import ProcgenGym3Env

    env = ProcgenGym3Env(
        num=1,
        env_name="coinrun",
        start_level=0,
        num_levels=1,
        distribution_mode="hard",
        num_threads=0,
    )
    try:
        _, initial_observation, _ = env.observe()
        assert initial_observation["rgb"].shape == (1, 64, 64, 3)
        assert env.ac_space.eltype.n == 15

        env.act(np.array([0], dtype=np.int32))  # Exactly one permitted transition.
        reward, next_observation, first = env.observe()

        assert reward.shape == (1,)
        assert next_observation["rgb"].shape == (1, 64, 64, 3)
        assert first.shape == (1,)
    finally:
        _close_if_supported(env)
