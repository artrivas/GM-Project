"""Unit and CPU integration tests for Plan 1 data collection."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from src.data_collection.collector import (
    advance_collection_step,
    collect_episodes,
)
from src.data_collection.episode_writer import (
    EVENT_CODES,
    EpisodeWriter,
    load_episode,
    validate_episode,
)
from src.data_collection.policies import (
    RandomPolicy,
    ScriptedHopPolicy,
    StickyPolicy,
)
from src.data_collection.quick_report import quick_report
from src.data_collection.replay import verify_with_procgen


def metadata(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "episode_id": "synthetic-000",
        "level_seed": 0,
        "source_policy": "synthetic",
        "distribution_mode": "hard",
        "procgen_version": "0.10.7",
        "collector_git_commit": "test",
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    "policy",
    [
        RandomPolicy(rng_seed=1),
        StickyPolicy(rng_seed=1, repeat_probability=0.7),
        ScriptedHopPolicy(),
    ],
)
def test_each_policy_only_emits_valid_actions(policy: object) -> None:
    observation = np.zeros((64, 64, 3), dtype=np.uint8)
    policy.reset()
    actions = [policy.act(observation) for _ in range(1_000)]
    assert all(0 <= action < 15 for action in actions)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: RandomPolicy(rng_seed=42),
        lambda: StickyPolicy(rng_seed=42, repeat_probability=0.7),
        lambda: ScriptedHopPolicy(),
    ],
)
def test_policy_action_sequence_is_deterministic(factory: object) -> None:
    observation = np.zeros((64, 64, 3), dtype=np.uint8)
    first = factory()
    second = factory()
    assert [first.act(observation) for _ in range(100)] == [
        second.act(observation) for _ in range(100)
    ]


def test_sticky_probability_validation() -> None:
    with pytest.raises(ValueError, match="repeat_probability"):
        StickyPolicy(rng_seed=0, repeat_probability=1.1)


def test_episode_round_trip_and_transition_alignment(tmp_path: Path) -> None:
    first_observation = np.arange(64 * 64 * 3, dtype=np.uint8).reshape(64, 64, 3)
    second_observation = np.flip(first_observation, axis=1).copy()
    reset_observation = np.full((64, 64, 3), 255, dtype=np.uint8)
    path = tmp_path / "episode_000.h5"
    writer = EpisodeWriter(path, metadata(), initial_state=b"state-bytes")
    writer.append(
        observation=first_observation,
        action=7,
        reward=1.25,
        done=False,
        truncated=False,
        event="continue",
        next_observation=second_observation,
        timestamp_ns=100,
    )
    writer.append(
        observation=second_observation,
        action=8,
        reward=10.0,
        done=True,
        truncated=False,
        event="victory",
        next_observation=reset_observation,
        timestamp_ns=200,
    )
    writer.finalize()

    summary = validate_episode(path)
    loaded = load_episode(path)
    arrays = loaded["arrays"]
    assert summary["transitions"] == 2
    np.testing.assert_array_equal(arrays["observations"][0], first_observation)
    np.testing.assert_array_equal(arrays["next_observations"][0], second_observation)
    np.testing.assert_array_equal(arrays["next_observations"][1], 0)
    np.testing.assert_array_equal(arrays["actions"], [7, 8])
    np.testing.assert_allclose(arrays["rewards"], [1.25, 10.0])
    np.testing.assert_array_equal(arrays["dones"], [False, True])
    np.testing.assert_array_equal(arrays["next_observation_valid"], [True, False])
    np.testing.assert_array_equal(arrays["events"], [0, 1])
    np.testing.assert_array_equal(arrays["frame_indices"], [0, 1])
    np.testing.assert_array_equal(arrays["timestamps_ns"], [100, 200])
    assert loaded["attrs"]["prev_level_complete"] == 1
    assert loaded["attrs"]["death"] == 0
    assert arrays["initial_state"].tobytes() == b"state-bytes"


def test_truncated_episode_metadata(tmp_path: Path) -> None:
    path = tmp_path / "episode_000.h5"
    observation = np.zeros((64, 64, 3), dtype=np.uint8)
    writer = EpisodeWriter(path, metadata(), initial_state=b"state")
    writer.append(
        observation=observation,
        action=4,
        reward=0.0,
        done=False,
        truncated=False,
        event="continue",
        next_observation=observation,
    )
    writer.mark_last_truncated()
    writer.finalize()
    with h5py.File(path, "r") as episode:
        assert episode.attrs["truncated"] == 1
        assert episode["truncateds"][0]
        assert episode["events"][0] == EVENT_CODES["truncated"]


def test_source_policy_can_be_relabelled_before_publication(tmp_path: Path) -> None:
    path = tmp_path / "episode_000.h5"
    observation = np.zeros((64, 64, 3), dtype=np.uint8)
    writer = EpisodeWriter(
        path,
        metadata(source_policy="human"),
        initial_state=b"state",
    )
    writer.append(
        observation=observation,
        action=4,
        reward=0.0,
        done=False,
        truncated=True,
        event="truncated",
        next_observation=observation,
    )
    writer.relabel_source_policy("gui_smoke_idle")
    writer.finalize()
    summary = validate_episode(path)
    assert summary["source_policy"] == "gui_smoke_idle"


def test_missing_metadata_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing episode metadata"):
        EpisodeWriter(tmp_path / "bad.h5", {}, initial_state=b"")


def test_world_model_rollout_cannot_advance_procgen() -> None:
    with pytest.raises(PermissionError, match="forbidden"):
        advance_collection_step(object(), 0, purpose="world_model_rollout")


def test_simulator_transition_is_centralized() -> None:
    source_root = Path(__file__).resolve().parents[1] / "src" / "data_collection"
    occurrences: list[str] = []
    forbidden_step_calls: list[str] = []
    for path in source_root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "env.act(" in text:
            occurrences.append(path.name)
        if "env.step(" in text:
            forbidden_step_calls.append(path.name)
    assert occurrences == ["collector.py"]
    assert forbidden_step_calls == []


def test_real_procgen_collection_round_trip_and_replay(tmp_path: Path) -> None:
    output = tmp_path / "real"
    results = collect_episodes(
        policy=RandomPolicy(rng_seed=123),
        level_seed=0,
        episodes=1,
        output_dir=output,
        distribution_mode="hard",
        max_steps=8,
        collection_id="pytest-real",
    )
    path = output / "episode_000.h5"
    assert results[0]["source_policy"] == "random"
    assert results[0]["level_seed"] == 0
    replay = verify_with_procgen(path)
    assert replay["simulator_match"] == 1
    assert replay["transitions_verified"] == results[0]["transitions"]


def test_same_seeds_produce_same_actions_and_frames(tmp_path: Path) -> None:
    paths = []
    for run in ("a", "b"):
        output = tmp_path / run
        collect_episodes(
            policy=RandomPolicy(rng_seed=777),
            level_seed=0,
            episodes=1,
            output_dir=output,
            distribution_mode="hard",
            max_steps=12,
            collection_id=f"determinism-{run}",
        )
        paths.append(output / "episode_000.h5")
    first = load_episode(paths[0])["arrays"]
    second = load_episode(paths[1])["arrays"]
    np.testing.assert_array_equal(first["actions"], second["actions"])
    np.testing.assert_array_equal(first["observations"], second["observations"])
    np.testing.assert_array_equal(first["rewards"], second["rewards"])
    np.testing.assert_array_equal(first["dones"], second["dones"])


def test_quick_report_counts_complete_metadata(tmp_path: Path) -> None:
    output = tmp_path / "report"
    collect_episodes(
        policy=ScriptedHopPolicy(),
        level_seed=0,
        episodes=1,
        output_dir=output,
        distribution_mode="hard",
        max_steps=100,
        collection_id="pytest-report",
    )
    report = quick_report(str(output / "episode_*.h5"))
    assert report["episodes"] == 1
    assert report["frames"] > 0
    assert report["metadata_percent"] == 100.0
