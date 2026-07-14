"""Isolation and loader tests for the promoted CoinRun teams dataset."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.data.frame_dataset import CoinRunFrameDataset, EpisodeGroupedSampler


DATASET_ROOT = Path("data/datasets/coinrun_teams_v1")


def test_active_external_splits_have_no_team_or_seed_leakage():
    summary = json.loads((DATASET_ROOT / "summary.json").read_text(encoding="utf-8"))
    assert summary["checks"]["no_team_leakage"]
    assert summary["checks"]["no_seed_leakage"]
    assert summary["checks"]["episode_conservation"]
    assert summary["checks"]["frame_conservation"]
    assert summary["splits"]["train"]["episodes"] == 1277
    assert summary["splits"]["val"]["episodes"] == 281
    assert summary["splits"]["test"]["episodes"] == 341


def test_smoke_manifests_are_small_and_cover_every_train_team():
    summary = json.loads((DATASET_ROOT / "summary.json").read_text(encoding="utf-8"))
    assert summary["smoke"]["train"]["frames"] <= 256
    assert summary["smoke"]["val"]["frames"] <= 512
    assert summary["smoke"]["test"]["frames"] <= 256
    assert summary["smoke"]["train"]["teams"] == ["Team1", "Team2", "Team3", "Team4"]
    assert set(summary["smoke"]["train"]["by_outcome"]) == {
        "non_win_terminal",
        "truncated",
        "victory",
    }


def test_npz_smoke_loader_normalizes_real_frames():
    dataset = CoinRunFrameDataset(DATASET_ROOT / "smoke/train/manifest.jsonl")
    assert len(dataset) == 179
    assert dataset.storage_formats == ["npz_external_v1"]
    sample = dataset[0]
    assert sample["image"].shape == (3, 64, 64)
    assert sample["image"].dtype == torch.float32
    assert 0.0 <= float(sample["image"].min()) <= float(sample["image"].max()) <= 1.0
    assert sample["episode_id"].startswith("team")


def test_full_external_train_manifest_uses_lazy_loading():
    dataset = CoinRunFrameDataset(DATASET_ROOT / "splits/train/manifest.jsonl")
    assert len(dataset) == 107353
    assert dataset.lazy
    assert dataset.frames is None


def test_grouped_sampler_conserves_every_frame_once():
    dataset = CoinRunFrameDataset(
        DATASET_ROOT / "smoke/train/manifest.jsonl", eager_frame_limit=0
    )
    indices = list(EpisodeGroupedSampler(dataset, seed=123))
    assert len(indices) == len(dataset)
    assert set(indices) == set(range(len(dataset)))
