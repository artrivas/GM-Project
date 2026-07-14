"""Tests for the independent autoencoder audit calculations."""

from __future__ import annotations

import torch

from src.evaluation.independent_reconstruction_review import (
    choose_diverse,
    independent_ssim,
    yellow_object_mask,
)


def test_independent_ssim_is_one_for_identical_images() -> None:
    image = torch.rand(3, 3, 64, 64)
    assert torch.allclose(independent_ssim(image, image), torch.ones(3), atol=1e-5)


def test_independent_coin_mask_detects_known_yellow_pixels() -> None:
    image = torch.zeros(1, 3, 64, 64)
    image[:, 0, 10:12, 20:22] = 1.0
    image[:, 1, 10:12, 20:22] = 0.8
    config = {
        "evaluation": {"coin_red_min": 220, "coin_green_min": 170, "coin_blue_max": 40}
    }
    assert int(yellow_object_mask(image, config).sum()) == 4


def test_diverse_selection_is_deterministic_and_unique() -> None:
    candidates = [
        {
            "episode_id": f"ep-{index // 2}",
            "frame_index": index,
            "target": torch.full((3, 64, 64), index / 10),
        }
        for index in range(8)
    ]
    first = choose_diverse(candidates, 5)
    second = choose_diverse(list(reversed(candidates)), 5)
    first_ids = [(row["episode_id"], row["frame_index"]) for row in first]
    second_ids = [(row["episode_id"], row["frame_index"]) for row in second]
    assert first_ids == second_ids
    assert len(set(first_ids)) == 5
