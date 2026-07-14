"""CPU tests for the Plan 3 convolutional autoencoder."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.data.frame_dataset import CoinRunFrameDataset
from src.evaluation.reconstruction_metrics import coin_mask
from src.models.autoencoder import ConvAutoencoder
from src.models.losses import ReconstructionLoss, soft_yellow_score, ssim_per_image
from src.training.train_autoencoder import checkpoint_payload, seed_everything


def test_shape_latent_and_output_range():
    model = ConvAutoencoder(latent_dim=64, base_channels=8)
    image = torch.rand(3, 3, 64, 64)
    latent = model.encode(image)
    reconstruction = model(image)
    assert latent.shape == (3, 64)
    assert reconstruction.shape == image.shape
    assert torch.isfinite(reconstruction).all()
    assert 0.0 <= float(reconstruction.min()) <= float(reconstruction.max()) <= 1.0


def test_detail_v2_retains_eight_by_eight_features_without_skip_connections():
    model = ConvAutoencoder(
        latent_dim=128,
        base_channels=8,
        architecture="detail_v2",
    )
    image = torch.rand(2, 3, 64, 64)
    encoded_features = model.encoder_conv(image)
    latent = model.encode(image)
    reconstruction = model.decode(latent)
    assert encoded_features.shape == (2, 32, 8, 8)
    assert latent.shape == (2, 128)
    assert reconstruction.shape == image.shape
    assert model.feature_size == 8


def test_unknown_autoencoder_architecture_is_rejected():
    with pytest.raises(ValueError, match="unknown autoencoder architecture"):
        ConvAutoencoder(architecture="not-real")


def test_loss_is_zero_for_identical_images_and_ssim_is_one():
    image = torch.rand(2, 3, 64, 64)
    criterion = ReconstructionLoss()
    values = criterion(image, image)
    assert float(values["l1"]) == pytest.approx(0.0, abs=1e-8)
    assert float(values["ssim"]) == pytest.approx(1.0, abs=1e-5)
    assert float(values["loss"]) == pytest.approx(0.0, abs=1e-5)


def test_ssim_decreases_for_a_perturbed_image():
    image = torch.rand(2, 3, 64, 64)
    identical = ssim_per_image(image, image)
    perturbed = ssim_per_image(torch.zeros_like(image), image)
    assert torch.all(perturbed < identical)


def test_backward_gradients_are_finite():
    model = ConvAutoencoder(latent_dim=16, base_channels=4)
    image = torch.rand(2, 3, 64, 64)
    loss = ReconstructionLoss()(model(image), image)["loss"]
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_real_manifest_loader_normalizes_without_losing_frames():
    dataset = CoinRunFrameDataset("data/splits/val/manifest.jsonl")
    assert len(dataset) == 600
    sample = dataset[0]
    assert sample["image"].dtype == torch.float32
    assert sample["image"].shape == (3, 64, 64)
    assert 0.0 <= float(sample["image"].min())
    assert float(sample["image"].max()) <= 1.0
    assert sample["level_seed"] == 0


def test_coin_mask_finds_a_two_by_two_yellow_object():
    image = torch.zeros(1, 3, 64, 64)
    image[:, 0, 20:22, 30:32] = 1.0
    image[:, 1, 20:22, 30:32] = 0.8
    config = {
        "evaluation": {
            "coin_red_min": 220,
            "coin_green_min": 170,
            "coin_blue_max": 40,
        }
    }
    mask = coin_mask(image, config)
    assert int(mask.sum()) == 4


def test_coin_weighting_penalizes_a_missing_coin_more_than_plain_l1():
    target = torch.zeros(1, 3, 64, 64)
    target[:, 0, 20:22, 30:32] = 1.0
    target[:, 1, 20:22, 30:32] = 0.8
    prediction = torch.zeros_like(target)
    plain = ReconstructionLoss(l1_weight=1.0, ssim_weight=0.0)(prediction, target)["l1"]
    weighted = ReconstructionLoss(
        l1_weight=1.0, ssim_weight=0.0, coin_roi_weight=10.0
    )(prediction, target)["l1"]
    assert weighted > plain


def test_normalized_coin_terms_do_not_disappear_into_global_average():
    target = torch.zeros(2, 3, 64, 64)
    target[0, 0, 20:22, 30:32] = 1.0
    target[0, 1, 20:22, 30:32] = 0.8
    prediction = torch.zeros_like(target)
    criterion = ReconstructionLoss(
        l1_weight=0.8,
        ssim_weight=0.2,
        coin_roi_loss_weight=0.25,
        coin_pixel_loss_weight=0.50,
        coin_roi_size=7,
    )
    values = criterion(prediction, target)
    assert values["coin_pixel_l1"] > values["l1"] * 100
    assert values["coin_roi_l1"] > values["l1"] * 2
    assert values["loss"] > ReconstructionLoss()(prediction, target)["loss"]


def test_coin_terms_are_zero_for_batches_without_yellow_objects():
    image = torch.zeros(2, 3, 64, 64)
    values = ReconstructionLoss(
        coin_roi_loss_weight=0.25,
        coin_pixel_loss_weight=0.50,
    )(image, image)
    assert float(values["coin_roi_l1"]) == 0.0
    assert float(values["coin_pixel_l1"]) == 0.0


def test_soft_yellow_mask_rewards_true_coin_and_penalizes_false_yellow():
    target = torch.zeros(1, 3, 64, 64)
    target[:, 0, 20:22, 30:32] = 1.0
    target[:, 1, 20:22, 30:32] = 0.8
    clean_prediction = target.clone()
    false_yellow_prediction = target.clone()
    false_yellow_prediction[:, 0, 40:55, 5:20] = 1.0
    false_yellow_prediction[:, 1, 40:55, 5:20] = 0.8
    assert float(soft_yellow_score(target)[0, 0, 20, 30]) > 0.75
    criterion = ReconstructionLoss(
        l1_weight=0.0,
        ssim_weight=0.0,
        coin_mask_loss_weight=1.0,
        coin_mask_negative_weight=5.0,
        coin_hard_negative_fraction=0.01,
    )
    clean_loss = criterion(clean_prediction, target)["coin_mask_bce"]
    false_yellow_loss = criterion(false_yellow_prediction, target)["coin_mask_bce"]
    assert false_yellow_loss > clean_loss


def test_detail_v2_can_overfit_and_retain_tiny_yellow_objects():
    seed_everything(17)
    model = ConvAutoencoder(latent_dim=32, base_channels=4, architecture="detail_v2")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = ReconstructionLoss(
        l1_weight=0.8,
        ssim_weight=0.2,
        coin_roi_loss_weight=0.25,
        coin_pixel_loss_weight=0.50,
        coin_mask_loss_weight=0.25,
        coin_mask_negative_weight=5.0,
        coin_hard_negative_fraction=0.01,
        coin_roi_size=7,
    )
    images = torch.zeros(4, 3, 64, 64)
    positions = ((10, 10), (18, 42), (40, 20), (50, 50))
    for index, (y, x) in enumerate(positions):
        images[index] += 0.03 * index
        images[index, 0, y : y + 2, x : x + 2] = 1.0
        images[index, 1, y : y + 2, x : x + 2] = 0.8
        images[index, 2, y : y + 2, x : x + 2] = 0.0
    for _ in range(160):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(images), images)["loss"]
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        reconstruction = model(images)
    original_coin = coin_mask(images, {"evaluation": {
        "coin_red_min": 220, "coin_green_min": 170, "coin_blue_max": 40
    }})
    reconstructed_coin = coin_mask(reconstruction, {"evaluation": {
        "coin_red_min": 220, "coin_green_min": 170, "coin_blue_max": 40
    }})
    recall = float((original_coin & reconstructed_coin).sum() / original_coin.sum())
    assert recall >= 0.75


def test_checkpoint_round_trip_preserves_output(tmp_path: Path):
    seed_everything(123)
    model = ConvAutoencoder(latent_dim=16, base_channels=4)
    optimizer = torch.optim.AdamW(model.parameters())
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    image = torch.rand(1, 3, 64, 64)
    expected = model(image).detach()
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        epoch=2,
        best_val_loss=0.1,
        config={"model": {"latent_dim": 16, "base_channels": 4}},
        mean_frame=torch.zeros(3, 64, 64),
        global_step=7,
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(payload, path)
    restored = ConvAutoencoder(latent_dim=16, base_channels=4)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    restored.load_state_dict(loaded["model_state"])
    assert torch.equal(expected, restored(image).detach())
    assert loaded["epoch"] == 2
    assert loaded["global_step"] == 7


def test_same_seed_produces_same_initial_model():
    seed_everything(987)
    first = ConvAutoencoder(latent_dim=16, base_channels=4)
    first_state = copy.deepcopy(first.state_dict())
    seed_everything(987)
    second = ConvAutoencoder(latent_dim=16, base_channels=4)
    assert all(torch.equal(first_state[key], value) for key, value in second.state_dict().items())


def test_overfit_one_small_batch_reduces_loss():
    seed_everything(5)
    model = ConvAutoencoder(latent_dim=32, base_channels=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    criterion = ReconstructionLoss(l1_weight=1.0, ssim_weight=0.0)
    image = torch.zeros(2, 3, 64, 64)
    image[0, :, 8:32, 8:32] = 1.0
    image[1, 0, 24:48, 16:40] = 1.0
    with torch.no_grad():
        initial = float(criterion(model(image), image)["loss"])
    for _ in range(120):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(image), image)["loss"]
        loss.backward()
        optimizer.step()
    final = float(criterion(model(image), image)["loss"])
    assert final < initial * 0.20, {"initial": initial, "final": final}
    assert final < 0.05


def test_model_code_does_not_import_or_advance_procgen():
    paths = [
        Path("src/models/autoencoder.py"),
        Path("src/training/train_autoencoder.py"),
        Path("src/evaluation/reconstruction_metrics.py"),
    ]
    source = "\n".join(path.read_text(encoding="utf-8") for path in paths)
    assert "import procgen" not in source
    assert ".act(" not in source
