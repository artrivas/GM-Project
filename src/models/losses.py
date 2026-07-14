"""Differentiable reconstruction losses for images normalized to [0, 1]."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def soft_yellow_score(images: torch.Tensor, softness: float = 0.05) -> torch.Tensor:
    """Differentiable approximation of the evaluation's yellow-pixel threshold."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("yellow score expects NCHW RGB tensors")
    red_margin = (images[:, 0:1] - 220.0 / 255.0) / softness
    green_margin = (images[:, 1:2] - 170.0 / 255.0) / softness
    blue_margin = (40.0 / 255.0 - images[:, 2:3]) / softness
    # A yellow pixel must satisfy all three inequalities. The negative log-sum-exp is a
    # smooth minimum of their signed margins.
    smooth_minimum = -torch.logsumexp(
        -torch.cat((red_margin, green_margin, blue_margin), dim=1), dim=1, keepdim=True
    )
    return torch.sigmoid(smooth_minimum)


def ssim_per_image(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 7,
    data_range: float = 1.0,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("SSIM expects equal NCHW tensors")
    if window_size % 2 != 1:
        raise ValueError("SSIM window size must be odd")
    padding = window_size // 2
    mu_x = F.avg_pool2d(prediction, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(prediction * prediction, window_size, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, window_size, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(prediction * target, window_size, 1, padding) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator.clamp_min(1e-12)).mean(dim=(1, 2, 3))


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        l1_weight: float = 0.8,
        ssim_weight: float = 0.2,
        window_size: int = 7,
        coin_roi_weight: float = 1.0,
        coin_roi_loss_weight: float = 0.0,
        coin_pixel_loss_weight: float = 0.0,
        coin_mask_loss_weight: float = 0.0,
        coin_mask_negative_weight: float = 1.0,
        coin_hard_negative_fraction: float = 0.0,
        coin_roi_size: int = 7,
    ) -> None:
        super().__init__()
        weights = (
            l1_weight,
            ssim_weight,
            coin_roi_loss_weight,
            coin_pixel_loss_weight,
            coin_mask_loss_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("loss weights must be non-negative and not all zero")
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.window_size = window_size
        self.coin_roi_weight = coin_roi_weight
        self.coin_roi_loss_weight = coin_roi_loss_weight
        self.coin_pixel_loss_weight = coin_pixel_loss_weight
        self.coin_mask_loss_weight = coin_mask_loss_weight
        self.coin_mask_negative_weight = coin_mask_negative_weight
        self.coin_hard_negative_fraction = coin_hard_negative_fraction
        if coin_mask_negative_weight <= 0:
            raise ValueError("coin mask negative weight must be positive")
        if not 0.0 <= coin_hard_negative_fraction <= 1.0:
            raise ValueError("coin hard-negative fraction must be in [0, 1]")
        self.coin_roi_size = coin_roi_size
        if coin_roi_size < 1 or coin_roi_size % 2 != 1:
            raise ValueError("coin ROI size must be a positive odd integer")

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        absolute_error = (prediction - target).abs()
        coin = (
            (target[:, 0] * 255.0 >= 220)
            & (target[:, 1] * 255.0 >= 170)
            & (target[:, 2] * 255.0 <= 40)
        ).float().unsqueeze(1)
        if self.coin_roi_weight > 1.0:
            coin_roi = F.max_pool2d(coin, kernel_size=5, stride=1, padding=2)
            weights = 1.0 + (self.coin_roi_weight - 1.0) * coin_roi
            l1 = (absolute_error * weights).sum() / (weights.sum() * target.shape[1])
        else:
            l1 = absolute_error.mean()
        expanded_coin = coin.expand(-1, target.shape[1], -1, -1)
        coin_pixels = expanded_coin.sum()
        coin_pixel_l1 = (
            (absolute_error * expanded_coin).sum() / coin_pixels.clamp_min(1.0)
        )
        coin_roi = F.max_pool2d(
            coin,
            kernel_size=self.coin_roi_size,
            stride=1,
            padding=self.coin_roi_size // 2,
        ).expand(-1, target.shape[1], -1, -1)
        roi_pixels = coin_roi.sum()
        coin_roi_l1 = (absolute_error * coin_roi).sum() / roi_pixels.clamp_min(1.0)
        predicted_yellow = soft_yellow_score(prediction).clamp(1e-6, 1.0 - 1e-6)
        positive_pixels = coin.sum()
        negative_mask = 1.0 - coin
        positive_mask_loss = -(
            coin * predicted_yellow.log()
        ).sum() / positive_pixels.clamp_min(1.0)
        negative_losses = -negative_mask * torch.log1p(-predicted_yellow)
        if self.coin_hard_negative_fraction > 0:
            flattened = negative_losses.flatten(1)
            hard_count = max(
                1, int(round(flattened.shape[1] * self.coin_hard_negative_fraction))
            )
            negative_mask_loss = flattened.topk(hard_count, dim=1).values.mean()
        else:
            negative_mask_loss = negative_losses.sum() / negative_mask.sum().clamp_min(1.0)
        coin_mask_bce = (
            positive_mask_loss + self.coin_mask_negative_weight * negative_mask_loss
        )
        ssim = ssim_per_image(prediction, target, window_size=self.window_size).mean()
        dssim = (1.0 - ssim) / 2.0
        total = (
            self.l1_weight * l1
            + self.ssim_weight * dssim
            + self.coin_roi_loss_weight * coin_roi_l1
            + self.coin_pixel_loss_weight * coin_pixel_l1
            + self.coin_mask_loss_weight * coin_mask_bce
        )
        return {
            "loss": total,
            "l1": l1,
            "ssim": ssim,
            "coin_roi_l1": coin_roi_l1,
            "coin_pixel_l1": coin_pixel_l1,
            "coin_mask_bce": coin_mask_bce,
        }
