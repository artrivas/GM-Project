"""Differentiable reconstruction losses for images normalized to [0, 1]."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


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
    ) -> None:
        super().__init__()
        if l1_weight < 0 or ssim_weight < 0 or l1_weight + ssim_weight <= 0:
            raise ValueError("loss weights must be non-negative and not both zero")
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.window_size = window_size
        self.coin_roi_weight = coin_roi_weight

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        absolute_error = (prediction - target).abs()
        if self.coin_roi_weight > 1.0:
            coin = (
                (target[:, 0] * 255.0 >= 220)
                & (target[:, 1] * 255.0 >= 170)
                & (target[:, 2] * 255.0 <= 40)
            ).float().unsqueeze(1)
            coin_roi = F.max_pool2d(coin, kernel_size=5, stride=1, padding=2)
            weights = 1.0 + (self.coin_roi_weight - 1.0) * coin_roi
            l1 = (absolute_error * weights).sum() / (weights.sum() * target.shape[1])
        else:
            l1 = absolute_error.mean()
        ssim = ssim_per_image(prediction, target, window_size=self.window_size).mean()
        dssim = (1.0 - ssim) / 2.0
        total = self.l1_weight * l1 + self.ssim_weight * dssim
        return {"loss": total, "l1": l1, "ssim": ssim}
