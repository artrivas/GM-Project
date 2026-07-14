"""Compact convolutional autoencoder for 64x64 CoinRun frames."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
import yaml


def _groups(channels: int) -> int:
    for groups in range(min(8, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class DownBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
        )


class UpBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_groups(out_channels), out_channels),
            nn.SiLU(),
        )


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 64, base_channels: int = 32) -> None:
        super().__init__()
        b = base_channels
        self.latent_dim = latent_dim
        self.base_channels = b
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, b, 3, padding=1),
            nn.SiLU(),
            DownBlock(b, b),
            DownBlock(b, b * 2),
            DownBlock(b * 2, b * 3),
            DownBlock(b * 3, b * 4),
        )
        self.encoder_linear = nn.Linear(b * 4 * 4 * 4, latent_dim)
        self.decoder_linear = nn.Linear(latent_dim, b * 4 * 4 * 4)
        self.decoder_conv = nn.Sequential(
            UpBlock(b * 4, b * 3),
            UpBlock(b * 3, b * 2),
            UpBlock(b * 2, b),
            UpBlock(b, b),
            nn.Conv2d(b, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        features = self.encoder_conv(image)
        return self.encoder_linear(features.flatten(1))

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        features = self.decoder_linear(latent).reshape(
            latent.shape[0], self.base_channels * 4, 4, 4
        )
        return self.decoder_conv(features)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(image))


def build_autoencoder(config: dict) -> ConvAutoencoder:
    model_config = config["model"]
    return ConvAutoencoder(
        latent_dim=int(model_config["latent_dim"]),
        base_channels=int(model_config["base_channels"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device(args.device)
    model = build_autoencoder(config).to(device)
    if args.smoke_test:
        image = torch.rand(2, 3, 64, 64, device=device)
        with torch.no_grad():
            latent = model.encode(image)
            reconstruction = model(image)
        print(
            json.dumps(
                {
                    "device": str(device),
                    "parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "input_shape": list(image.shape),
                    "latent_shape": list(latent.shape),
                    "output_shape": list(reconstruction.shape),
                    "output_min": float(reconstruction.min()),
                    "output_max": float(reconstruction.max()),
                },
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
