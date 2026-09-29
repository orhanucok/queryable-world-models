"""Small CNN encoder. No pretrained weights, no decoder."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Map one RGB frame [B, 3, H, W] to an embedding [B, embed_dim]."""

    def __init__(
        self,
        image_size: int,
        in_channels: int,
        conv_channels: Sequence[int],
        embed_dim: int,
    ):
        super().__init__()
        if int(image_size) <= 0 or int(in_channels) <= 0 or int(embed_dim) <= 0:
            raise ValueError("encoder sizes must be positive")
        if not conv_channels:
            raise ValueError("conv_channels must be non-empty")
        layers = []
        c_in = int(in_channels)
        for c_out in conv_channels:
            layers.append(nn.Conv2d(c_in, int(c_out), kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU())
            c_in = int(c_out)
        self.conv = nn.Sequential(*layers)
        self.image_size = int(image_size)
        self.embed_dim = int(embed_dim)
        with torch.no_grad():
            dummy = torch.zeros(1, int(in_channels), self.image_size, self.image_size)
            flat = int(self.conv(dummy).reshape(1, -1).shape[1])
        if flat <= 0:
            raise RuntimeError("encoder spatial collapse produced an empty map")
        self.fc = nn.Linear(flat, self.embed_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 4:
            raise ValueError(f"encoder expects [B,C,H,W], got {tuple(obs.shape)}")
        if obs.shape[-2] != self.image_size or obs.shape[-1] != self.image_size:
            raise ValueError(
                f"encoder image size is {self.image_size}, got {tuple(obs.shape[-2:])}"
            )
        return self.fc(self.conv(obs).reshape(obs.shape[0], -1))
