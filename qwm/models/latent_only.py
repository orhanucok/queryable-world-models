"""Latent-only baseline: the QWM encoder and GRU with no query-state heads."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from qwm.data import schema
from qwm.models.qwm import _encoder_and_dynamics


class LatentOnly(nn.Module):
    """Same transition as QWM. The only output is the next latent."""

    def __init__(self, cfg: Dict):
        super().__init__()
        self.encoder, self.dynamics = _encoder_and_dynamics(cfg)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> Dict[str, torch.Tensor]:
        if obs.ndim != 5 or action.ndim != 3:
            raise ValueError(f"expected obs [B,L,C,H,W] and action [B,L,4], got {obs.shape}, {action.shape}")
        if obs.shape[1] < 2:
            raise ValueError("sequence length must be >= 2")
        if action.shape[-1] != schema.ACTION_DIM:
            raise ValueError(f"action dim {action.shape[-1]} != {schema.ACTION_DIM}")
        batch, length = obs.shape[0], obs.shape[1]
        state = self.dynamics.initial_state(batch, obs.device, obs.dtype)
        latents = []
        for t in range(length - 1):
            embed = self.encoder(obs[:, t])
            state, latent = self.dynamics.step(embed, action[:, t], state)
            latents.append(latent)
        return {"latent": torch.stack(latents, dim=1)}
