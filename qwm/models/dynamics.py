"""One-layer GRU transition. Action is embedded and concatenated with the image embedding."""
from __future__ import annotations

import torch
import torch.nn as nn

from qwm.data import schema


class GRUDynamics(nn.Module):
    """Step (embed_t, action_t, h_t) -> (h_{t+1}, latent_{t+1})."""

    def __init__(
        self,
        embed_dim: int,
        action_embed: int,
        gru_hidden: int,
        latent_dim: int,
        action_dim: int = schema.ACTION_DIM,
    ):
        super().__init__()
        if int(action_dim) != schema.ACTION_DIM:
            raise ValueError(f"action_dim is frozen at {schema.ACTION_DIM}, got {action_dim}")
        self.hidden = int(gru_hidden)
        self.latent_dim = int(latent_dim)
        self.action_fc = nn.Linear(schema.ACTION_DIM, int(action_embed))
        self.gru = nn.GRU(int(embed_dim) + int(action_embed), self.hidden, batch_first=True)
        self.latent_fc = nn.Linear(self.hidden, self.latent_dim)

    def initial_state(self, batch: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(int(batch), self.hidden, device=device, dtype=dtype)

    def step(self, embed: torch.Tensor, action: torch.Tensor, state: torch.Tensor):
        if action.shape[-1] != schema.ACTION_DIM:
            raise ValueError(f"action last dim must be {schema.ACTION_DIM}, got {action.shape[-1]}")
        features = torch.cat([embed, self.action_fc(action)], dim=-1).unsqueeze(1)
        _out, hidden = self.gru(features, state.unsqueeze(0))
        state = hidden.squeeze(0)
        return state, self.latent_fc(state)
