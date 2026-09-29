"""Queryable world model: shared encoder + GRU, plus supervised query heads."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from qwm.data import schema
from qwm.models.dynamics import GRUDynamics
from qwm.models.encoder import Encoder
from qwm.models.heads import QueryHeads


def _encoder_and_dynamics(cfg: Dict) -> tuple:
    encoder = Encoder(
        image_size=int(cfg["image_size"]),
        in_channels=int(cfg["in_channels"]),
        conv_channels=list(cfg["conv_channels"]),
        embed_dim=int(cfg["embed_dim"]),
    )
    dynamics = GRUDynamics(
        embed_dim=int(cfg["embed_dim"]),
        action_embed=int(cfg["action_embed"]),
        gru_hidden=int(cfg["gru_hidden"]),
        latent_dim=int(cfg["latent_dim"]),
    )
    return encoder, dynamics


class QWM(nn.Module):
    """Predict the next latent and the next normalized query vector.

    ``obs`` is [B, L, 3, H, W] and ``action`` is [B, L, 4]. Step ``t`` uses
    ``obs[:, t]`` and ``action[:, t]`` to predict ``query[:, t+1]``. The
    returned query has length L-1.
    """

    def __init__(self, cfg: Dict):
        super().__init__()
        self.encoder, self.dynamics = _encoder_and_dynamics(cfg)
        self.heads = QueryHeads(self.dynamics.hidden)
        # Grip and object XYZ are deltas added to the previous query state.
        # Binary slices stay absolute logits. No extra parameters.
        self.residual = bool(cfg.get("residual", False))

    def _mix_residual(self, delta: torch.Tensor, prev: torch.Tensor) -> torch.Tensor:
        if prev.shape != delta.shape:
            raise ValueError(f"prev query {tuple(prev.shape)} != head delta {tuple(delta.shape)}")
        mixed = delta.clone()
        mixed[:, schema.GRIP_XYZ] = prev[:, schema.GRIP_XYZ] + delta[:, schema.GRIP_XYZ]
        mixed[:, schema.OBJ_XYZ] = prev[:, schema.OBJ_XYZ] + delta[:, schema.OBJ_XYZ]
        return mixed

    def forward(self, obs: torch.Tensor, action: torch.Tensor, prev_query: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if obs.ndim != 5 or action.ndim != 3:
            raise ValueError(f"expected obs [B,L,C,H,W] and action [B,L,4], got {obs.shape}, {action.shape}")
        if obs.shape[0] != action.shape[0] or obs.shape[1] != action.shape[1]:
            raise ValueError("obs and action batch/time dimensions differ")
        if obs.shape[1] < 2:
            raise ValueError("sequence length must be >= 2")
        if self.residual and prev_query is None:
            raise ValueError("residual QWM requires the previous query state")
        if action.shape[-1] != schema.ACTION_DIM:
            raise ValueError(f"action dim {action.shape[-1]} != {schema.ACTION_DIM}")
        batch, length = obs.shape[0], obs.shape[1]
        state = self.dynamics.initial_state(batch, obs.device, obs.dtype)
        latents = []
        queries = []
        for t in range(length - 1):
            embed = self.encoder(obs[:, t])
            state, latent = self.dynamics.step(embed, action[:, t], state)
            latents.append(latent)
            delta = self.heads(state)
            if self.residual:
                delta = self._mix_residual(delta, prev_query[:, t])
            queries.append(delta)
        return {
            "latent": torch.stack(latents, dim=1),
            "query": torch.stack(queries, dim=1),
        }

    def rollout(
        self,
        obs0: torch.Tensor,
        actions: torch.Tensor,
        query0: torch.Tensor,
        teacher: torch.Tensor = None,
        own_prob: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        """K-step residual rollout. The image is encoded only at step 0.

        ``own_prob=1`` feeds the predicted query forward (closed loop).
        ``own_prob=0`` feeds the true query at every step (teacher forcing).
        A value in between samples that choice independently at each step.
        """
        if not self.residual:
            raise RuntimeError("rollout requires residual grip/obj heads")
        if obs0.ndim != 4 or actions.ndim != 3 or query0.ndim != 2:
            raise ValueError("rollout expects obs0 [B,C,H,W], actions [B,K,4], query0 [B,QUERY_DIM]")
        if actions.shape[-1] != schema.ACTION_DIM or query0.shape[-1] != schema.QUERY_DIM:
            raise ValueError("rollout action or query width does not match the schema")
        if obs0.shape[0] != actions.shape[0] or obs0.shape[0] != query0.shape[0]:
            raise ValueError("rollout batch sizes differ")
        own_prob = float(own_prob)
        if not 0.0 <= own_prob <= 1.0:
            raise ValueError(f"own_prob must be in [0, 1], got {own_prob}")
        k = actions.shape[1]
        if own_prob < 1.0:
            if teacher is None or teacher.shape != (obs0.shape[0], k, schema.QUERY_DIM):
                raise ValueError("scheduled sampling needs teacher [B, K, QUERY_DIM]")
        state = self.dynamics.initial_state(obs0.shape[0], obs0.device, obs0.dtype)
        embed = self.encoder(obs0)
        carried = query0
        latents = []
        queries = []
        for t in range(k):
            if t > 0:
                embed = torch.zeros_like(embed)
            state, latent = self.dynamics.step(embed, actions[:, t], state)
            pred = self._mix_residual(self.heads(state), carried)
            latents.append(latent)
            queries.append(pred)
            if t + 1 < k and own_prob < 1.0:
                use_own = torch.rand(obs0.shape[0], 1, device=pred.device) < own_prob
                carried = torch.where(use_own, pred, teacher[:, t + 1])
            else:
                carried = pred
        return {
            "latent": torch.stack(latents, dim=1),
            "query": torch.stack(queries, dim=1),
        }


def trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
