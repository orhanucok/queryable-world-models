"""Open-loop rollout from the first frame.

The model has no image decoder. Step 0 encodes the real frame. Later steps
pass a zero image embedding and integrate actions through the GRU only.
Query heads, when the model has them, read that recurrent state.
This does not add parameters.
"""
from __future__ import annotations

from typing import Dict

import torch

from qwm.data import schema


@torch.no_grad()
def open_loop(model, obs0: torch.Tensor, actions: torch.Tensor, query0: torch.Tensor = None) -> Dict[str, torch.Tensor]:
    """Roll ``actions`` [B, H, 4] forward from ``obs0`` [B, 3, H, W].

    Returned tensors have time length H. Index ``k`` is the prediction of the
    state after ``actions[:, k]``, i.e. the query at offset ``k + 1``.
    """
    if obs0.ndim != 4:
        raise ValueError(f"obs0 must be [B,C,H,W], got {tuple(obs0.shape)}")
    if actions.ndim != 3 or actions.shape[-1] != schema.ACTION_DIM:
        raise ValueError(f"actions must be [B,H,{schema.ACTION_DIM}], got {tuple(actions.shape)}")
    if obs0.shape[0] != actions.shape[0]:
        raise ValueError("obs0 and actions batch sizes differ")
    residual = bool(getattr(model, "residual", False))
    if residual and (query0 is None or query0.shape != (obs0.shape[0], schema.QUERY_DIM)):
        raise ValueError("residual rollout needs query0 [B, QUERY_DIM] at the first frame")
    if residual:
        return model.rollout(obs0, actions, query0)
    state = model.dynamics.initial_state(obs0.shape[0], obs0.device, obs0.dtype)
    embed = model.encoder(obs0)
    latents = []
    queries = []
    heads = getattr(model, "heads", None)
    carried = query0
    for t in range(actions.shape[1]):
        if t > 0:
            embed = torch.zeros_like(embed)
        state, latent = model.dynamics.step(embed, actions[:, t], state)
        latents.append(latent)
        if heads is not None:
            delta = heads(state)
            if residual:
                delta = model._mix_residual(delta, carried)
                carried = delta
            queries.append(delta)
    out = {"latent": torch.stack(latents, dim=1)}
    if queries:
        out["query"] = torch.stack(queries, dim=1)
    return out
