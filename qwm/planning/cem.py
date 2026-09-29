"""Cross-entropy method over clipped action sequences. Same budget for every model."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from qwm.data import schema
from qwm.evaluation.rollout import open_loop
from qwm.planning.costs import latent_cost, query_cost


def cem_action(
    model,
    obs: torch.Tensor,
    task_id: int,
    t2_mode: Optional[int],
    goal_xy: Optional[tuple],
    target_index: int,
    *,
    horizon: int,
    population: int,
    elites: int,
    iterations: int,
    std_min: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return the first action of the CEM mean sequence, float32[4] in [-1, 1]."""
    if elites < 1 or elites > population:
        raise ValueError(f"elites must be in 1..population, got {elites} / {population}")
    if horizon < 1 or iterations < 1:
        raise ValueError("horizon and iterations must be >= 1")
    if obs.ndim != 4 or obs.shape[0] != 1:
        raise ValueError(f"obs must be [1,C,H,W], got {tuple(obs.shape)}")
    device = obs.device
    mean = torch.zeros(horizon, schema.ACTION_DIM, device=device)
    std = torch.ones(horizon, schema.ACTION_DIM, device=device)
    has_query = getattr(model, "heads", None) is not None
    noop_latent = None
    if not has_query:
        zeros = torch.zeros(1, horizon, schema.ACTION_DIM, device=device)
        noop_latent = open_loop(model, obs, zeros)["latent"]
    for _ in range(int(iterations)):
        noise = rng.standard_normal((population, horizon, schema.ACTION_DIM)).astype(np.float32)
        actions = mean + std * torch.from_numpy(noise).to(device)
        actions = actions.clamp(schema.ACTION_LOW, schema.ACTION_HIGH)
        obs_b = obs.expand(population, -1, -1, -1).contiguous()
        rolled = open_loop(model, obs_b, actions)
        if has_query:
            cost = query_cost(rolled["query"], task_id, t2_mode, goal_xy, target_index)
        else:
            cost = latent_cost(rolled["latent"], noop_latent, task_id, t2_mode)
        pick = torch.topk(cost, int(elites), largest=False).indices
        elite = actions[pick]
        mean = elite.mean(dim=0)
        std = elite.std(dim=0, unbiased=False).clamp_min(float(std_min))
    first = mean[0].clamp(schema.ACTION_LOW, schema.ACTION_HIGH)
    return first.detach().float().cpu().numpy().astype(np.float32)
