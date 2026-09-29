"""Task costs for CEM. Lower is better.

QWM scores the predicted query vector. Latent-only has no query head and is
not given one. Its cost is the mean squared distance between the candidate
latent rollout and a zero-action rollout from the same image. T2-keep
minimizes that distance. T1, T2-knock, and T3 maximize it. The latent cost
does not know the goal coordinates or the object pose.
"""
from __future__ import annotations

from typing import Optional

import torch

from qwm.data import schema


def _sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)


def query_cost(
    query: torch.Tensor,
    task_id: int,
    t2_mode: Optional[int],
    goal_xy: Optional[tuple],
    target_index: int,
) -> torch.Tensor:
    """Cost [B] from predicted query [B, H, QUERY_DIM]."""
    if query.shape[-1] != schema.QUERY_DIM:
        raise ValueError(f"query last dim must be {schema.QUERY_DIM}, got {query.shape[-1]}")
    ti = int(target_index)
    if not (0 <= ti < schema.NMAX):
        raise ValueError(f"target index {ti} is outside 0..{schema.NMAX - 1}")
    upright = _sigmoid(query[:, :, schema.OBJ_UPRIGHT][:, :, ti])
    on_table = _sigmoid(query[:, :, schema.OBJ_ON_TABLE][:, :, ti])
    if task_id == schema.TASK_PUSH_TO_XY:
        if goal_xy is None:
            raise ValueError("T1 cost requires goal_xy")
        obj = query[:, :, schema.OBJ_XYZ].reshape(query.shape[0], query.shape[1], schema.NMAX, 3)
        scale = query.new_tensor(schema.POS_SCALE)
        origin = query.new_tensor(schema.POS_ORIGIN)
        xy = obj[:, :, ti, :2] * scale[:2] + origin[:2]
        goal = query.new_tensor([float(goal_xy[0]), float(goal_xy[1])])
        dist = torch.linalg.norm(xy - goal, dim=-1)
        return dist.mean(dim=1) + (1.0 - on_table).mean(dim=1)
    if task_id == schema.TASK_KNOCK_OR_KEEP:
        if t2_mode is None:
            raise ValueError("T2 cost requires t2_mode")
        if int(t2_mode) == schema.T2_MODE_KNOCK:
            return upright.mean(dim=1)
        if int(t2_mode) == schema.T2_MODE_KEEP:
            return (1.0 - upright).mean(dim=1)
        raise ValueError(f"unknown t2_mode {t2_mode}")
    if task_id == schema.TASK_OFF_TABLE:
        return on_table.mean(dim=1)
    raise ValueError(f"unknown task_id {task_id}")


def latent_cost(
    latent: torch.Tensor,
    noop: torch.Tensor,
    task_id: int,
    t2_mode: Optional[int],
) -> torch.Tensor:
    """Distance of a candidate latent rollout from the zero-action rollout.

    ``noop`` is broadcast over the population. No query tensor is read.
    """
    delta = (latent.float() - noop.float()).pow(2).mean(dim=(1, 2))
    if task_id == schema.TASK_KNOCK_OR_KEEP and int(t2_mode) == schema.T2_MODE_KEEP:
        return delta
    return -delta
