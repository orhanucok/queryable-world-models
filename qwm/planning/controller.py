"""Receding-horizon controller: plan a sequence, execute only the first action."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from qwm.planning.cem import cem_action


def first_action(
    model,
    obs_rgb: np.ndarray,
    task_id: int,
    t2_mode: Optional[int],
    goal_xy: Optional[tuple],
    target_index: int,
    planner: dict,
    rng: np.random.Generator,
    device: torch.device,
) -> np.ndarray:
    """Normalize one uint8 frame and return the action CEM would execute now."""
    frame = np.asarray(obs_rgb)
    if frame.dtype != np.uint8 or frame.ndim != 3:
        raise ValueError(f"obs rgb must be uint8 CHW, got {frame.dtype} {frame.shape}")
    obs = torch.from_numpy(np.ascontiguousarray(frame)).to(device=device, dtype=torch.float32)
    obs = obs.mul_(1.0 / 255.0).unsqueeze(0)
    return cem_action(
        model,
        obs,
        int(task_id),
        None if t2_mode is None else int(t2_mode),
        goal_xy,
        int(target_index),
        horizon=int(planner["horizon"]),
        population=int(planner["population"]),
        elites=int(planner["elites"]),
        iterations=int(planner["iterations"]),
        std_min=float(planner["std_min"]),
        rng=rng,
    )
