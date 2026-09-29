"""Masked query-state losses. Position slices are MSE; binary slices are BCE-with-logits.

Object terms are multiplied by ``object_valid_mask`` at the target time.
Gripper position is never masked. An empty object mask fails loud.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from qwm.data import schema

HEAD_NAMES = ("grip_xyz", "obj_xyz", "upright", "contact", "on_table")


def latent_prediction_loss(model, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """MSE from the transition latent to a stop-grad projection of the next frame.

    ``embed_dim`` equals ``gru_hidden`` in the frozen configs, so the existing
    ``latent_fc`` can read the next image embedding directly. No new
    parameters and no query labels.
    """
    if obs.ndim != 5 or action.shape[-1] != schema.ACTION_DIM:
        raise ValueError(f"latent loss expects obs [B,L,C,H,W] and action [B,L,4], got {obs.shape} {action.shape}")
    if model.encoder.embed_dim != model.dynamics.hidden:
        raise RuntimeError(
            "latent loss reuses latent_fc on the image embedding; "
            f"embed_dim {model.encoder.embed_dim} != gru_hidden {model.dynamics.hidden}"
        )
    batch = obs.shape[0]
    steps = obs.shape[1] - 1
    if steps < 1:
        raise ValueError("latent loss needs sequence length >= 2")
    state = model.dynamics.initial_state(batch, obs.device, obs.dtype)
    total = obs.new_zeros(())
    for t in range(steps):
        state, z_pred = model.dynamics.step(model.encoder(obs[:, t]), action[:, t], state)
        with torch.no_grad():
            z_tgt = model.dynamics.latent_fc(model.encoder(obs[:, t + 1]))
        total = total + F.mse_loss(z_pred, z_tgt)
    return total / steps


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum()
    if int(denom.detach().item()) <= 0:
        raise RuntimeError("object_valid_mask is empty; refusing to zero the object loss")
    return (values * mask).sum() / denom


def query_losses(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: Dict[str, float],
) -> tuple:
    """Return ``(weighted_total, unweighted_head_dict)``.

    pred/target: [B, T, QUERY_DIM]. Binary channels of ``pred`` are logits.
    mask: [B, T, NMAX] at the same times as ``target``.
    """
    if pred.shape != target.shape or pred.shape[-1] != schema.QUERY_DIM:
        raise ValueError(f"query tensors must be [B,T,{schema.QUERY_DIM}], got {pred.shape} {target.shape}")
    if mask.shape != pred.shape[:2] + (schema.NMAX,):
        raise ValueError(f"mask shape {tuple(mask.shape)} does not match {tuple(pred.shape[:2])}")
    missing = [name for name in HEAD_NAMES if name not in weights]
    if missing:
        raise KeyError(f"loss weights missing {missing}")

    grip_sq = (pred[..., schema.GRIP_XYZ] - target[..., schema.GRIP_XYZ]) ** 2
    grip = grip_sq.mean()

    obj_pred = pred[..., schema.OBJ_XYZ].reshape(*pred.shape[:2], schema.NMAX, 3)
    obj_tgt = target[..., schema.OBJ_XYZ].reshape(*target.shape[:2], schema.NMAX, 3)
    obj_mask = mask.unsqueeze(-1).to(dtype=obj_pred.dtype).expand_as(obj_pred)
    obj = _masked_mean((obj_pred - obj_tgt) ** 2, obj_mask)

    def binary(sl):
        bce = F.binary_cross_entropy_with_logits(pred[..., sl], target[..., sl], reduction="none")
        return _masked_mean(bce, mask.to(dtype=bce.dtype))

    parts = {
        "grip_xyz": grip,
        "obj_xyz": obj,
        "upright": binary(schema.OBJ_UPRIGHT),
        "contact": binary(schema.OBJ_CONTACT),
        "on_table": binary(schema.OBJ_ON_TABLE),
    }
    total = parts["grip_xyz"] * float(weights["grip_xyz"])
    for name in HEAD_NAMES[1:]:
        total = total + parts[name] * float(weights[name])
    logs = {name: float(parts[name].detach()) for name in HEAD_NAMES}
    logs["total"] = float(total.detach())
    return total, logs
