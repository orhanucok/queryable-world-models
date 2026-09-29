"""Frozen horizon-evaluation protocol. This module is the source of truth.

Horizons, the logit decision threshold, and the metric formulas live here.
The test split is not a development split: callers must pass ``val``.
Planner budget is frozen here and must match configs/planner.yaml.
Planning episodes are fresh simulator seeds, shared by both models.
They are not the 20-episode dataset test split.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from qwm.data import schema

HORIZONS = (1, 8, 16, 32)
# Closed-loop CEM budget. Both models use these exact numbers.
PLAN_EVAL_EPISODES = 50
PLAN_EVAL_SEED = 0
PLAN_HORIZON = 8
PLAN_POPULATION = 64
PLAN_ELITES = 4
PLAN_ITERATIONS = 4
PLAN_STD_MIN = 0.1
# Binary heads emit logits. Positive class iff logit > 0, i.e. probability > 0.5.
LOGIT_THRESHOLD = 0.0
ALLOWED_DEV_SPLITS = ("train", "val")


def require_dev_split(name: str) -> str:
    """Refuse the test split. Training updates use train; checkpointing and this eval use val."""
    if name == "test":
        raise RuntimeError("test split is off limits; use train for updates and val for selection/eval")
    if name not in ALLOWED_DEV_SPLITS:
        raise ValueError(f"split must be one of {ALLOWED_DEV_SPLITS}, got {name!r}")
    return name


def planning_seeds() -> list:
    """Fifty frozen simulator seeds. Identical for QWM and latent-only."""
    from qwm.utils.seed import spawn_seeds

    seeds = spawn_seeds(PLAN_EVAL_SEED, PLAN_EVAL_EPISODES, "planning_eval")
    if len(seeds) != PLAN_EVAL_EPISODES or len(set(seeds)) != PLAN_EVAL_EPISODES:
        raise RuntimeError("planning seed list is not 50 unique ids")
    return seeds


def check_horizons(horizons: Sequence[int]) -> None:
    if tuple(int(h) for h in horizons) != HORIZONS:
        raise ValueError(f"horizons must be {HORIZONS}, got {list(horizons)}")


def _masked_binary(pred_logit: np.ndarray, target: np.ndarray, mask: np.ndarray):
    pred = np.asarray(pred_logit, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if pred.shape != tgt.shape or m.shape != pred.shape:
        raise ValueError(f"binary metric shapes differ: {pred.shape} {tgt.shape} {m.shape}")
    if not m.any():
        raise RuntimeError("object_valid_mask is empty")
    positive = pred[m] > LOGIT_THRESHOLD
    truth = tgt[m] > 0.5
    return positive, truth


def binary_f1(pred_logit: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Positive-class F1 on valid object slots. All-negative truth and prediction scores 1."""
    positive, truth = _masked_binary(pred_logit, target, mask)
    tp = int(np.logical_and(positive, truth).sum())
    fp = int(np.logical_and(positive, np.logical_not(truth)).sum())
    fn = int(np.logical_and(np.logical_not(positive), truth).sum())
    if tp + fp + fn == 0:
        return 1.0
    return float((2 * tp) / (2 * tp + fp + fn))


def binary_accuracy(pred_logit: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Accuracy on valid object slots. Padded slots do not count."""
    positive, truth = _masked_binary(pred_logit, target, mask)
    return float(np.mean(positive == truth))


def _position_vectors(pred: np.ndarray, target: np.ndarray, mask: np.ndarray):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if pred.shape != target.shape or pred.shape[-1] != schema.QUERY_DIM:
        raise ValueError(f"query vectors must be [N, {schema.QUERY_DIM}]")
    if m.shape != pred.shape[:-1] + (schema.NMAX,):
        raise ValueError(f"mask shape {m.shape} does not match {pred.shape}")
    grip = schema.denormalize_xyz(pred[..., schema.GRIP_XYZ]) - schema.denormalize_xyz(target[..., schema.GRIP_XYZ])
    grip_err = np.linalg.norm(grip.astype(np.float64), axis=-1)
    obj_hat = schema.denormalize_xyz(pred[..., schema.OBJ_XYZ].reshape(-1, 3)).reshape(pred.shape[:-1] + (schema.NMAX, 3))
    obj_true = schema.denormalize_xyz(target[..., schema.OBJ_XYZ].reshape(-1, 3)).reshape(target.shape[:-1] + (schema.NMAX, 3))
    obj_err = np.linalg.norm((obj_hat - obj_true).astype(np.float64), axis=-1)
    if not m.any():
        raise RuntimeError("object_valid_mask is empty")
    return grip_err, obj_err, m


def grip_l2_m(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean gripper Euclidean error in meters."""
    grip_err, _obj_err, _mask = _position_vectors(pred, target, mask)
    return float(grip_err.reshape(-1).mean())


def obj_l2_m(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean Euclidean error in meters over valid object slots. Pads are excluded."""
    _grip_err, obj_err, m = _position_vectors(pred, target, mask)
    return float(obj_err[m].mean())


def position_error_m(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    """Mean Euclidean error in meters over the gripper and valid object slots.

    ``pred`` and ``target`` are normalized query vectors [N, QUERY_DIM].
    Padded object slots are excluded. Gripper error is always included.
    """
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if pred.shape != target.shape or pred.shape[-1] != schema.QUERY_DIM:
        raise ValueError(f"query vectors must be [N, {schema.QUERY_DIM}]")
    if m.shape != pred.shape[:-1] + (schema.NMAX,):
        raise ValueError(f"mask shape {m.shape} does not match {pred.shape}")
    grip_err, obj_err, m = _position_vectors(pred, target, mask)
    return float(np.concatenate([grip_err.reshape(-1), obj_err[m]]).mean())
