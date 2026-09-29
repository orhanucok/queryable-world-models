"""Frozen horizons and metric formulas. Synthetic tensors only; no test split."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from qwm.data import schema
from qwm.evaluation.protocol import (
    HORIZONS,
    binary_accuracy,
    binary_f1,
    position_error_m,
    require_dev_split,
)
from qwm.evaluation.rollout import open_loop
from qwm.models import build_model
from qwm.utils.config import load_config


def test_horizons_are_frozen():
    assert HORIZONS == (1, 8, 16, 32)


def test_test_split_is_refused():
    with pytest.raises(RuntimeError, match="test split"):
        require_dev_split("test")
    assert require_dev_split("val") == "val"


def _vec(grip, obj, upright, contact, on_table):
    query = schema.build_query(grip, obj, upright, contact, on_table)
    return schema.flatten_query(query, normalized=True), query["object_valid_mask"]


def test_perfect_prediction_scores():
    grip = np.array([0.1, -0.2, 0.9], dtype=np.float32)
    obj = np.zeros((2, 3), dtype=np.float32)
    obj[0] = [0.2, 0.0, 0.78]
    obj[1] = [-0.1, 0.1, 0.8]
    vec, mask = _vec(grip, obj, np.array([1, 0], np.float32), np.array([0, 1], np.float32), np.ones(2, np.float32))
    pred = vec.reshape(1, -1)
    target = pred.copy()
    m = mask.reshape(1, -1)
    assert position_error_m(pred, target, m) == pytest.approx(0.0, abs=1e-5)
    assert binary_f1(pred[:, schema.OBJ_CONTACT], target[:, schema.OBJ_CONTACT], m) == 1.0
    assert binary_accuracy(pred[:, schema.OBJ_UPRIGHT], target[:, schema.OBJ_UPRIGHT], m) == 1.0
    assert binary_accuracy(pred[:, schema.OBJ_ON_TABLE], target[:, schema.OBJ_ON_TABLE], m) == 1.0


def test_position_error_is_meters_and_ignores_pad():
    grip = np.zeros(3, dtype=np.float32)
    grip[2] = 0.75
    obj = np.zeros((1, 3), dtype=np.float32)
    obj[0, 2] = 0.75
    base, _mask = _vec(grip, obj, np.ones(1, np.float32), np.zeros(1, np.float32), np.ones(1, np.float32))
    shifted = base.copy()
    shifted[schema.OBJ_XYZ][0] += 1.0  # +1 normalized x = +0.45 m
    # also shove a padded slot; it must not change the error
    shifted[schema.OBJ_XYZ][3:6] += 5.0
    mask = np.zeros((1, schema.NMAX), dtype=bool)
    mask[0, 0] = True
    err = position_error_m(shifted.reshape(1, -1), base.reshape(1, -1), mask)
    # one grip at 0 error and one object at 0.45 m
    assert err == pytest.approx(0.45 / 2.0, abs=1e-5)


def test_masked_slot_does_not_change_f1():
    target = np.zeros((1, schema.NMAX), dtype=np.float64)
    target[0, 0] = 1.0
    pred = np.zeros((1, schema.NMAX), dtype=np.float64)
    pred[0, 0] = 5.0
    pred[0, 5] = -4.0
    mask = np.zeros((1, schema.NMAX), dtype=bool)
    mask[0, 0] = True
    assert binary_f1(pred, target, mask) == 1.0
    pred[0, 0] = -5.0
    assert binary_f1(pred, target, mask) == 0.0


def test_open_loop_shapes_do_not_grow_the_model():
    cfg = load_config(__import__("pathlib").Path(__file__).resolve().parents[1] / "configs" / "model_qwm.yaml")
    model = build_model(cfg)
    before = sum(p.numel() for p in model.parameters())
    obs0 = torch.zeros(2, 3, 84, 84)
    actions = torch.zeros(2, 4, schema.ACTION_DIM)
    query0 = torch.zeros(2, schema.QUERY_DIM)
    out = open_loop(model, obs0, actions, query0)
    assert out["query"].shape == (2, 4, schema.QUERY_DIM)
    assert out["latent"].shape[1] == 4
    assert sum(p.numel() for p in model.parameters()) == before
