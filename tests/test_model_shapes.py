"""QWM / latent-only shapes, the 20M cap, frozen query slices, and masked loss."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from qwm.data import schema
from qwm.models import build_model, trainable_parameters
from qwm.models.heads import QueryHeads
from qwm.training.losses import query_losses
from qwm.training.trainer import select_overfit_indices
from qwm.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
QWM_CFG = load_config(ROOT / "configs" / "model_qwm.yaml")
LATENT_CFG = load_config(ROOT / "configs" / "model_latent.yaml")
WEIGHTS = QWM_CFG["loss"]


def _arch(cfg):
    block = dict(cfg["model"])
    block.pop("name")
    block.pop("residual", None)
    return block


def test_architectures_match_and_stay_under_20m():
    assert _arch(QWM_CFG) == _arch(LATENT_CFG)
    qwm = build_model(QWM_CFG)
    latent = build_model(LATENT_CFG)
    qwm_n = trainable_parameters(qwm)
    latent_n = trainable_parameters(latent)
    assert qwm_n < 20_000_000
    assert latent_n < 20_000_000
    assert latent_n < qwm_n
    assert qwm_n - latent_n == trainable_parameters(qwm.heads)
    assert not hasattr(latent, "heads")
    assert all(p.requires_grad for p in qwm.parameters())


def test_forward_shapes_and_frozen_head_slices():
    qwm = build_model(QWM_CFG)
    latent = build_model(LATENT_CFG)
    obs = torch.rand(2, 4, 3, 84, 84)
    action = torch.zeros(2, 4, schema.ACTION_DIM)
    prev = torch.zeros(2, 4, schema.QUERY_DIM)
    out = qwm(obs, action, prev)
    base = latent(obs, action)
    assert out["query"].shape == (2, 3, schema.QUERY_DIM)
    assert out["latent"].shape == (2, 3, QWM_CFG["model"]["latent_dim"])
    assert base["latent"].shape == out["latent"].shape
    assert "query" not in base
    heads = QueryHeads(8)
    with torch.no_grad():
        for param in heads.parameters():
            param.zero_()
        heads.grip.bias.fill_(0.25)
        heads.upright.bias.fill_(1.5)
    packed = heads(torch.zeros(2, 8))
    assert torch.all(packed[:, schema.GRIP_XYZ] == 0.25)
    assert torch.all(packed[:, schema.OBJ_UPRIGHT] == 1.5)
    assert torch.all(packed[:, schema.OBJ_XYZ] == 0)
    assert torch.all(packed[:, schema.OBJ_CONTACT] == 0)
    assert torch.all(packed[:, schema.OBJ_ON_TABLE] == 0)


def test_object_loss_ignores_padded_slots():
    target = torch.zeros(2, 3, schema.QUERY_DIM)
    mask = torch.zeros(2, 3, schema.NMAX, dtype=torch.bool)
    mask[:, :, :2] = True
    view = target[..., schema.OBJ_XYZ].reshape(2, 3, schema.NMAX, 3).clone()
    view[:, :, 5] = 9.0  # padded slot
    pred = target.clone()
    pred[..., schema.OBJ_XYZ] = view.reshape(2, 3, schema.NMAX * 3)
    _total, clean = query_losses(pred, target, mask, WEIGHTS)
    assert clean["obj_xyz"] < 1e-6
    view[:, :, 0] = 4.0  # valid slot
    pred[..., schema.OBJ_XYZ] = view.reshape(2, 3, schema.NMAX * 3)
    _total, hit = query_losses(pred, target, mask, WEIGHTS)
    assert hit["obj_xyz"] > clean["obj_xyz"]
    with pytest.raises(RuntimeError, match="empty"):
        query_losses(pred, target, torch.zeros_like(mask), WEIGHTS)


def test_overfit_selection_is_exactly_16_unique_windows():
    first = select_overfit_indices(100, 16, seed=0)
    assert first == select_overfit_indices(100, 16, seed=0)
    assert len(first) == 16 and len(set(first)) == 16
    assert min(first) >= 0 and max(first) < 100
    with pytest.raises(ValueError, match="exactly 16"):
        select_overfit_indices(100, 8, seed=0)
