"""CEM costs and the frozen planner budget. No 50-episode rollout."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from qwm.data import schema
from qwm.evaluation.protocol import (
    PLAN_ELITES,
    PLAN_EVAL_EPISODES,
    PLAN_HORIZON,
    PLAN_ITERATIONS,
    PLAN_POPULATION,
    planning_seeds,
)
from qwm.planning.costs import latent_cost, query_cost
from qwm.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_planner_yaml_matches_protocol():
    raw = load_config(ROOT / "configs" / "planner.yaml")["planner"]
    assert raw["horizon"] == PLAN_HORIZON
    assert raw["population"] == PLAN_POPULATION
    assert raw["elites"] == PLAN_ELITES
    assert raw["iterations"] == PLAN_ITERATIONS
    assert raw["episodes"] == PLAN_EVAL_EPISODES
    seeds = planning_seeds()
    assert seeds == planning_seeds()
    assert len(seeds) == 50


def test_t1_cost_is_lower_at_the_goal():
    query = torch.zeros(2, 2, schema.QUERY_DIM)
    # slot 0 at the goal (0, 0) which is normalized 0, z = 0 so meters z = 0.75
    on = schema.OBJ_ON_TABLE.start
    query[:, :, on] = 5.0
    far = query.clone()
    far[:, :, schema.OBJ_XYZ.start] = 1.0
    near = query_cost(query, schema.TASK_PUSH_TO_XY, None, (0.0, 0.0), 0)
    away = query_cost(far, schema.TASK_PUSH_TO_XY, None, (0.0, 0.0), 0)
    assert torch.all(near < away)


def test_t3_cost_prefers_off_table_and_latent_cost_ignores_query():
    query = torch.zeros(1, 2, schema.QUERY_DIM)
    on = schema.OBJ_ON_TABLE.start
    query[:, :, on] = 5.0
    off = query.clone()
    off[:, :, on] = -5.0
    assert float(query_cost(off, schema.TASK_OFF_TABLE, None, None, 0)) < float(
        query_cost(query, schema.TASK_OFF_TABLE, None, None, 0)
    )
    latent = torch.zeros(2, 3, 4)
    latent[1] = 2.0
    noop = torch.zeros(1, 3, 4)
    keep = latent_cost(latent, noop, schema.TASK_KNOCK_OR_KEEP, schema.T2_MODE_KEEP)
    knock = latent_cost(latent, noop, schema.TASK_KNOCK_OR_KEEP, schema.T2_MODE_KNOCK)
    assert float(keep[1]) > float(keep[0])
    assert float(knock[1]) < float(knock[0])
    assert not np.isnan(float(keep.sum()))
