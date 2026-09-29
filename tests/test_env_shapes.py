"""Environment shape tests (STEP 2): obs/action/query-state shapes, dtypes,
padding masks, deterministic resets, task suite, episode termination."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from qwm.data import schema
from qwm.envs.tabletop import TabletopEnv
from qwm.utils.config import get as cfg_get
from qwm.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
ENV_CFG = load_config(ROOT / "configs" / "env.yaml")


@pytest.fixture()
def env():
    e = TabletopEnv(ENV_CFG)
    e.reset(seed=0)
    yield e
    e.close()


def _assert_query_shapes(env, obs, initial=False):
    n = env.num_objects
    assert obs["rgb"].shape == (schema.OBS_CHANNELS, env.res, env.res)
    assert obs["rgb"].dtype == np.uint8
    assert obs["grip_xyz"].shape == (3,)
    assert obs["grip_xyz"].dtype == np.float32
    for key in ("obj_xyz",):
        assert obs[key].shape == (schema.NMAX, 3)
    for key in ("obj_upright", "obj_contact", "obj_on_table"):
        assert obs[key].shape == (schema.NMAX,)
    assert obs["object_valid_mask"].shape == (schema.NMAX,)
    assert obs["object_valid_mask"].dtype == np.bool_
    # padding: exactly n valid slots, padded slots carry the frozen pad values
    assert obs["object_valid_mask"][:n].all()
    assert not obs["object_valid_mask"][n:].any()
    np.testing.assert_allclose(
        obs["obj_xyz"][n:],
        np.broadcast_to(schema.POS_ORIGIN, (schema.NMAX - n, 3)),
    )
    assert (obs["obj_upright"][n:] == 0.0).all()
    assert (obs["obj_contact"][n:] == 0.0).all()
    assert (obs["obj_on_table"][n:] == 0.0).all()
    # binary fields are exactly {0,1} on valid slots
    for key in ("obj_upright", "obj_contact", "obj_on_table"):
        assert set(np.unique(obs[key][:n])).issubset({0.0, 1.0})
    assert obs["task_goal"].shape == (schema.TASK_GOAL_DIM,)
    assert int(obs["task_id"]) in schema.TASK_IDS
    assert 0 <= int(obs["target_object_index"]) < n
    if initial:
        # target starts on the table (frozen success semantics); later steps
        # may legitimately show it off the table (e.g. a successful T3 push)
        assert obs["obj_on_table"][int(obs["target_object_index"])] == 1.0
    # EE starts inside the frozen workspace
    assert abs(obs["grip_xyz"][0]) <= schema.EE_XY_LIMIT + 1e-6
    assert schema.EE_Z_MIN - 1e-6 <= obs["grip_xyz"][2] <= schema.EE_Z_MAX + 1e-6


def test_reset_obs_shapes(env):
    obs = env.reset(seed=5)
    _assert_query_shapes(env, obs, initial=True)


def test_rgb_is_actually_rendered(env):
    obs = env.reset(seed=5)
    assert obs["rgb"].std() > 2.0, "camera image is constant; renderer broken?"
    assert obs["rgb"].min() >= 0 and obs["rgb"].max() <= 255


def test_action_validation_and_step(env):
    env.reset(seed=7)
    # wrong-size actions fail loud
    with pytest.raises(ValueError):
        env.step(np.zeros(3, dtype=np.float32))
    # action components outside bounds get clipped, not applied raw
    obs, reward, done, info = env.step(np.array([5.0, -5.0, 0.0, 0.5], dtype=np.float32))
    _assert_query_shapes(env, obs)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert not done
    assert "success" in info and "task_id" in info
    ee = obs["grip_xyz"]
    assert schema.EE_Z_MIN - 1e-3 <= ee[2] <= schema.EE_Z_MAX + 1e-3


def test_deterministic_reset_same_and_different_seed():
    cfg = ENV_CFG
    e1 = TabletopEnv(cfg)
    e2 = TabletopEnv(cfg)
    try:
        a = e1.reset(seed=11)
        b = e2.reset(seed=11)
        assert np.array_equal(a["rgb"], b["rgb"])
        np.testing.assert_allclose(a["obj_xyz"], b["obj_xyz"])
        np.testing.assert_allclose(a["grip_xyz"], b["grip_xyz"])
        assert a["task_id"] == b["task_id"]
        np.testing.assert_array_equal(a["task_goal"], b["task_goal"])
        # same-seed ROLL-OUTS stay identical
        acts = [np.array([x, y, 0.2, 0.3], dtype=np.float32)
                for x, y in zip(np.linspace(-1, 1, 8), np.linspace(1, -1, 8))]
        for act in acts:
            oa, ra, da, ia = e1.step(act)
            ob, rb, db, ib = e2.step(act)
            assert np.array_equal(oa["rgb"], ob["rgb"])
            np.testing.assert_allclose(oa["obj_xyz"], ob["obj_xyz"])
            assert ra == rb and da == db and ia["success"] == ib["success"]
        c = e2.reset(seed=12)
        assert not np.allclose(a["obj_xyz"], c["obj_xyz"])
    finally:
        e1.close()
        e2.close()


def test_episode_length_and_done(env):
    obs = env.reset(seed=21)
    done_at = None
    for t in range(schema.EPISODE_LENGTH):
        obs, _, done, info = env.step(np.zeros(schema.ACTION_DIM, dtype=np.float32))
        if done:
            done_at = t
            break
    assert done_at == schema.EPISODE_LENGTH - 1
    assert info["done_reason"] == "episode_length"


def test_task_suite_semantics():
    cfg = ENV_CFG
    e = TabletopEnv(cfg)
    try:
        # T1: explicit goal within the frozen goal box
        obs = e.reset(seed=33, task={"task_id": schema.TASK_PUSH_TO_XY,
                                     "t2_mode": None, "goal_xy": (0.25, -0.25)})
        assert obs["task_id"] == schema.TASK_PUSH_TO_XY
        goal = obs["task_goal"]
        assert goal[1] == pytest.approx(0.25) and goal[2] == pytest.approx(-0.25)
        assert goal[3] == schema.TASK_GOAL_UNUSED
        assert 0 <= int(goal[0]) < e.num_objects
        # T2: both modes; mode lands in the goal vector
        for mode in (schema.T2_MODE_KNOCK, schema.T2_MODE_KEEP):
            obs = e.reset(seed=34 + mode, task={"task_id": schema.TASK_KNOCK_OR_KEEP,
                                                "t2_mode": mode, "goal_xy": None})
            assert obs["task_id"] == schema.TASK_KNOCK_OR_KEEP
            assert int(obs["task_goal"][3]) == mode
            assert obs["task_goal"][1] == schema.TASK_GOAL_UNUSED
        # T3: target must start near an edge (spawn guarantee)
        obs = e.reset(seed=36, task={"task_id": schema.TASK_OFF_TABLE,
                                     "t2_mode": None, "goal_xy": None})
        assert obs["task_id"] == schema.TASK_OFF_TABLE
        ti = int(obs["target_object_index"])
        xy = obs["obj_xyz"][ti, :2]
        assert max(abs(xy[0]), abs(xy[1])) >= schema.TABLE_HALF_XY - schema.T3_EDGE_MARGIN - 1e-6
        # T2 target must be upright at spawn
        obs = e.reset(seed=37, task={"task_id": schema.TASK_KNOCK_OR_KEEP,
                                     "t2_mode": schema.T2_MODE_KNOCK, "goal_xy": None})
        ti = int(obs["target_object_index"])
        assert obs["obj_upright"][ti] == 1.0
        # unknown task fails loud
        with pytest.raises(ValueError):
            e.reset(seed=38, task={"task_id": 99, "t2_mode": None, "goal_xy": None})
    finally:
        e.close()


def test_num_objects_bounds(env):
    env.reset(seed=40)
    with pytest.raises(ValueError):
        env.reset(seed=41, num_objects=0)
    with pytest.raises(ValueError):
        env.reset(seed=42, num_objects=schema.NMAX + 1)
    # 20-object reset works (schema capacity check; STEP 3 lifts the default)
    obs = env.reset(seed=43, num_objects=schema.NMAX)
    assert env.num_objects == schema.NMAX
    assert obs["object_valid_mask"].all()
    _assert_query_shapes(env, obs, initial=True)


def test_frozen_constants_are_not_configurable():
    # res outside the two documented values fails loud
    bad = {"env": {"obs": {"res": 48}}}
    with pytest.raises(ValueError):
        TabletopEnv(bad)
    assert cfg_get(ENV_CFG, "env.obs.res", 84) in (84, 64)


# --------------------------------------------- STEP 3: multi-object tests --
def test_default_num_objects_is_eight(env):
    env.reset(seed=60)
    assert env.num_objects == 8
    assert cfg_get(ENV_CFG, "env.objects.num_objects", 1) == 8


def test_object_indexing_matches_world(env):
    """Padded slots 0..n-1 must correspond 1:1 to the real spawned objects."""
    for n in (1, 8, 20):
        obs = env.reset(seed=61 + n, num_objects=n)
        st = env.state()
        assert st["obj_xyz"].shape == (n, 3)
        np.testing.assert_allclose(obs["obj_xyz"][:n], st["obj_xyz"], atol=1e-6)
        # spawn metas agree with the padded slots (no sim steps taken yet)
        for i, meta in enumerate(env.object_metas):
            assert np.allclose(meta["xy"], obs["obj_xyz"][i, :2], atol=1e-6)
            assert obs["object_valid_mask"][i]
        assert not obs["object_valid_mask"][n:].any()


def test_target_selection_multi_object(env):
    """Task-aware target selection across many objects and seeds."""
    for seed in range(70, 78):
        for task_id in schema.TASK_IDS:
            task = {"task_id": task_id, "t2_mode": None, "goal_xy": None}
            obs = env.reset(seed=seed, task=task, num_objects=8)
            ti = int(obs["target_object_index"])
            assert 0 <= ti < env.num_objects
            if task_id == schema.TASK_KNOCK_OR_KEEP:
                # T2 target must be upright at spawn
                assert obs["obj_upright"][ti] == 1.0
            if task_id == schema.TASK_OFF_TABLE:
                # T3 target must start in the near-edge zone
                xy = obs["obj_xyz"][ti, :2]
                assert max(abs(xy[0]), abs(xy[1])) >= schema.TABLE_HALF_XY - schema.T3_EDGE_MARGIN - 1e-6
            assert obs["obj_on_table"][ti] == 1.0
            assert obs["object_valid_mask"][ti]


def test_twenty_object_reset_deterministic_and_stable():
    cfg = ENV_CFG
    e1 = TabletopEnv(cfg)
    e2 = TabletopEnv(cfg)
    try:
        a = e1.reset(seed=90, num_objects=schema.NMAX)
        b = e2.reset(seed=90, num_objects=schema.NMAX)
        np.testing.assert_allclose(a["obj_xyz"], b["obj_xyz"])
        assert np.array_equal(a["rgb"], b["rgb"])
        # a few steps stay deterministic and on-table at spawn heights
        act = np.array([0.5, -0.5, -0.2, 1.0], dtype=np.float32)
        for _ in range(3):
            oa, _, _, _ = e1.step(act)
            ob, _, _, _ = e2.step(act)
            np.testing.assert_allclose(oa["obj_xyz"], ob["obj_xyz"], atol=1e-9)
            assert np.array_equal(oa["rgb"], ob["rgb"])
        # no object fell through the table/floor during startup steps
        assert (oa["obj_xyz"][:schema.NMAX, 2] > schema.FLOOR_Z - 1e-6).all()
    finally:
        e1.close()
        e2.close()
