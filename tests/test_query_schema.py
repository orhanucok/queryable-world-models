"""Frozen query-schema invariant tests (single source of truth:
qwm/data/schema.py, documented in schema.md). These tests exist to make
SILENT protocol drift fail loud (spec sections 6, 13, 15)."""
from __future__ import annotations

import numpy as np
import pytest

from qwm.data import schema


# ------------------------------------------------------------ layout ----
def test_query_layout_is_frozen():
    assert schema.NMAX == 20
    assert schema.QUERY_DIM == 123 == 3 + 6 * schema.NMAX
    # exact slice table (indices are part of the frozen schema)
    assert schema.GRIP_XYZ == slice(0, 3)
    assert schema.OBJ_XYZ == slice(3, 63)
    assert schema.OBJ_UPRIGHT == slice(63, 83)
    assert schema.OBJ_CONTACT == slice(83, 103)
    assert schema.OBJ_ON_TABLE == slice(103, 123)
    # contiguous, non-overlapping, covering the full vector
    bounds = [
        (schema.GRIP_XYZ, 0), (schema.OBJ_XYZ, 3), (schema.OBJ_UPRIGHT, 63),
        (schema.OBJ_CONTACT, 83), (schema.OBJ_ON_TABLE, 103),
    ]
    prev_stop = 0
    for sl, expected_start in bounds:
        assert sl.start == expected_start == prev_stop
        prev_stop = sl.stop
    assert prev_stop == schema.QUERY_DIM


def test_frozen_sizes_and_thresholds():
    assert schema.ACTION_DIM == 4
    assert schema.EPISODE_LENGTH == 50
    assert schema.DEFAULT_RES == 84
    assert schema.FALLBACK_RES == 64
    assert schema.OBS_CHANNELS == 3
    assert schema.SCHEMA_VERSION == "1.0.0"
    # world geometry
    assert schema.TABLE_HALF_XY == 0.45
    assert schema.TABLE_TOP_Z == 0.75
    assert schema.FLOOR_Z == 0.0
    # action space
    assert schema.ACTION_LOW == -1.0 and schema.ACTION_HIGH == 1.0
    assert schema.ACTION_XYZ_SCALE == 0.05
    assert schema.GRIPPER_CLOSED_Q == 0.012
    assert schema.GRIPPER_OPEN_Q == 0.045
    # EE workspace
    assert (schema.EE_XY_LIMIT, schema.EE_Z_MIN, schema.EE_Z_MAX) == (0.40, 0.86, 1.06)
    # label thresholds / task rules
    assert schema.UPRIGHT_TILT_DEG == 25.0
    assert schema.ON_TABLE_Z_MIN == 0.72
    assert schema.ON_TABLE_Z_MAX == 1.05
    assert schema.T1_DIST_THRESH == 0.06
    assert schema.T1_GOAL_XY_LIMIT == 0.30
    assert schema.T3_EDGE_MARGIN == 0.18
    assert schema.TASK_IDS == (1, 2, 3)
    assert schema.TASK_GOAL_DIM == 4
    assert schema.TASK_GOAL_UNUSED == -1.0
    assert (schema.T2_MODE_KNOCK, schema.T2_MODE_KEEP) == (0, 1)


def test_normalization_constants_and_roundtrip():
    np.testing.assert_allclose(schema.POS_ORIGIN, [0.0, 0.0, 0.75])
    np.testing.assert_allclose(schema.POS_SCALE, [0.45, 0.45, 0.30])
    rng = np.random.default_rng(0)
    pts = rng.uniform(-1.0, 1.2, size=(64, 3)) * [0.6, 0.6, 0.35] + [0, 0, 0.75]
    norm = schema.normalize_xyz(pts)
    back = schema.denormalize_xyz(norm)
    np.testing.assert_allclose(back, pts, atol=1e-6)
    # on-table points map roughly into [-1, 1]
    on_table = np.array([[0.4, -0.44, 0.78], [0.0, 0.0, 0.75], [-0.45, 0.45, 1.0]])
    nrm = schema.normalize_xyz(on_table)
    assert np.all(np.abs(nrm[:, :2]) <= 1.0001)
    assert nrm[1, 2] == pytest.approx(0.0)


# ---------------------------------------------------- build/flatten ----
def test_build_query_padding_conventions():
    n = 3
    grip = np.array([0.1, -0.2, 0.95], dtype=np.float32)
    obj = np.array([[0.05, 0.06, 0.78], [-0.1, 0.0, 0.80], [0.2, 0.2, 0.76]], dtype=np.float32)
    upright = np.array([1.0, 0.0, 1.0])
    contact = np.array([0.0, 1.0, 0.0])
    on_table = np.array([1.0, 1.0, 0.0])
    q = schema.build_query(grip, obj, upright, contact, on_table)

    assert q["object_valid_mask"].dtype == np.bool_
    assert q["object_valid_mask"][:n].all() and not q["object_valid_mask"][n:].any()
    # padded slots: POS_ORIGIN in METERS (normalized 0) + zero flags
    np.testing.assert_allclose(q["obj_xyz"][n:], np.broadcast_to(schema.POS_ORIGIN, (schema.NMAX - n, 3)))
    for key in ("obj_upright", "obj_contact", "obj_on_table"):
        assert (q[key][n:] == 0.0).all()
        np.testing.assert_allclose(q[key][:n], {"obj_upright": upright, "obj_contact": contact, "obj_on_table": on_table}[key])
    # n out of bounds fails loud
    with pytest.raises(ValueError):
        schema.build_query(grip, np.zeros((schema.NMAX + 1, 3)), np.zeros(schema.NMAX + 1),
                           np.zeros(schema.NMAX + 1), np.zeros(schema.NMAX + 1))


def test_flatten_unflatten_roundtrip_both_spaces():
    n = 4
    grip = np.array([0.0, 0.1, 0.92], dtype=np.float32)
    rng = np.random.default_rng(7)
    obj = rng.uniform(-0.4, 0.4, size=(n, 3)).astype(np.float32) + np.array([0, 0, 0.75], np.float32)
    obj[n - 1] = [0.33, -0.33, 0.76]
    upright = (rng.random(n) > 0.5).astype(np.float32)
    contact = (rng.random(n) > 0.7).astype(np.float32)
    on_table = (rng.random(n) > 0.2).astype(np.float32)
    q = schema.build_query(grip, obj, upright, contact, on_table)

    for normalized in (False, True):
        vec = schema.flatten_query(q, normalized=normalized)
        assert vec.shape == (schema.QUERY_DIM,) and vec.dtype == np.float32
        back = schema.unflatten_query(vec, normalized=normalized)
        np.testing.assert_allclose(back["grip_xyz"], q["grip_xyz"], atol=1e-6)
        np.testing.assert_allclose(back["obj_xyz"][:n], obj, atol=1e-6)
        np.testing.assert_allclose(back["obj_upright"][:n], upright, atol=1e-6)
        np.testing.assert_allclose(back["obj_contact"][:n], contact, atol=1e-6)
        np.testing.assert_allclose(back["obj_on_table"][:n], on_table, atol=1e-6)
        # padded slots map back to the pad convention
        if normalized:
            # normalized zero denormalizes to POS_ORIGIN (the meter-space pad)
            np.testing.assert_allclose(
                back["obj_xyz"][n:], np.broadcast_to(schema.POS_ORIGIN, (schema.NMAX - n, 3)),
                atol=1e-6,
            )
        else:
            np.testing.assert_allclose(
                back["obj_xyz"][n:],
                np.broadcast_to(schema.POS_ORIGIN, (schema.NMAX - n, 3)),
                atol=1e-6,
            )


def test_flatten_layout_indices_are_exact():
    # distinctive values verify the DOCUMENTED index table, not just roundtrip
    n = 2
    grip = np.array([0.111, 0.222, 0.333], dtype=np.float32)
    obj = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    upright = np.array([1.0, 0.0])
    contact = np.array([0.0, 1.0])
    on_table = np.array([1.0, 1.0])
    q = schema.build_query(grip, obj, upright, contact, on_table)
    vec = schema.flatten_query(q, normalized=False)
    # [0:3) grip
    np.testing.assert_allclose(vec[schema.GRIP_XYZ], grip)
    # [3:63) obj slot-major: slot0 = [1,2,3], slot1 = [4,5,6]
    np.testing.assert_allclose(vec[schema.OBJ_XYZ][:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(vec[schema.OBJ_XYZ][3:6], [4.0, 5.0, 6.0])
    # [63:83) upright
    np.testing.assert_allclose(vec[schema.OBJ_UPRIGHT][:2], [1.0, 0.0])
    assert (vec[schema.OBJ_UPRIGHT][2:] == 0.0).all()
    # [83:103) contact
    np.testing.assert_allclose(vec[schema.OBJ_CONTACT][:2], [0.0, 1.0])
    # [103:123) on_table
    np.testing.assert_allclose(vec[schema.OBJ_ON_TABLE][:2], [1.0, 1.0])
    # accessors agree with the slices
    np.testing.assert_allclose(schema.query_grip(vec), grip)
    np.testing.assert_allclose(schema.query_obj_xyz(vec)[:n], obj)
    np.testing.assert_allclose(schema.query_binary(vec, "obj_upright")[:2], upright)
    with pytest.raises(KeyError):
        schema.query_binary(vec, "obj_nonexistent")


def test_episode_field_specs_consistent():
    fields = schema.EPISODE_FIELDS
    for name in ("obs_rgb", "action", "reward", "done", "task_id", "task_goal",
                 "grip_xyz", "obj_xyz", "obj_upright", "obj_contact",
                 "obj_on_table", "object_valid_mask", "target_object_index",
                 "episode_id", "timestep"):
        assert name in fields, name
    dt, shape = fields["obj_xyz"]
    assert dt is np.float32 and shape == (schema.NMAX, 3)
    dt, shape = fields["action"]
    assert dt is np.float32 and shape == (schema.ACTION_DIM,)
    dt, shape = fields["task_goal"]
    assert dt is np.float32 and shape == (schema.TASK_GOAL_DIM,)
    dt, shape = fields["obs_rgb"]
    assert dt is np.uint8 and shape == (schema.OBS_CHANNELS, schema.DEFAULT_RES, schema.DEFAULT_RES)
    dt, _ = fields["object_valid_mask"]
    assert dt is np.bool_
    for name in ("reward", "task_id", "target_object_index", "episode_id", "timestep"):
        assert fields[name][1] == ()
    assert schema.REQUIRED_EPISODE_META == frozenset({
        "collection_source", "seed", "n_objects", "task_id", "t2_mode", "success", "res"})
    assert schema.QUERY_KEYS == ("grip_xyz", "obj_xyz", "obj_upright",
                                 "obj_contact", "obj_on_table", "object_valid_mask")


def test_task_goal_vector_layout():
    from qwm.envs.tasks import parse_task_goal, task_goal_vector
    goal = task_goal_vector(
        {"task_id": schema.TASK_PUSH_TO_XY, "t2_mode": None, "goal_xy": (0.21, -0.3)}, 7
    )
    assert goal.dtype == np.float32 and goal.shape == (4,)
    # 0.21 and -0.3 are not exact in float32; compare in float32 space.
    np.testing.assert_allclose(
        goal, np.array([7.0, 0.21, -0.3, -1.0], dtype=np.float32)
    )
    parsed = parse_task_goal(goal)
    assert parsed["target_object_index"] == 7
    assert parsed["goal_xy"][0] == pytest.approx(0.21, abs=1e-6)
    assert parsed["goal_xy"][1] == pytest.approx(-0.3, abs=1e-6)
    assert parsed["t2_mode"] is None
    # T2 encodes the mode, T1/T3 leave mode unused
    goal2 = task_goal_vector(
        {"task_id": schema.TASK_KNOCK_OR_KEEP, "t2_mode": schema.T2_MODE_KEEP, "goal_xy": None}, 3
    )
    assert goal2[1] == -1.0 and goal2[2] == -1.0 and goal2[3] == 1.0
