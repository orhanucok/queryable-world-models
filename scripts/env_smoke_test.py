"""Environment smoke test + proof GIF (STEP 2 gate).

Runs one short scripted rollout per task (T1/T2/T3) with a simple inline
push heuristic (ground-truth state access is allowed for the DATA ENGINE and
demos; models never see it), validates every obs/action/query-state shape,
checks deterministic resets, and writes data/env_smoke.gif.

Usage:
    python scripts/env_smoke_test.py --config configs/env.yaml
    python scripts/env_smoke_test.py --config configs/env.yaml --gif data/env_smoke.gif
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwm.data import schema                      # noqa: E402
from qwm.envs.tabletop import TabletopEnv       # noqa: E402
from qwm.utils.config import get as cfg_get     # noqa: E402
from qwm.utils.config import load_config        # noqa: E402
from qwm.utils.logging import get_logger        # noqa: E402

logger = get_logger("env_smoke_test")


# --------------------------------------------------------------------- --
def check_shapes(env: TabletopEnv, obs: dict, initial: bool = False) -> None:
    """Fail-loud shape/type validation of one observation.

    initial=True additionally asserts the target starts on the table
    (legitimately false later, e.g. after a successful T3 push).
    """
    res = env.res
    assert obs["rgb"].shape == (schema.OBS_CHANNELS, res, res), obs["rgb"].shape
    assert obs["rgb"].dtype == np.uint8
    assert obs["grip_xyz"].shape == (3,) and obs["grip_xyz"].dtype == np.float32
    assert obs["obj_xyz"].shape == (schema.NMAX, 3)
    assert obs["obj_upright"].shape == (schema.NMAX,)
    assert obs["obj_contact"].shape == (schema.NMAX,)
    assert obs["obj_on_table"].shape == (schema.NMAX,)
    assert obs["object_valid_mask"].shape == (schema.NMAX,)
    assert obs["object_valid_mask"].dtype == np.bool_
    assert obs["task_goal"].shape == (schema.TASK_GOAL_DIM,)
    assert int(obs["task_id"]) in schema.TASK_IDS
    n = env.num_objects
    assert obs["object_valid_mask"][:n].all()
    assert not obs["object_valid_mask"][n:].any()
    assert 0 <= int(obs["target_object_index"]) < n
    if initial:
        assert float(obs["obj_on_table"][int(obs["target_object_index"])]) == 1.0, \
            "target must start on the table"


# --------------------------------------------------------------------- --
def scripted_push_action(env: TabletopEnv) -> np.ndarray:
    """Simple inline heuristic: approach behind the target, push toward the
    task-relevant direction (GT state is allowed for data-engine demos)."""
    s = env.state()
    ee = np.asarray(s["grip_xyz"], dtype=np.float64)
    ti = s["target_object_index"]
    obj = np.asarray(s["obj_xyz"][ti], dtype=np.float64)
    task_id = s["task_id"]

    if task_id == schema.TASK_PUSH_TO_XY:
        goal = np.array([s["goal_xy"][0], s["goal_xy"][1]], dtype=np.float64)
        d = goal - obj[:2]
        norm = np.linalg.norm(d)
        direction = d / max(norm, 1e-6)
        aim = obj[:2] + direction * 0.10
        z_aim = min(max(obj[2] + 0.01, schema.EE_Z_MIN), schema.EE_Z_MAX)
        # if the goal is basically reached, retreat upward
        if norm < schema.T1_DIST_THRESH:
            aim = ee[:2]
            z_aim = 1.02
    elif task_id == schema.TASK_KNOCK_OR_KEEP:
        if s["t2_mode"] == schema.T2_MODE_KNOCK:
            aim = obj[:2] + 0.10  # sweep through the object sideways
            z_aim = min(max(obj[2] + 0.04, schema.EE_Z_MIN), schema.EE_Z_MAX)
        else:  # keep upright: gentle push at low height
            aim = obj[:2] + 0.06
            z_aim = min(max(obj[2] - 0.01, schema.EE_Z_MIN), schema.EE_Z_MAX)
    else:  # T3: push toward the nearest table edge
        xy = obj[:2]
        edge = np.array([
            np.sign(xy[0]) * schema.TABLE_HALF_XY if abs(xy[0]) > abs(xy[1]) else xy[0],
            np.sign(xy[1]) * schema.TABLE_HALF_XY if abs(xy[1]) >= abs(xy[0]) else xy[1],
        ], dtype=np.float64)
        d = edge - xy
        direction = d / max(np.linalg.norm(d), 1e-6)
        aim = obj[:2] + direction * 0.12
        z_aim = min(max(obj[2] + 0.01, schema.EE_Z_MIN), schema.EE_Z_MAX)

    target = np.array([aim[0], aim[1], z_aim])
    delta = target - ee
    action = np.zeros(schema.ACTION_DIM, dtype=np.float32)
    action[:3] = np.clip(delta / schema.ACTION_XYZ_SCALE, -1.0, 1.0)
    action[3] = 1.0  # keep the gripper open while pushing
    return action


# --------------------------------------------------------------------- --
def run_task_episode(env: TabletopEnv, seed: int, task: dict, frames: list) -> dict:
    obs = env.reset(seed=seed, task=task)
    check_shapes(env, obs, initial=True)
    total_r, achieved = 0.0, False
    for t in range(schema.EPISODE_LENGTH):
        a = scripted_push_action(env)
        assert a.shape == (schema.ACTION_DIM,) and np.all(a >= -1.0) and np.all(a <= 1.0)
        obs, reward, done, info = env.step(a)
        check_shapes(env, obs)
        frames.append(np.transpose(obs["rgb"], (1, 2, 0)))  # HWC for the GIF
        total_r += float(reward)
        achieved = achieved or bool(info["success"])
        assert done == (t == schema.EPISODE_LENGTH - 1)
    return {"task_id": task["task_id"], "return": round(total_r, 3),
            "success": achieved, "length": schema.EPISODE_LENGTH}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/env.yaml")
    ap.add_argument("--gif", default="data/env_smoke.gif")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(ROOT / args.config)
    logger.info("building TabletopEnv (res=%s, substeps=%s, render=%s)",
                cfg_get(cfg, "env.obs.res", 84), cfg_get(cfg, "env.physics.substeps_per_action", 20),
                cfg_get(cfg, "env.render", "direct"))
    env = TabletopEnv(cfg)
    frames: list = []
    results = []

    # --- shape + determinism checks -------------------------------------
    obs_a = env.reset(seed=args.seed)
    check_shapes(env, obs_a, initial=True)
    n = env.num_objects
    rgb_a, obj_a = obs_a["rgb"].copy(), obs_a["obj_xyz"].copy()

    obs_b = env.reset(seed=args.seed)
    check_shapes(env, obs_b, initial=True)
    if not np.array_equal(rgb_a, obs_b["rgb"]):
        raise SystemExit("FAIL: same-seed reset produced different RGB")
    if not np.allclose(obj_a, obs_b["obj_xyz"]):
        raise SystemExit("FAIL: same-seed reset produced different object state")

    obs_c = env.reset(seed=args.seed + 1234)
    check_shapes(env, obs_c, initial=True)
    if np.array_equal(rgb_a, obs_c["rgb"]):
        logger.warning("different seed gave identical RGB (possible but unlikely)")
    logger.info("deterministic reset check OK (seed=%s)", args.seed)

    # --- multi-object checks (STEP 3) -------------------------------------
    for n_check in (8, 20):
        obs_n = env.reset(seed=args.seed + 500 + n_check, num_objects=n_check)
        check_shapes(env, obs_n, initial=True)
        if env.num_objects != n_check:
            raise SystemExit(f"FAIL: requested {n_check} objects, got {env.num_objects}")
        # object indexing: padded slots 0..n-1 must match the real objects
        st = env.state()
        if not np.allclose(obs_n["obj_xyz"][:n_check], st["obj_xyz"]):
            raise SystemExit("FAIL: padded obj_xyz slots diverge from real objects")
        if not obs_n["object_valid_mask"][:n_check].all():
            raise SystemExit("FAIL: valid mask incomplete")
        ti = int(obs_n["target_object_index"])
        if not (0 <= ti < n_check):
            raise SystemExit(f"FAIL: target index {ti} out of range for n={n_check}")
        logger.info("multi-object check OK (n=%d, target=%d, near_edge=%s)",
                    n_check, ti, env.object_metas[ti]["near_edge"])
    obs = env.reset(seed=args.seed)  # restore 1-object default flow below

    # --- one short scripted rollout per task -----------------------------
    task_specs = [
        {"task_id": schema.TASK_PUSH_TO_XY, "t2_mode": None, "goal_xy": (0.2, -0.15)},
        {"task_id": schema.TASK_KNOCK_OR_KEEP, "t2_mode": schema.T2_MODE_KNOCK, "goal_xy": None},
        {"task_id": schema.TASK_KNOCK_OR_KEEP, "t2_mode": schema.T2_MODE_KEEP, "goal_xy": None},
        {"task_id": schema.TASK_OFF_TABLE, "t2_mode": None, "goal_xy": None},
    ]
    for i, spec in enumerate(task_specs):
        res = run_task_episode(env, args.seed + 100 + i, spec, frames)
        results.append(res)
        logger.info("task %s (mode=%s): return=%s success=%s",
                    res["task_id"], spec["t2_mode"], res["return"], res["success"])

    env.close()

    # --- GIF --------------------------------------------------------------
    gif_path = ROOT / args.gif
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    import imageio

    imageio.mimsave(str(gif_path), frames, duration=90, loop=0)
    size_kb = gif_path.stat().st_size / 1024.0

    print("\n=== ENV SMOKE TEST: PASS ===")
    for r in results:
        print(f"  task_id={r['task_id']} length={r['length']} return={r['return']} success={r['success']}")
    print(f"  obs rgb shape: ({schema.OBS_CHANNELS},{env.res},{env.res}) uint8")
    print(f"  query arrays: obj_xyz[{schema.NMAX},3] flags[{schema.NMAX}] mask[{schema.NMAX}]")
    print(f"  GIF: {gif_path} ({len(frames)} frames, {size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
