"""Task definitions (frozen task suite): T1, T2, T3.

Sampling (dataset/collection): tasks are drawn uniformly; unspecified task
components (T2 mode, T1 goal xy) are sampled with the episode RNG.

Spawn placement requirements honored by the environment:
  - T1: no special requirement;
  - T2: at least one UPRIGHT, tippable object (box or standing cylinder);
  - T3: at least one object whose center lies within T3_EDGE_MARGIN (0.18 m)
        of a table edge, so pushing it off is geometrically reachable.

Success rules (STICKY; frozen in schema.py; evaluated from simulator state):
  - T1: exists t: ||obj_xy(target) - goal_xy|| <= 0.06 AND on_table(target);
  - T2 knock (mode 0): exists t: NOT upright(target);
  - T2 keep  (mode 1): (exists t: contact(robot, target))
                        AND (for all t: upright(target));
  - T3: exists t: NOT on_table(target).

Rewards are deliberately simple (spec section 4): they are stored in the
dataset as a data field but are NOT a training signal for either model.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from qwm.data import schema


def sample_task(rng: np.random.Generator) -> Dict:
    """Sample a task spec (task_id, t2_mode, goal_xy) for an episode."""
    task_id = int(rng.choice(np.asarray(schema.TASK_IDS)))
    spec: Dict = {
        "task_id": task_id,
        "t2_mode": None,
        "goal_xy": None,
    }
    if task_id == schema.TASK_PUSH_TO_XY:
        goal_xy = rng.uniform(-schema.T1_GOAL_XY_LIMIT, schema.T1_GOAL_XY_LIMIT, size=2)
        spec["goal_xy"] = (float(goal_xy[0]), float(goal_xy[1]))
    elif task_id == schema.TASK_KNOCK_OR_KEEP:
        spec["t2_mode"] = int(rng.integers(0, 2))  # 0 knock, 1 keep
    return spec


def task_goal_vector(spec: Dict, target_index: int) -> np.ndarray:
    """Build the float32[4] task_goal from a spec + chosen target index."""
    goal = np.full(schema.TASK_GOAL_DIM, schema.TASK_GOAL_UNUSED, dtype=np.float32)
    goal[0] = float(target_index)
    if spec["task_id"] == schema.TASK_PUSH_TO_XY and spec["goal_xy"] is not None:
        goal[1], goal[2] = float(spec["goal_xy"][0]), float(spec["goal_xy"][1])
    if spec["task_id"] == schema.TASK_KNOCK_OR_KEEP and spec["t2_mode"] is not None:
        goal[3] = float(spec["t2_mode"])
    return goal


def parse_task_goal(goal: np.ndarray) -> Dict:
    """Inverse of task_goal_vector (for planners/evaluators)."""
    g = np.asarray(goal, dtype=np.float32).reshape(-1)
    if g.shape[0] != schema.TASK_GOAL_DIM:
        raise ValueError(f"task_goal must have dim {schema.TASK_GOAL_DIM}")
    return {
        "target_object_index": int(g[0]),
        "goal_xy": (float(g[1]), float(g[2])) if g[1] != schema.TASK_GOAL_UNUSED else None,
        "t2_mode": int(g[3]) if g[3] != schema.TASK_GOAL_UNUSED else None,
    }


def choose_target(
    rng: np.random.Generator,
    task_id: int,
    object_metas: list,
) -> int:
    """Choose the target object index among qualifying spawned objects.

    object_metas: list of dicts with keys 'upright_at_spawn' (bool) and
    'near_edge' (bool) maintained by the environment.
    Fails loud if no object qualifies (spawn logic must guarantee one).
    """
    if task_id == schema.TASK_KNOCK_OR_KEEP:
        # tippable: currently upright (knock or keep both require an upright
        # start to be meaningful)
        qualifying = [
            i for i, m in enumerate(object_metas)
            if m.get("upright_at_spawn", False)
        ]
    elif task_id == schema.TASK_OFF_TABLE:
        qualifying = [i for i, m in enumerate(object_metas) if m.get("near_edge", False)]
    else:  # T1: any object
        qualifying = list(range(len(object_metas)))
    if not qualifying:
        raise ValueError(f"no qualifying target object for task {task_id}")
    return int(rng.choice(np.asarray(qualifying)))


def evaluate_success(
    task_id: int,
    target_index: int,
    goal_xy: Optional[tuple],
    t2_mode: Optional[int],
    query: Dict[str, np.ndarray],
) -> bool:
    """Evaluate the PER-STEP success condition from a padded query dict.

    The caller (environment) makes success STICKY over the episode: once
    this returns True at any step, episode success is True (frozen rules).
    For T2 keep-mode the "never tipped" half of the rule is enforced by the
    caller via the episode-level tipped-so-far flag (see TabletopEnv).
    """
    ti = int(target_index)
    obj_xy = np.asarray(query["obj_xyz"], dtype=np.float64)[ti, :2]
    upright = float(query["obj_upright"][ti])
    contact = float(query["obj_contact"][ti])
    on_table = float(query["obj_on_table"][ti])

    if task_id == schema.TASK_PUSH_TO_XY:
        if goal_xy is None:
            raise ValueError("T1 requires goal_xy")
        dist = float(np.linalg.norm(obj_xy - np.asarray(goal_xy, dtype=np.float64)))
        return (dist <= schema.T1_DIST_THRESH) and (on_table > 0.5)
    if task_id == schema.TASK_KNOCK_OR_KEEP:
        if t2_mode is None:
            raise ValueError("T2 requires t2_mode")
        if t2_mode == schema.T2_MODE_KNOCK:
            return upright < 0.5
        # keep-mode: per-step half = currently upright AND (this step or an
        # earlier one) contact; caller ANDs it with not-tipped-so-far.
        return (upright > 0.5) and (contact > 0.5)
    if task_id == schema.TASK_OFF_TABLE:
        return on_table < 0.5
    raise ValueError(f"unknown task_id {task_id}")


def task_reward(
    task_id: int,
    target_index: int,
    goal_xy: Optional[tuple],
    t2_mode: Optional[int],
    query: Dict[str, np.ndarray],
    achieved_now: bool,
) -> float:
    """Simple shaped reward, stored as a data field (not a model signal)."""
    ti = int(target_index)
    if task_id == schema.TASK_PUSH_TO_XY:
        if goal_xy is None:
            raise ValueError("T1 requires goal_xy")
        obj_xy = np.asarray(query["obj_xyz"], dtype=np.float64)[ti, :2]
        dist = float(np.linalg.norm(obj_xy - np.asarray(goal_xy, dtype=np.float64)))
        return 2.0 if achieved_now else -dist
    if task_id == schema.TASK_KNOCK_OR_KEEP:
        if t2_mode == schema.T2_MODE_KNOCK:
            return 1.0 if achieved_now else 0.0
        # keep mode: penalize a tip hard, reward fresh contact lightly
        tipped = float(query["obj_upright"][ti]) < 0.5
        if tipped:
            return -1.0
        return 0.1 if achieved_now else 0.0
    if task_id == schema.TASK_OFF_TABLE:
        return 1.0 if achieved_now else 0.0
    raise ValueError(f"unknown task_id {task_id}")
