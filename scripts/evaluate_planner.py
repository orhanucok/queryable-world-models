"""Closed-loop CEM evaluation on 50 frozen simulator episodes.

    python scripts/evaluate_planner.py --config configs/eval.yaml --run runs/qwm
    python scripts/evaluate_planner.py --config configs/eval.yaml --run runs/latent_only

Both runs use the same seeds, tasks (via reset seed), and CEM budget.
The dataset test split is not loaded.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwm.data import schema                                      # noqa: E402
from qwm.envs.tabletop import TabletopEnv                        # noqa: E402
from qwm.evaluation.protocol import (                            # noqa: E402
    PLAN_ELITES,
    PLAN_EVAL_EPISODES,
    PLAN_HORIZON,
    PLAN_ITERATIONS,
    PLAN_POPULATION,
    PLAN_STD_MIN,
    planning_seeds,
)
from qwm.models import build_model, trainable_parameters         # noqa: E402
from qwm.planning.controller import first_action                 # noqa: E402
from qwm.training.checkpoints import load_checkpoint             # noqa: E402
from qwm.utils.config import get as cfg_get                      # noqa: E402
from qwm.utils.config import load_config                         # noqa: E402
from qwm.utils.device import get_device                          # noqa: E402
from qwm.utils.logging import get_logger                         # noqa: E402
from qwm.utils.seed import make_rng                              # noqa: E402

logger = get_logger("evaluate_planner")


def _planner_cfg(path: Path) -> dict:
    raw = load_config(path)["planner"]
    expected = {
        "horizon": PLAN_HORIZON,
        "population": PLAN_POPULATION,
        "elites": PLAN_ELITES,
        "iterations": PLAN_ITERATIONS,
        "std_min": PLAN_STD_MIN,
        "episodes": PLAN_EVAL_EPISODES,
    }
    for key, value in expected.items():
        if raw[key] != value:
            raise SystemExit(f"planner.{key}={raw[key]} does not match protocol {value}")
    return raw


def _summarize(outcomes):
    by_task = {}
    for row in outcomes:
        key = str(row["task_id"])
        bucket = by_task.setdefault(key, {"success": 0, "episodes": 0})
        bucket["episodes"] += 1
        bucket["success"] += int(row["success"])
    return {
        "episodes": len(outcomes),
        "success": int(sum(r["success"] for r in outcomes)),
        "success_rate": float(np.mean([r["success"] for r in outcomes])) if outcomes else 0.0,
        "by_task": by_task,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--run", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    planner_path = ROOT / cfg_get(cfg, "eval.planner_config", "configs/planner.yaml")
    planner = _planner_cfg(planner_path)
    n_eps = int(args.episodes if args.episodes is not None else planner["episodes"])
    if n_eps < 1:
        raise SystemExit("episodes must be >= 1")
    seeds = planning_seeds()
    if n_eps > len(seeds):
        raise SystemExit(f"requested {n_eps} episodes but only {len(seeds)} frozen seeds exist")
    seeds = seeds[:n_eps]
    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    meta = json.loads((run_dir / "best.json").read_text(encoding="utf-8"))
    device = get_device("auto")
    model = build_model(meta["model_config"])
    load_checkpoint(run_dir / "best.pt", model, device=str(device))
    counted = trainable_parameters(model)
    if counted != int(meta["parameters"]):
        raise SystemExit(f"parameter count {counted} != saved {meta['parameters']}")
    model.to(device)
    model.eval()
    env_cfg = load_config(ROOT / "configs" / "env.yaml")
    env = TabletopEnv(env_cfg)
    outcomes = []
    t0 = time.perf_counter()
    try:
        for index, seed in enumerate(seeds):
            obs = env.reset(seed=int(seed))
            task_id = int(env.task["task_id"])
            t2_mode = env.task["t2_mode"]
            goal_xy = env.task["goal_xy"]
            target = int(env.target_object_index)
            rng = make_rng(int(seed) + 17)
            info = {"success": False}
            for _ in range(schema.EPISODE_LENGTH):
                action = first_action(
                    model, obs["rgb"], task_id, t2_mode, goal_xy, target, planner, rng, device,
                )
                obs, _reward, _done, info = env.step(action)
            outcomes.append({
                "episode_index": index,
                "seed": int(seed),
                "task_id": task_id,
                "t2_mode": None if t2_mode is None else int(t2_mode),
                "target_object_index": target,
                "n_objects": int(env.num_objects),
                "success": bool(info["success"]),
            })
            done_n = index + 1
            elapsed = time.perf_counter() - t0
            eta = elapsed / done_n * (n_eps - done_n)
            logger.info(
                "episode %d/%d task=%s success=%s  eta %.0fs",
                done_n, n_eps, task_id, info["success"], eta,
            )
    finally:
        env.close()
    summary = _summarize(outcomes)
    payload = {
        "model": meta["kind"],
        "parameters": counted,
        "planner": {k: planner[k] for k in ("horizon", "population", "elites", "iterations", "std_min")},
        "episodes_requested": n_eps,
        "test_split_used": False,
        "latent_only_cost": (
            "mean squared distance from the zero-action latent rollout; "
            "minimized for T2-keep and maximized otherwise; no query head"
            if meta["kind"] == "latent_only" else None
        ),
        "summary": summary,
        "outcomes": outcomes,
    }
    out = run_dir / "planning.json"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        f"{meta['kind']}: success {summary['success']}/{summary['episodes']} "
        f"rate={summary['success_rate']:.3f} by_task={summary['by_task']}"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
