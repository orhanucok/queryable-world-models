"""PyBullet tabletop environment (v0) for QWM.

World layout (frozen constants in qwm/data/schema.py):
  - floor at z = 0, table top at z = 0.75 (0.9 x 0.9 m surface);
  - gantry-style robot (qwm/envs/assets/gantry_gripper.urdf): 3 prismatic
    axes => the paddle/wrist center ("grip_xyz") is directly position
    controlled; two mirrored fingers form the gripper;
  - fixed external camera rendering RGB [3, H, W] uint8 (default 84x84).

Action space (FROZEN): float32[4] = [dx, dy, dz, gripper], each clipped to
[-1, 1]; xyz deltas are ACTION_XYZ_SCALE meters applied to the CURRENT
ACTUAL wrist position (feedback; no windup); the EE target is clamped to
the frozen workspace; gripper maps linearly to finger half-opening.

Episodes are FIXED-LENGTH (EPISODE_LENGTH = 50): done is True only at the
final step; success is derived from simulator state (see qwm/envs/tasks.py)
and is reported in info, never used for early termination.

Determinism: reset(seed, ...) fully rebuilds the simulation (floor, table,
robot, objects) with all stochastic draws from the seeded episode RNG, so
the same (seed, task, num_objects) yields the same initial state. Physics
is PyBullet DIRECT mode with a fixed 1/240 s step.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from qwm.data import schema
from qwm.envs import tasks as task_lib
from qwm.envs.query_state import extract_query, is_on_table
from qwm.utils.config import get as cfg_get
from qwm.utils.logging import get_logger
from qwm.utils.seed import make_rng

logger = get_logger(__name__)

ASSETS_DIR = Path(__file__).resolve().parent / "assets"

# URDF-consistent joint<->world mapping (documented in the URDF header):
#   wrist_x = -0.55 + q_x ; wrist_y = -0.55 + q_y ; wrist_z = 1.50 + q_z
_QX_OFF = 0.55
_QY_OFF = 0.55
_QZ_OFF = 1.50

# Finger joint limits (must match the URDF).
_FINGER_LO, _FINGER_HI = 0.012, 0.045


class TabletopEnv:
    """Tabletop push/manipulation environment with query-state ground truth."""

    def __init__(self, config: dict, render: Optional[str] = None):
        import pybullet as p  # lazy, fail loud at construction

        self._p = p
        cfg = config.get("env", config)
        self.cfg = cfg

        # --- non-frozen, configurable knobs (frozen constants live in schema) ---
        self.res = int(cfg_get(cfg, "obs.res", schema.DEFAULT_RES))
        if self.res not in (schema.DEFAULT_RES, schema.FALLBACK_RES):
            raise ValueError(f"obs.res must be {schema.DEFAULT_RES} or {schema.FALLBACK_RES}, got {self.res}")
        self.substeps = int(cfg_get(cfg, "physics.substeps_per_action", 20))
        self.solver_iterations = int(cfg_get(cfg, "physics.solver_iterations", 50))
        self.max_velocity = float(cfg_get(cfg, "robot.max_velocity", 1.5))
        self.force_xy = float(cfg_get(cfg, "robot.force_xy", 500.0))
        self.force_z = float(cfg_get(cfg, "robot.force_z", 300.0))
        self.force_finger = float(cfg_get(cfg, "robot.force_finger", 40.0))
        self.default_num_objects = int(cfg_get(cfg, "objects.num_objects", 1))
        self.spawn_radius = float(cfg_get(cfg, "objects.spawn_radius_xy", 0.33))
        cam = cfg_get(cfg, "camera", {}) or {}
        self._cam_distance = float(cam.get("distance", 1.35))
        self._cam_yaw = float(cam.get("yaw", 35.0))
        self._cam_pitch = float(cam.get("pitch", -35.0))
        self._cam_target = [float(v) for v in cam.get("target", [0.0, 0.0, 0.72])]
        self._cam_fov = float(cam.get("fov", 50.0))

        render_mode = render or str(cfg_get(cfg, "render", "direct"))
        if render_mode == "direct":
            self._cid = p.connect(p.DIRECT)
        elif render_mode == "gui":
            self._cid = p.connect(p.GUI)
        else:
            raise ValueError(f"render must be 'direct' or 'gui', got {render_mode!r}")
        self.render_mode = render_mode

        # Fixed camera matrices (computed once; camera never moves).
        self._view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self._cam_target,
            distance=self._cam_distance,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0.0,
            upAxisIndex=2,
        )
        self._proj = p.computeProjectionMatrixFOV(
            fov=self._cam_fov, aspect=1.0, nearVal=0.02, farVal=5.0
        )

        # Episode state (populated by reset).
        self._rng: Optional[np.random.Generator] = None
        self._t = 0
        self.robot_id: Optional[int] = None
        self.object_ids: List[int] = []
        self.object_metas: List[Dict] = []
        self.task: Dict = {}
        self.task_goal: np.ndarray = np.zeros(schema.TASK_GOAL_DIM, dtype=np.float32)
        self.target_object_index: int = 0
        self._ever_contact = False
        self._ever_tipped = False
        self._achieved = False
        self.seed_value: Optional[int] = None

    # ------------------------------------------------------------ helpers --
    def _set_physics(self) -> None:
        p = self._p
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self._cid)
        try:
            p.setPhysicsEngineParameter(
                fixedTimeStep=1.0 / 240.0,
                numSolverIterations=self.solver_iterations,
                deterministicOverlappingPairs=1,
                physicsClientId=self._cid,
            )
        except TypeError:  # older pybullet without the determinism flag
            logger.warning("deterministicOverlappingPairs unavailable; continuing")
            p.setPhysicsEngineParameter(
                fixedTimeStep=1.0 / 240.0,
                numSolverIterations=self.solver_iterations,
                physicsClientId=self._cid,
            )

    def _build_static(self) -> None:
        p = self._p
        # Floor (large box, top at z = 0).
        floor_half = [1.5, 1.5, 0.005]
        cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=floor_half, physicsClientId=self._cid)
        vid = p.createVisualShape(
            p.GEOM_BOX, halfExtents=floor_half, rgbaColor=[0.55, 0.56, 0.58, 1.0],
            physicsClientId=self._cid,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=cid,
            baseVisualShapeIndex=vid,
            basePosition=[0.0, 0.0, -0.005],
            physicsClientId=self._cid,
        )
        # Table (static box, top at TABLE_TOP_Z).
        table_half = [schema.TABLE_HALF_XY, schema.TABLE_HALF_XY, 0.0375]
        cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half, physicsClientId=self._cid)
        vid = p.createVisualShape(
            p.GEOM_BOX, halfExtents=table_half, rgbaColor=[0.42, 0.35, 0.28, 1.0],
            physicsClientId=self._cid,
        )
        p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=cid,
            baseVisualShapeIndex=vid,
            basePosition=[0.0, 0.0, schema.TABLE_TOP_Z - 0.0375],
            physicsClientId=self._cid,
        )
        # Robot.
        urdf_path = ASSETS_DIR / "gantry_gripper.urdf"
        if not urdf_path.is_file():
            raise FileNotFoundError(f"robot URDF missing: {urdf_path}")
        self.robot_id = p.loadURDF(
            str(urdf_path), basePosition=[0.0, 0.0, 0.0], physicsClientId=self._cid
        )
        self._joint_index = {}
        self.wrist_link_index = None
        # pybullet 3.2.7 getJointInfo tuple: [1]=jointName, [12]=linkName
        for j in range(p.getNumJoints(self.robot_id, physicsClientId=self._cid)):
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self._cid)
            jname = info[1].decode("utf-8")
            lname = info[12].decode("utf-8")
            self._joint_index[jname] = j
            if lname == "wrist":
                self.wrist_link_index = j
        needed = ("x_axis", "y_axis", "z_axis", "finger_left", "finger_right")
        missing = [n for n in needed if n not in self._joint_index]
        if missing:
            raise RuntimeError(f"robot URDF missing joints: {missing}")
        if self.wrist_link_index is None:
            raise RuntimeError("robot URDF missing link 'wrist'")

    def _reset_robot_home(self) -> None:
        p = self._p
        home = [float(v) for v in cfg_get(self.cfg, "robot.home_xyz", [0.0, 0.0, 0.92])]
        q_home = {
            "x_axis": home[0] + _QX_OFF,
            "y_axis": home[1] + _QY_OFF,
            "z_axis": home[2] - _QZ_OFF,
            "finger_left": 0.03,
            "finger_right": 0.03,
        }
        for name, q in q_home.items():
            p.resetJointState(
                self.robot_id, self._joint_index[name], q, 0.0, physicsClientId=self._cid
            )
            p.setJointMotorControl2(
                self.robot_id, self._joint_index[name], p.POSITION_CONTROL,
                targetPosition=q,
                force=self.force_finger if "finger" in name else self.force_xy,
                physicsClientId=self._cid,
            )

    # ------------------------------------------------------------- spawn ---
    # Documented spawn policy: pairwise center distance must be >= gap +
    # r_i + r_j (bounding radii). Per object: 150 rejection samples, then a
    # deterministic grid/edge-ladder fallback (shuffled, small jitter), then
    # the whole spawn restarts (fresh draws) with gap halved, floor 0.005,
    # up to 6 restarts. Object 0 is always upright/tippable (T2) and is
    # forced into the near-edge zone for T3 (task-aware placement).
    def _sample_object(self, rng: np.random.Generator, force_upright: bool) -> Dict:
        """Sample one object's geometry; returns meta WITHOUT position."""
        p = self._p
        if force_upright or rng.random() < 0.6:
            kind = "box"
        else:
            kind = "cylinder"
        if kind == "box":
            hx = float(rng.uniform(0.020, 0.045))
            hy = float(hx * rng.uniform(0.7, 1.3))
            hz = float(rng.uniform(0.025, 0.060))
            volume = 8.0 * hx * hy * hz
            meta = {
                "kind": "box", "hx": hx, "hy": hy, "hz": hz,
                "radius_xy": math.hypot(hx, hy), "volume": volume,
                "upright_at_spawn": True,
                "spawn_pose": "standing",
            }
        else:
            r = float(rng.uniform(0.020, 0.035))
            if force_upright or rng.random() < 0.7:
                h = float(rng.uniform(0.05, 0.13))
                meta = {
                    "kind": "cylinder", "r": r, "h": h,
                    "radius_xy": r, "volume": math.pi * r * r * h,
                    "upright_at_spawn": True,
                    "spawn_pose": "standing",
                }
            else:
                h = float(rng.uniform(0.06, 0.12))
                meta = {
                    "kind": "cylinder", "r": r, "h": h,
                    "radius_xy": r, "volume": math.pi * r * r * h,
                    "upright_at_spawn": False,
                    "spawn_pose": "lying",
                }
        meta["mass"] = max(0.03, meta["volume"] * 600.0)
        meta["lateral_friction"] = float(rng.uniform(0.55, 0.9))
        meta["rgba"] = [float(rng.uniform(0.35, 0.9)) for _ in range(3)] + [1.0]
        return meta

    def _spawn_objects(self, n: int, task_id: int) -> List[Dict]:
        """Place + instantiate n objects with the documented spawn policy."""
        gap0 = float(np.clip(0.20 / math.sqrt(max(n, 1)), 0.005, 0.10))
        for restart in range(6):
            gap = max(0.005, gap0 / (2.0 ** restart))
            placed = self._try_place(n, task_id, gap)
            if placed is not None:
                self._instantiate(placed)
                self.object_ids = [m["body_id"] for m in placed]
                self.object_metas = placed
                return placed
            logger.debug("spawn restart %d/6 failed (gap=%.3f)", restart + 1, gap)
        raise RuntimeError(
            f"could not place {n} objects after 6 restarts (documented spawn policy)"
        )

    def _try_place(self, n: int, task_id: int, gap: float) -> Optional[List[Dict]]:
        rng = self._rng
        edge_zone = schema.TABLE_HALF_XY - schema.T3_EDGE_MARGIN  # 0.27
        # Deterministic fallback ladders.
        cols = max(1, int(math.ceil(math.sqrt(n))))
        cell = (2.0 * self.spawn_radius) / cols
        offs = np.linspace(
            -self.spawn_radius + cell / 2.0, self.spawn_radius - cell / 2.0, cols
        )
        grid_centers = [(float(x), float(y)) for x in offs for y in offs]
        edge_ladder: List = []
        for v in np.linspace(-0.30, 0.30, 7):
            for c in (edge_zone + 0.02, 0.33):
                edge_ladder += [(c, float(v)), (-c, float(v)),
                                (float(v), c), (float(v), -c)]

        placed: List[Dict] = []
        for i in range(n):
            meta = self._sample_object(rng, force_upright=(i == 0))
            forced_edge = (i == 0 and task_id == schema.TASK_OFF_TABLE)
            xy = self._sample_position(
                rng, placed, gap, meta["radius_xy"], forced_edge,
                edge_zone, grid_centers, edge_ladder,
            )
            if xy is None:
                return None
            meta["xy"] = xy
            meta["near_edge"] = bool(max(abs(xy[0]), abs(xy[1])) >= edge_zone)
            placed.append(meta)
        return placed

    def _sample_position(
        self,
        rng: np.random.Generator,
        placed: List[Dict],
        gap: float,
        r_new: float,
        forced_edge: bool,
        edge_zone: float,
        grid_centers: List,
        edge_ladder: List,
    ) -> Optional[np.ndarray]:
        def fits(xy: np.ndarray) -> bool:
            return all(
                float(np.linalg.norm(xy - q["xy"])) >= gap + q["radius_xy"] + r_new
                for q in placed
            )

        # 1) rejection sampling (edge band for the forced T3 object).
        for _ in range(150):
            if forced_edge:
                side = int(rng.integers(0, 2))
                coord = float(rng.uniform(edge_zone, 0.34)) * float(rng.choice([-1.0, 1.0]))
                other = float(rng.uniform(-self.spawn_radius, self.spawn_radius))
                xy = np.array([coord, other], dtype=np.float64) if side == 0 \
                    else np.array([other, coord], dtype=np.float64)
            else:
                xy = rng.uniform(-self.spawn_radius, self.spawn_radius, size=2)
            if fits(xy):
                return xy
        # 2) deterministic ladder fallback (shuffled order, small jitter).
        ladder = edge_ladder if forced_edge else grid_centers
        for k in rng.permutation(len(ladder)):
            gx, gy = ladder[int(k)]
            xy = np.array(
                [gx + rng.uniform(-0.01, 0.01), gy + rng.uniform(-0.01, 0.01)],
                dtype=np.float64,
            )
            if fits(xy):
                return xy
        return None

    def _instantiate(self, placed: List[Dict]) -> None:
        """Create PyBullet bodies for placed object metas (draws yaw from rng)."""
        p = self._p
        rng = self._rng
        for meta in placed:
            xy = meta["xy"]
            yaw = float(rng.uniform(0.0, 2.0 * math.pi))
            if meta["kind"] == "box":
                z = schema.TABLE_TOP_Z + meta["hz"]
                euler = (0.0, 0.0, yaw)
                half = [meta["hx"], meta["hy"], meta["hz"]]
                cid = p.createCollisionShape(
                    p.GEOM_BOX, halfExtents=half, physicsClientId=self._cid
                )
                vid = p.createVisualShape(
                    p.GEOM_BOX, halfExtents=half, rgbaColor=meta["rgba"],
                    physicsClientId=self._cid,
                )
            else:
                if meta["spawn_pose"] == "standing":
                    z = schema.TABLE_TOP_Z + meta["h"] / 2.0
                    euler = (0.0, 0.0, yaw)
                else:
                    z = schema.TABLE_TOP_Z + meta["r"]
                    euler = (math.pi / 2.0, 0.0, yaw)
                # pybullet quirk: collision uses 'height', visual uses 'length'
                cid = p.createCollisionShape(
                    p.GEOM_CYLINDER, radius=meta["r"], height=meta["h"],
                    physicsClientId=self._cid,
                )
                vid = p.createVisualShape(
                    p.GEOM_CYLINDER, radius=meta["r"], length=meta["h"],
                    rgbaColor=meta["rgba"], physicsClientId=self._cid,
                )
            quat = p.getQuaternionFromEuler(euler)
            oid = p.createMultiBody(
                baseMass=meta["mass"],
                baseCollisionShapeIndex=cid,
                baseVisualShapeIndex=vid,
                basePosition=[float(xy[0]), float(xy[1]), float(z)],
                baseOrientation=[float(v) for v in quat],
                physicsClientId=self._cid,
            )
            p.changeDynamics(
                oid, -1, lateralFriction=meta["lateral_friction"],
                physicsClientId=self._cid,
            )
            meta["body_id"] = oid
            meta["z"] = z
            meta["on_table_at_spawn"] = bool(is_on_table([xy[0], xy[1], z]))

    # ------------------------------------------------------------- reset ---
    def reset(
        self,
        seed: Optional[int] = None,
        task: Optional[Dict] = None,
        num_objects: Optional[int] = None,
    ) -> Dict:
        """Reset the environment.

        seed: required on the FIRST reset; reused if omitted later.
        task: {"task_id": int, "t2_mode": int|None, "goal_xy": (x,y)|None};
              missing parts are sampled with the episode RNG.
        num_objects: override the configured object count (1..NMAX).
        """
        p = self._p
        if seed is not None:
            self.seed_value = int(seed)
            self._rng = make_rng(int(seed))
        elif self._rng is None:
            raise RuntimeError("reset() requires an explicit seed on the first call")
        rng = self._rng

        n = int(num_objects if num_objects is not None else self.default_num_objects)
        if not (1 <= n <= schema.NMAX):
            raise ValueError(f"num_objects must be in [1, {schema.NMAX}], got {n}")

        # Task (sample or adopt the provided spec).
        if task is None:
            spec = task_lib.sample_task(rng)
        else:
            spec = {
                "task_id": int(task["task_id"]),
                "t2_mode": task.get("t2_mode"),
                "goal_xy": task.get("goal_xy"),
            }
            if spec["task_id"] not in schema.TASK_IDS:
                raise ValueError(f"unknown task_id {spec['task_id']}")
            if spec["task_id"] == schema.TASK_PUSH_TO_XY and spec["goal_xy"] is None:
                gx, gy = rng.uniform(-schema.T1_GOAL_XY_LIMIT, schema.T1_GOAL_XY_LIMIT, size=2)
                spec["goal_xy"] = (float(gx), float(gy))
            if spec["task_id"] == schema.TASK_KNOCK_OR_KEEP and spec["t2_mode"] is None:
                spec["t2_mode"] = int(rng.integers(0, 2))
        self.task = spec

        # Rebuild the world deterministically.
        p.resetSimulation(physicsClientId=self._cid)
        self._set_physics()
        self._build_static()
        self._spawn_objects(n, spec["task_id"])
        self._reset_robot_home()
        self.target_object_index = task_lib.choose_target(
            rng, spec["task_id"], self.object_metas
        )
        self.task_goal = task_lib.task_goal_vector(spec, self.target_object_index)

        # Sticky success state.
        self._t = 0
        self._ever_contact = False
        self._ever_tipped = False
        self._achieved = False

        return self._observe()

    # -------------------------------------------------------------- step --
    def _ee_position(self) -> np.ndarray:
        state = self._p.getLinkState(
            self.robot_id, self.wrist_link_index, physicsClientId=self._cid
        )
        return np.asarray(state[0], dtype=np.float64)

    def _apply_action(self, action: np.ndarray) -> None:
        p = self._p
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(4),
                    schema.ACTION_LOW, schema.ACTION_HIGH)
        wrist = self._ee_position()
        target = wrist + a[:3] * schema.ACTION_XYZ_SCALE
        target[0] = float(np.clip(target[0], -schema.EE_XY_LIMIT, schema.EE_XY_LIMIT))
        target[1] = float(np.clip(target[1], -schema.EE_XY_LIMIT, schema.EE_XY_LIMIT))
        target[2] = float(np.clip(target[2], schema.EE_Z_MIN, schema.EE_Z_MAX))
        q = {
            "x_axis": target[0] + _QX_OFF,
            "y_axis": target[1] + _QY_OFF,
            "z_axis": target[2] - _QZ_OFF,
        }
        for name, qi in q.items():
            force = self.force_z if name == "z_axis" else self.force_xy
            p.setJointMotorControl2(
                self.robot_id, self._joint_index[name], p.POSITION_CONTROL,
                targetPosition=float(qi), force=force, maxVelocity=self.max_velocity,
                physicsClientId=self._cid,
            )
        q_f = schema.GRIPPER_CLOSED_Q + (a[3] + 1.0) / 2.0 * (
            schema.GRIPPER_OPEN_Q - schema.GRIPPER_CLOSED_Q
        )
        q_f = float(np.clip(q_f, _FINGER_LO, _FINGER_HI))
        for name in ("finger_left", "finger_right"):
            p.setJointMotorControl2(
                self.robot_id, self._joint_index[name], p.POSITION_CONTROL,
                targetPosition=q_f, force=self.force_finger, maxVelocity=0.5,
                physicsClientId=self._cid,
            )

    def step(self, action) -> tuple:
        if self.robot_id is None:
            raise RuntimeError("call reset() before step()")
        self._apply_action(np.asarray(action, dtype=np.float64))
        for _ in range(self.substeps):
            self._p.stepSimulation(physicsClientId=self._cid)
        self._t += 1

        obs = self._observe()
        query = {k: obs[k] for k in schema.QUERY_KEYS}
        ti = self.target_object_index
        spec = self.task

        # Per-step condition + sticky bookkeeping (frozen rules; see tasks.py).
        per_step = task_lib.evaluate_success(
            spec["task_id"], ti, spec["goal_xy"], spec["t2_mode"], query
        )
        upright_now = float(query["obj_upright"][ti]) > 0.5
        contact_now = float(query["obj_contact"][ti]) > 0.5
        self._ever_contact = self._ever_contact or contact_now
        self._ever_tipped = self._ever_tipped or (not upright_now)

        if spec["task_id"] == schema.TASK_KNOCK_OR_KEEP and spec["t2_mode"] == schema.T2_MODE_KEEP:
            # keep-upright: episode-final rule (ever contact AND never tipped);
            # a later tip revokes success (frozen formula in tasks.py docstring).
            self._achieved = self._ever_contact and not self._ever_tipped
            achieved_now = per_step
        else:
            achieved_now = per_step and not self._achieved
            self._achieved = self._achieved or per_step

        reward = float(task_lib.task_reward(
            spec["task_id"], ti, spec["goal_xy"], spec["t2_mode"], query, achieved_now
        ))
        done = self._t >= schema.EPISODE_LENGTH
        info = {
            "success": bool(self._achieved),
            "newly_achieved": bool(achieved_now and self._achieved),
            "task_id": spec["task_id"],
            "timestep": self._t,
            "done_reason": "episode_length" if done else "",
        }
        return obs, reward, bool(done), info

    # --------------------------------------------------------- observation --
    def render_rgb(self) -> np.ndarray:
        p = self._p
        w, h, rgb, _, _ = p.getCameraImage(
            self.res, self.res,
            viewMatrix=self._view,
            projectionMatrix=self._proj,
            renderer=p.ER_TINY_RENDERER,
            physicsClientId=self._cid,
        )
        arr = np.reshape(np.asarray(rgb, dtype=np.uint8), (int(h), int(w), 4))
        return np.ascontiguousarray(arr[:, :, :3].transpose(2, 0, 1))

    def _observe(self) -> Dict:
        query = extract_query(
            self._p, self.robot_id, self.wrist_link_index, self.object_ids,
            client_id=self._cid,
        )
        obs = {"rgb": self.render_rgb()}
        obs.update(query)
        obs["task_id"] = int(self.task["task_id"])
        obs["task_goal"] = np.asarray(self.task_goal, dtype=np.float32)
        obs["target_object_index"] = int(self.target_object_index)
        obs["timestep"] = int(self._t)
        return obs

    # ------------------------------------------------------------ extras --
    def state(self) -> Dict:
        """Compact UNPADDED ground-truth state (data-engine and eval use only).

        Models NEVER receive this; it backs scripted policies and metrics.
        """
        query = extract_query(
            self._p, self.robot_id, self.wrist_link_index, self.object_ids,
            client_id=self._cid,
        )
        n = len(self.object_ids)
        return {
            "grip_xyz": np.array(query["grip_xyz"], dtype=np.float32),
            "obj_xyz": np.array(query["obj_xyz"][:n], dtype=np.float32),
            "obj_upright": np.array(query["obj_upright"][:n], dtype=np.float32),
            "obj_contact": np.array(query["obj_contact"][:n], dtype=np.float32),
            "obj_on_table": np.array(query["obj_on_table"][:n], dtype=np.float32),
            "task_id": int(self.task["task_id"]),
            "t2_mode": self.task["t2_mode"],
            "goal_xy": self.task["goal_xy"],
            "target_object_index": int(self.target_object_index),
            "timestep": int(self._t),
        }

    @property
    def num_objects(self) -> int:
        return len(self.object_ids)

    @property
    def timestep(self) -> int:
        return self._t

    @property
    def episode_length(self) -> int:
        return schema.EPISODE_LENGTH

    def close(self) -> None:
        try:
            self._p.disconnect(self._cid)
        except Exception:
            pass

    def __enter__(self) -> "TabletopEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
