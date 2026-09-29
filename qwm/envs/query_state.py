"""Ground-truth query-state extraction from the PyBullet simulator.

Every physical quantity in query_state is extracted HERE, using the frozen
constants and definitions in qwm/data/schema.py:

  grip_xyz     wrist (paddle center) position, WORLD frame, meters
  obj_xyz[i]   object i center-of-mass position, meters
  obj_upright   tilt(local +z vs world +z) < 25 deg
  obj_contact   >= 1 contact manifold with any robot link at step end
  obj_on_table  center inside table footprint and within [0.72, 1.05] m

Ground-truth query_state from the simulator is explicitly allowed in v0
(spec section 4) for dataset labels and evaluation; models only ever
receive RGB + actions + task conditioning as INPUTS.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from qwm.data import schema

_UPRIGHT_TILT_RAD = math.radians(schema.UPRIGHT_TILT_DEG)


def tilt_angle(quaternion: np.ndarray) -> float:
    """Angle (radians) between the object's local +z axis and world +z."""
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    x, y, z, w = q  # PyBullet quaternion order: (x, y, z, w)
    # local +z axis in world coordinates for quaternion q:
    up = np.array([
        2.0 * (x * z + w * y),
        2.0 * (y * z - w * x),
        1.0 - 2.0 * (x * x + y * y),
    ])
    return float(math.acos(min(1.0, max(-1.0, up[2]))))


def is_upright(quaternion: np.ndarray) -> float:
    """1.0 if tilt < frozen 25-degree threshold else 0.0."""
    return 1.0 if tilt_angle(quaternion) < _UPRIGHT_TILT_RAD else 0.0


def is_on_table(xyz: np.ndarray) -> float:
    """1.0 if the object center is inside the table volume, else 0.0."""
    p = np.asarray(xyz, dtype=np.float64).reshape(3)
    if abs(p[0]) > schema.TABLE_HALF_XY or abs(p[1]) > schema.TABLE_HALF_XY:
        return 0.0
    if not (schema.ON_TABLE_Z_MIN <= p[2] <= schema.ON_TABLE_Z_MAX):
        return 0.0
    return 1.0


def extract_query(
    p,                     # pybullet module (passed to avoid a hard import)
    robot_id: int,
    wrist_link_index: int,
    object_ids: List[int],
    client_id: int = 0,
) -> Dict[str, np.ndarray]:
    """Extract the padded ground-truth query dict at the current sim state.

    Contact is evaluated on the CURRENT contact manifolds: callers must
    invoke this at the END of an environment step (frozen definition).
    client_id MUST be passed when more than one pybullet client exists
    (all reads/writes are scoped to this client; see tabletop.py).
    """
    n = len(object_ids)
    obj_xyz = np.zeros((n, 3), dtype=np.float32)
    obj_upright = np.zeros(n, dtype=np.float32)
    obj_contact = np.zeros(n, dtype=np.float32)
    obj_on_table = np.zeros(n, dtype=np.float32)
    for i, oid in enumerate(object_ids):
        pos, quat = p.getBasePositionAndOrientation(oid, physicsClientId=client_id)
        obj_xyz[i] = np.asarray(pos, dtype=np.float32)
        obj_upright[i] = is_upright(np.asarray(quat, dtype=np.float64))
        obj_on_table[i] = is_on_table(pos)
        contacts = p.getContactPoints(
            bodyA=robot_id, bodyB=oid, physicsClientId=client_id
        )
        obj_contact[i] = 1.0 if len(contacts) > 0 else 0.0

    wrist_state = p.getLinkState(
        robot_id, wrist_link_index, physicsClientId=client_id
    )
    grip_xyz = np.asarray(wrist_state[0], dtype=np.float32)  # worldLinkFramePosition

    return schema.build_query(
        grip_xyz=grip_xyz,
        obj_xyz=obj_xyz,
        obj_upright=obj_upright,
        obj_contact=obj_contact,
        obj_on_table=obj_on_table,
    )
