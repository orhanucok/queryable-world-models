# QWM Data Schema — v1.0.0 (FROZEN)

This document and `qwm/data/schema.py` are the **single source of truth** for
the data schema. The code constants are enforced by
`tests/test_query_schema.py`. **Nothing in this file may be changed
silently** (spec sections 0, 6, 13, 15). Any necessary change must be
documented prominently here and in the code, and both primary models (QWM
and latent-only) must be retrained and re-evaluated.

The scientific core protected by this schema:

```
query_state = [grip_xyz, obj_xyz, upright, contact, on_table, ...]
```

> Dreamer/JEPA/Genie dream. QWM answers.

---

## 1. Coordinate frame (WORLD frame)

- Right-handed, **meters**, origin at floor level below the table center,
  `+z` up, `+x`/`+y` along the table sides.
- Table: centered at `x=0, y=0`; footprint `|x| <= 0.45`, `|y| <= 0.45`;
  top surface at `z = 0.75`.
- Floor at `z = 0`.
- The camera is **fixed** (external, never moves; parameters in
  `configs/env.yaml`, not part of the frozen schema).

## 2. Frozen sizes

| Constant | Value | Meaning |
|---|---|---|
| `NMAX` | 20 | padded object slots (never changed silently) |
| `ACTION_DIM` | 4 | `[dx, dy, dz, gripper]` |
| `QUERY_DIM` | 123 | flattened query-state prediction vector |
| `EPISODE_LENGTH` | 50 | fixed-length episodes; `done` only at the last step |
| `OBS_CHANNELS` | 3 | RGB |
| `DEFAULT_RES` / `FALLBACK_RES` | 84 / 64 | observation H=W (config chooses one) |

## 3. Action space (FROZEN)

`action: float32[4] = [dx, dy, dz, gripper]`, every component clipped to
`[-1, 1]`:

- `dx, dy, dz`: end-effector delta in **meters** = component x 0.05
  (`ACTION_XYZ_SCALE`), applied to the **current actual** wrist position
  (feedback; no windup). The resulting EE target is clamped to the EE
  workspace: `|x|, |y| <= 0.40`, `z in [0.86, 1.06]`.
- `gripper`: linear map to finger half-opening
  `q_f = 0.012 + (g + 1)/2 * 0.033 in [0.012, 0.045]`, i.e. opening
  `2*q_f in [0.024, 0.090]` m. `+1` = fully open, `-1` = closed.

The action space is **identical** between data collection and training.
There is no joint-space action option.

## 4. query_state layout (the model's prediction target)

Flattened `query_state` vector, `float32[123]`, **slot-major** flattening
inside each object block (slot 0: x, y, z; slot 1: x, y, z; ...):

| Indices | Field | Size | Units / encoding |
|---|---|---|---|
| `[0, 3)` | `grip_xyz` | 3 | normalized position (see §6) |
| `[3, 63)` | `obj_xyz` | 20x3 | normalized positions, padded (see §5) |
| `[63, 83)` | `obj_upright` | 20 | `{0, 1}`, padded |
| `[83, 103)` | `obj_contact` | 20 | `{0, 1}`, padded |
| `[103, 123)` | `obj_on_table` | 20 | `{0, 1}`, padded |

`object_valid_mask` (bool[20]) and `target_object_index` (int) are
**context**, not prediction targets: objects never despawn in v0, so the
mask is constant within an episode. The mask gates every object-head loss
and every per-object metric.

## 5. Padding and mask conventions (FROZEN)

- Real objects occupy slots `0..n-1` (`n = env.num_objects`, `1 <= n <= 20`).
- Padded slots (`i >= n`): `obj_xyz = POS_ORIGIN = (0, 0, 0.75)` in **meters**
  (exactly `0` in normalized space), and `0.0` for all binary flags.
- `object_valid_mask[i] = (i < n)`.
- Every consumer (loss, metric, planner) MUST gate on the mask; tests
  enforce that padded slots never contribute to losses or metrics.

## 6. Normalization convention (FROZEN)

Positions are normalized elementwise (never clipped; off-table objects may
legitimately exceed the range):

```
p_norm = (p - POS_ORIGIN) / POS_SCALE
POS_ORIGIN = (0.0, 0.0, 0.75)      # table-top center
POS_SCALE  = (0.45, 0.45, 0.30)    # xy = table half-extent, z span
denormalize: p = p_norm * POS_SCALE + POS_ORIGIN
```

**Storage keeps RAW METERS** (unnormalized, reversible); normalization is
applied by the dataset/model boundary. Binary flags are stored and used as
`{0, 1}` with no normalization.

## 7. Binary label definitions (FROZEN)

- **`obj_upright`**: 1 iff the angle between the object's local `+z` axis
  and world `+z` is strictly `< 25 deg` (`UPRIGHT_TILT_DEG`). Boxes spawn
  yaw-randomized but upright; cylinders spawn standing or lying.
- **`obj_contact`**: 1 iff, **at the end of an environment step**, PyBullet
  reports >= 1 contact manifold between the object and **any robot link**
  (paddle bar or either finger). Contact with the table or other objects
  does **not** count.
- **`obj_on_table`**: 1 iff the object center satisfies
  `|x| <= 0.45 and |y| <= 0.45` and `0.72 <= z <= 1.05`
  (`ON_TABLE_Z_MIN = TABLE_TOP_Z - 0.03`, `ON_TABLE_Z_MAX = TABLE_TOP_Z + 0.30`):
  inside the table's XY footprint and within the volume above the surface
  (not fallen to the floor). An object carried above the table still counts
  (documented convention).

## 8. Tasks and task_goal (FROZEN task suite)

`task_id` (int): `T1 = 1` push-to-xy, `T2 = 2` knock-or-keep-upright,
`T3 = 3` off-table.

`task_goal: float32[4] = [target_object_index, goal_x, goal_y, t2_mode]`:

| Slot | T1 | T2 | T3 |
|---|---|---|---|
| `[0]` target slot | target object index (int as float) | same | same |
| `[1]` goal_x | goal point x (m, world) | `-1.0` unused | `-1.0` unused |
| `[2]` goal_y | goal point y (m, world) | `-1.0` unused | `-1.0` unused |
| `[3]` t2_mode | `-1.0` unused | `0` knock, `1` keep | `-1.0` unused |

**Success rules (evaluated from simulator state):**

- **T1** (sticky): exists `t`: `||obj_xy(target, t) - goal_xy|| <= 0.06`
  AND `on_table(target, t)`.
- **T2 knock** (mode 0, sticky): exists `t`: NOT `upright(target, t)`.
- **T2 keep** (mode 1, episode-final): (exists `t`:
  `contact(robot, target, t)`) AND (for all `t`: `upright(target, t)`).
  A tip after first contact **revokes** success.
- **T3** (sticky): exists `t`: NOT `on_table(target, t)`.

**Rewards (simple, stored as data fields; NOT a model signal):**
T1: `2.0` on success else `-dist`; T2 knock: `1.0` on new tip else `0`;
T2 keep: `-1.0` if tipped else `0.1` on new contact else `0`; T3: `1.0` on
new off-table else `0`.

## 9. Episode storage format

One compressed `.npz` file per episode, `data/episodes/ep_<id:06d>.npz`,
each array stacked over time `[T=50, ...]`:

| Field | dtype | Shape | Notes |
|---|---|---|---|
| `obs_rgb` | uint8 | `[T, 3, H, W]` | H=W from collection config (84 default) |
| `action` | float32 | `[T, 4]` | clipped to `[-1,1]` |
| `reward` | float32 | `[T]` | §8 |
| `done` | bool | `[T]` | True only at the last step |
| `task_id` | int64 | `[T]` | repeated |
| `task_goal` | float32 | `[T, 4]` | §8 |
| `grip_xyz` | float32 | `[T, 3]` | meters |
| `obj_xyz` | float32 | `[T, 20, 3]` | meters, padded (§5) |
| `obj_upright` | float32 | `[T, 20]` | {0,1}, padded |
| `obj_contact` | float32 | `[T, 20]` | {0,1}, padded |
| `obj_on_table` | float32 | `[T, 20]` | {0,1}, padded |
| `object_valid_mask` | bool | `[T, 20]` | §5 |
| `target_object_index` | int64 | `[T]` | repeated |
| `episode_id` | int64 | `[T]` | stable id |
| `timestep` | int64 | `[T]` | `0..T-1` |
| `meta` | str (JSON) | scalar | see below |

Episode `meta` (JSON scalar in the npz; required keys):
`collection_source` (`scripted` | `random` | `teleop` | `mixed:...`),
`seed`, `n_objects`, `task_id`, `t2_mode`, `success`, `res`.

Transition `t` semantics: `(obs_rgb[t], action[t]) -> query_state at t+1`;
i.e. `grip_xyz/obj_*/...` at index `t` are the state **after** action `t`
(index 0 is the state after the first action from reset). The reset state
(before any action) is not stored; rollouts start at `t=0` by encoding
`obs_rgb[0]` and predicting from `action[0]` onward.

## 10. Dataset splits (FROZEN policy)

- Split **by episode**, never by transition: train 80% / val 10% / test 10%.
- Deterministic seeded assignment (`data/splits.json` stores the manifest:
  seed, episode-id lists, counts). No episode ever spans splits.
- Model selection uses **validation** data only; the test split is touched
  once, by the frozen final evaluation.

## 11. Spawn policy (env-side, documented)

Pairwise object-center distance at spawn `>= gap + r_i + r_j` (bounding
radii; `gap = clip(0.20/sqrt(n), 0.005, 0.10)`). Per object: 150 rejection
samples, then a deterministic grid/edge-ladder fallback (shuffled, small
jitter); the whole spawn may restart (fresh draws) with `gap` halved
(floor `0.005`, up to 6 restarts), then fails loud. Object slot 0 is always
upright/tippable (T2) and is forced into the near-edge zone (`|x|` or
`|y| >= 0.27`) for T3 episodes. Object types: boxes and cylinders only
(spheres excluded so `upright` is always meaningful).

## 12. Implementation notes that protect determinism

- Every PyBullet call in `qwm/envs/` passes an explicit `physicsClientId`;
  without it calls leak across clients and cross-instance runs diverge
  (verified by tests).
- Determinism contract: same `(seed, task, num_objects, config)` => same
  episode, bit-identical RGB included (same machine/build; cross-machine
  bit-exactness is not claimed).
- Ground-truth query state from the simulator is explicitly allowed in v0
  for dataset labels and evaluation; **models only ever receive RGB +
  actions + task conditioning as inputs.**

## 13. Change policy

If a change to any frozen value becomes necessary: (1) stop, (2) document
the change and its reason here AND in `qwm/data/schema.py` with a new
schema version, (3) re-run BOTH primary models under identical budgets,
(4) re-run the frozen evaluation protocol, (5) update the report. Never
change indices, meanings, padding, masks, normalization, ordering,
thresholds, or the action space silently.
