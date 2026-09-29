# Status

Training is stopped. Vision is stopped. The planner is stopped. No new run was started for this note.

`runs/qwm/best.pt` stays the residual K=8 checkpoint at step 5000.

## Claim

Unchanged: teacher-forced queryable residuals are task-tolerable at H=8; closed-loop compounding is not.

## BEST_SO_FAR

Residual K=8 QWM on `data/oneobj5k`. 973,243 parameters. `λ_pos = 10`.

| | obj L2 |
|---|---:|
| Teacher-forced H=8 | 0.055 m |
| Closed-loop H=8 | 0.173 m |

## VISION_BEST

Original camera, `data/oneobj20k`, single-frame CNN. Both resolutions land on the same object error.

| | grip L2 | obj L2 |
|---|---:|---:|
| 84×84 | 0.031 m | 0.132 m |
| 128×128 | 0.028 m | 0.132 m |
| Closer top-down (discarded) | 0.267 m | 0.441 m |

VISION_BEST stays **0.132 m** object L2 on the original 84×84 oneobj20k images. The closer top-down collect is discarded.

The gripper is readable from pixels, about 0.03 m. The object is not task-tolerable from pixels.

## Gate

No CEM was run. The gate remains closed-loop H=8 object L2 ≤ 0.10 m. Closed-loop H=8 is 0.173 m, so the gate is still closed.

## Next

The next allowed move is later, and only one of these:

- a collect where the cube is truly at least 40 px in frame, or
- freeze the claim as teacher-forced residual QWM.

No new architecture.
