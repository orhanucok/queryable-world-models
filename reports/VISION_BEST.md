# Vision best

Single-frame CNN only. Same conv stack, 84×84 unless noted. No GRU, no last state, no action, no ViT. `runs/qwm/best.pt` is still the step-5000 residual world model. No planner.

The best object error is **0.132 m**, on the original camera, with either 84×84 or 128×128 images. That is above 0.08 m. Vision stops here.

## Validation object and gripper error

| Run | grip L2 (m) | obj L2 (m) |
|---|---:|---:|
| oneobj20k, 84×84, original camera | 0.031 | 0.132 |
| oneobj20k, 128×128, original camera | 0.028 | 0.132 |
| oneobj5k_topdown, 84×84, closer top-down | 0.267 | 0.441 |
| Constant train-mean, top-down val | 0.353 | 0.642 |

The 0.132 m rows are `runs/frame_e2e_20k` (step 10800) and `runs/frame_e2e_20k_128` (step 17000). The top-down row is `runs/frame_e2e_topdown` (step 4000). Test splits were not read.

## Closer camera

`data/oneobj5k_topdown`: 100 scripted episodes, 5,000 transitions, 1 object, Nmax=20 pad, schema unchanged, same T1/T2/T3 tasks. Query check 100/100. Split seed 0 is 80/10/10. Scripted success was T1 2/28, T2 13/28, T3 32/44.

Camera only, in `configs/env_topdown.yaml`: distance 0.70, pitch −78°, yaw 0, fov 50, target (0, 0, 0.75). The default camera in `configs/env.yaml` is unchanged (distance 1.35, pitch −35°, yaw 35). On a shared spawn the object grew from about 10×13 px to about 15×17 px. Validation object error rose from 0.132 m to 0.441 m.

## Stop

84×84 and 128×128 match, so resolution is not the limit. A closer top-down camera does not bring validation object L2 under 0.08 m. Vision is halted.
