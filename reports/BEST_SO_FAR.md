# Best so far

Residual K=8 QWM on `data/oneobj5k`. Checkpoint step 5000. 973,243 parameters. `λ_pos = 10`. No planner.

Teacher-forced queryable residuals are task-tolerable at H=8; closed-loop compounding is not.

## Closed-loop horizon table

Validation windows of length 33. Latent-only has no query head.

| Model | H | grip L2 (m) | obj L2 (m) | contact F1 | upright acc | on-table acc |
|---|---:|---:|---:|---:|---:|---:|
| QWM | 1 | 0.0255 | 0.0407 | 0.0000 | 0.5444 | 0.6833 |
| QWM | 8 | 0.0622 | 0.1733 | 0.0000 | 0.8389 | 0.8333 |
| QWM | 16 | 0.1238 | 0.2447 | 0.0000 | 0.9111 | 0.9000 |
| QWM | 32 | 0.2749 | 0.3679 | 0.0000 | 0.9000 | 0.8000 |
| latent-only | 1, 8, 16, 32 | n/a | n/a | n/a | n/a | n/a |

## Teacher-forced vs closed-loop at H=8

Same validation windows.

| | obj L2 (m) |
|---|---:|
| Teacher-forced H=8 | 0.055 |
| Closed-loop H=8 | 0.173 |

The 0.10 m closed-loop gate is not met.
