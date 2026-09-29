# Queryable World Models — measured report

Question: does queryable state beat latent-only for planning?

**Answer: no.**

On 50 matched simulator episodes, QWM closed-loop success was 1/50 (0.020) and latent-only was 1/50 (0.020). Beat means a strictly higher success rate under the same CEM budget and the same seeds.

## Planning

QWM minimized a cost on the predicted query state (target distance plus off-table penalty, upright probability, or on-table probability). Latent-only has no query head and was not given one. Its CEM cost is the mean squared distance between the candidate latent rollout and a zero-action rollout from the same image: minimized for T2 keep, maximized otherwise. That cost does not know the goal coordinates or the object pose.

CEM budget for both: horizon 8, population 64, elites 4, iterations 4. Parameters: QWM 973243, latent-only 957376.

| task | QWM | latent-only |
|---|---:|---:|
| T1 | 0/11 | 1/11 |
| T2 | 1/20 | 0/20 |
| T3 | 0/19 | 0/19 |

## Horizon metrics (validation only)

These numbers are the open-loop val table. The dataset test split was not read.

| model | H | pos err (m) | contact F1 | upright acc | on-table acc |
|---|---:|---:|---:|---:|---:|
| latent_only | 1 | n/a | n/a | n/a | n/a |
| latent_only | 8 | n/a | n/a | n/a | n/a |
| latent_only | 16 | n/a | n/a | n/a | n/a |
| latent_only | 32 | n/a | n/a | n/a | n/a |
| qwm | 1 | 0.2638 | 0.0000 | 0.8759 | 0.9719 |
| qwm | 8 | 0.3167 | 0.0000 | 0.6898 | 0.8759 |
| qwm | 16 | 0.3433 | 0.0000 | 0.6169 | 0.8276 |
| qwm | 32 | 0.3526 | 0.0000 | 0.5967 | 0.8038 |

## Limitations

- Planning success is 1/50 for QWM and 1/50 for latent-only. T1 is 0/11 versus 1/11.
- The world model has no image decoder. After the first frame, rollouts and CEM pass a zero image embedding.
- Latent-only planning does not optimize the physical task cost.
- Horizon contact F1 on val is 0 at every H for QWM.
- This comparison is the 50 frozen simulator seeds, not a retune on the dataset test split.
