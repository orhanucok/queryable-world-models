# Queryable World Models: Predicting Outcomes, Not Pixels

Technical report DOI: https://doi.org/10.5281/zenodo.23040147

Zenodo record: https://zenodo.org/records/23040147

**Technical Report**  
Orhan Üçok, BAİBÜ Computer Engineering

This repository accompanies a technical report on an action-conditioned CNN+GRU that predicts named physical query outputs through a residual update from a supplied previous state. The retained validation result supports conditional prediction when that state is repeatedly available, but closed-loop error grows, contact prediction fails, and the tested single-frame encoder does not localize the object at task tolerance. These measurements do not establish successful planning or a complete perception-to-control system.

[Read the technical report](papers/qwm_tech_report_v3.pdf). The numerical sources are [`BEST_SO_FAR.md`](reports/BEST_SO_FAR.md), [`VISION_BEST.md`](reports/VISION_BEST.md), and [`STATUS.md`](reports/STATUS.md). Appendix B of the PDF records unresolved checkpoint-provenance and action/state-timing concerns.

## Locked results

Residual QWM: **973243 parameters**, **step 5000**, **λ_pos=10**, **oneobj5k**.

| Closed-loop horizon H | Object L2 (m) | Gripper L2 (m) | Contact F1 |
|---:|---:|---:|---:|
| 1 | 0.0407 | 0.0255 | 0.0000 |
| 8 | 0.1733 | 0.0622 | 0.0000 |
| 16 | 0.2447 | 0.1238 | 0.0000 |
| 32 | 0.3679 | 0.2749 | 0.0000 |

| Reported comparison | Object L2 (m) |
|---|---:|
| Teacher-forced H=8 | 0.055 |
| Closed-loop H=8 | 0.173 |

| Single-frame vision | Object L2 (m) | Gripper L2 (m) |
|---|---:|---:|
| Original 84×84 | 0.132 | 0.031 |
| Original 128×128 | 0.132 | 0.028 |
| Top-down, discarded | 0.441 | Not reported here |

The closed-loop H=8 object-L2 planning gate is **≤0.10 m** and is **CLOSED**. CEM was not run for the retained checkpoint. **Technical report. Planning gate closed.**

## Code and scope

`qwm/` contains the model, PyBullet environment, data utilities, evaluation code, and planning implementation. `configs/`, `scripts/`, `tests/`, and [`schema.md`](schema.md) are included so the source can be inspected and its invariants tested. A small `data/debug` fixture is tracked solely for the source tests. Install [`requirements.txt`](requirements.txt) in a compatible Python environment and run `python -m pytest -q` to check the source. This public repository excludes the full collected datasets, rollouts, and trained checkpoints; it cannot independently reproduce the PDF's measurements without recovering the exact checkpoint and evaluation invocation described in Appendix B.

[`reports/report.md`](reports/report.md) is a historical report from an earlier configuration. Its planning and horizon values must not be attributed to the retained step-5000 residual checkpoint or substituted for the locked tables above.

## Citation

Üçok, O. (2026). Queryable World Models: Predicting Outcomes, Not Pixels
(Technical Report). Zenodo. https://doi.org/10.5281/zenodo.23040147
