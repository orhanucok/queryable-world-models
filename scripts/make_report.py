"""One-command report from the frozen val horizon table and the 50-episode plan.

    python scripts/make_report.py --data data --runs runs --out reports

The planning answer is the comparison of closed-loop success rates.
A tie or a lower QWM rate is reported as no.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(path: Path):
    if not path.is_file():
        raise SystemExit(f"missing report input: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _rate(summary) -> float:
    return float(summary["success_rate"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--runs", default="runs")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    runs = Path(args.runs)
    if not runs.is_absolute():
        runs = ROOT / runs
    horizon = _load(out / "horizon_val.json")
    qwm = _load(runs / "qwm" / "planning.json")
    latent = _load(runs / "latent_only" / "planning.json")
    if qwm["planner"] != latent["planner"]:
        raise SystemExit("planner budgets differ; refusing to compare")
    q_rate = _rate(qwm["summary"])
    l_rate = _rate(latent["summary"])
    if q_rate > l_rate:
        answer = "yes"
    else:
        answer = "no"
    out.mkdir(parents=True, exist_ok=True)
    _plots(out, horizon, qwm, latent)
    text = _markdown(horizon, qwm, latent, answer, q_rate, l_rate)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(f"answer: {answer}")
    print(f"qwm {qwm['summary']['success']}/{qwm['summary']['episodes']}  latent {latent['summary']['success']}/{latent['summary']['episodes']}")
    print(f"wrote {out / 'report.md'}")
    return 0


def _plots(out: Path, horizon: dict, qwm: dict, latent: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = horizon["models"]["qwm"]["rows"]
    hs = [r["horizon"] for r in rows]
    pos = [r["pos_err_m"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hs, pos, marker="o")
    ax.set_xlabel("horizon")
    ax.set_ylabel("position error (m)")
    ax.set_title("QWM open-loop position error on val")
    ax.set_xticks(hs)
    fig.tight_layout()
    fig.savefig(out / "horizon_pos_error.png", dpi=120)
    plt.close(fig)

    tasks = ["1", "2", "3"]
    labels = ["T1 push", "T2 knock/keep", "T3 off-table"]
    q_rates = []
    l_rates = []
    for task in tasks:
        qb = qwm["summary"]["by_task"][task]
        lb = latent["summary"]["by_task"][task]
        q_rates.append(qb["success"] / qb["episodes"])
        l_rates.append(lb["success"] / lb["episodes"])
    import numpy as np
    x = np.arange(len(tasks))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - 0.15, q_rates, width=0.3, label="QWM")
    ax.bar(x + 0.15, l_rates, width=0.3, label="latent-only")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.set_ylabel("success rate")
    ax.set_title("Closed-loop success on 50 matched episodes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "planning_success.png", dpi=120)
    plt.close(fig)


def _markdown(horizon, qwm, latent, answer, q_rate, l_rate) -> str:
    lines = [
        "# Queryable World Models — measured report",
        "",
        "Question: does queryable state beat latent-only for planning?",
        "",
        f"**Answer: {answer}.**",
        "",
        f"On 50 matched simulator episodes, QWM closed-loop success was "
        f"{qwm['summary']['success']}/{qwm['summary']['episodes']} "
        f"({q_rate:.3f}) and latent-only was "
        f"{latent['summary']['success']}/{latent['summary']['episodes']} "
        f"({l_rate:.3f}). "
        "Beat means a strictly higher success rate under the same CEM budget and the same seeds.",
        "",
        "## Planning",
        "",
        "QWM minimized a cost on the predicted query state "
        "(target distance plus off-table penalty, upright probability, or on-table probability). "
        "Latent-only has no query head and was not given one. "
        "Its CEM cost is the mean squared distance between the candidate latent rollout "
        "and a zero-action rollout from the same image: minimized for T2 keep, maximized otherwise. "
        "That cost does not know the goal coordinates or the object pose.",
        "",
        f"CEM budget for both: horizon {qwm['planner']['horizon']}, "
        f"population {qwm['planner']['population']}, elites {qwm['planner']['elites']}, "
        f"iterations {qwm['planner']['iterations']}. "
        f"Parameters: QWM {qwm['parameters']}, latent-only {latent['parameters']}.",
        "",
        "| task | QWM | latent-only |",
        "|---|---:|---:|",
    ]
    for task, label in (("1", "T1"), ("2", "T2"), ("3", "T3")):
        qb = qwm["summary"]["by_task"][task]
        lb = latent["summary"]["by_task"][task]
        lines.append(
            f"| {label} | {qb['success']}/{qb['episodes']} | {lb['success']}/{lb['episodes']} |"
        )
    lines += [
        "",
        "## Horizon metrics (validation only)",
        "",
        "These numbers are the open-loop val table. The dataset test split was not read.",
        "",
        "| model | H | pos err (m) | contact F1 | upright acc | on-table acc |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, block in horizon["models"].items():
        for row in block["rows"]:
            def cell(key):
                value = row[key]
                return "n/a" if value is None else f"{value:.4f}"
            lines.append(
                f"| {name} | {row['horizon']} | {cell('pos_err_m')} | {cell('contact_f1')} | "
                f"{cell('upright_acc')} | {cell('on_table_acc')} |"
            )
    lines += [
        "",
        "## Limitations",
        "",
        f"- Planning success is {qwm['summary']['success']}/{qwm['summary']['episodes']} for QWM "
        f"and {latent['summary']['success']}/{latent['summary']['episodes']} for latent-only. "
        f"T1 is {qwm['summary']['by_task']['1']['success']}/{qwm['summary']['by_task']['1']['episodes']} "
        f"versus {latent['summary']['by_task']['1']['success']}/{latent['summary']['by_task']['1']['episodes']}.",
        "- The world model has no image decoder. After the first frame, rollouts and CEM pass a zero image embedding.",
        "- Latent-only planning does not optimize the physical task cost.",
        "- Horizon contact F1 on val is 0 at every H for QWM.",
        "- This comparison is the 50 frozen simulator seeds, not a retune on the dataset test split.",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
