"""Collect tabletop episodes (scripted, random, teleop, or mixed).

Debug gate (about 500 transitions):

    python scripts/collect.py --config configs/collect.yaml --episodes 10 --out data/debug

The 10k collection is a later run. This script does not start it.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwm.data.collect import collect_dataset          # noqa: E402
from qwm.utils.config import get as cfg_get           # noqa: E402
from qwm.utils.config import load_config              # noqa: E402
from qwm.utils.logging import get_logger              # noqa: E402

logger = get_logger("collect")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/collect.yaml")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    env_cfg = load_config(ROOT / cfg_get(cfg, "collect.env_config", "configs/env.yaml"))
    out = Path(args.out) if args.out is not None else None
    if out is not None and not out.is_absolute():
        out = ROOT / out
    elif out is None:
        raw_out = cfg_get(cfg, "collect.out", "data/debug")
        out = Path(raw_out)
        if not out.is_absolute():
            out = ROOT / out
    logger.info(
        "collect policy=%s episodes=%s out=%s",
        cfg_get(cfg, "collect.policy"),
        args.episodes if args.episodes is not None else cfg_get(cfg, "collect.episodes"),
        out,
    )
    collect_dataset(cfg, episodes=args.episodes, out=out, env_config=env_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
