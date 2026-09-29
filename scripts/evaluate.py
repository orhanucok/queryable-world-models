"""Horizon evaluation on the val split. Refuses the test split.

    python scripts/evaluate.py --config configs/eval.yaml

Prints position error (meters), contact F1, upright accuracy, and on_table
accuracy at H=1,8,16,32. Latent-only has no query head, so those cells are n/a.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwm.data.dataset import SequenceDataset                         # noqa: E402
from qwm.data.splits import load_manifest, manifest_path              # noqa: E402
from qwm.evaluation.horizon import evaluate_horizons, format_table    # noqa: E402
from qwm.evaluation.protocol import HORIZONS, check_horizons, require_dev_split  # noqa: E402
from qwm.models import build_model, trainable_parameters              # noqa: E402
from qwm.training.checkpoints import load_checkpoint                  # noqa: E402
from qwm.utils.config import get as cfg_get                           # noqa: E402
from qwm.utils.config import load_config                              # noqa: E402
from qwm.utils.device import get_device                               # noqa: E402


def _eval_run(run_dir: Path, data_dir: Path, manifest, device, batch_size: int):
    meta_path = run_dir / "best.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"missing checkpoint metadata: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    model_cfg = meta["model_config"]
    model = build_model(model_cfg)
    load_checkpoint(run_dir / "best.pt", model, device=str(device))
    counted = trainable_parameters(model)
    saved = int(meta["parameters"])
    if counted != saved:
        raise RuntimeError(f"{run_dir.name} parameter count {counted} != saved {saved}")
    model.to(device)
    split = require_dev_split("val")
    dataset = SequenceDataset(
        data_dir,
        [int(i) for i in manifest["episode_ids"][split]],
        max(HORIZONS) + 1,
    )
    rows = evaluate_horizons(model, dataset, HORIZONS, batch_size, device)
    return {"parameters": counted, "best_step": meta.get("best_step"), "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--split", default=None)
    parser.add_argument("--run", action="append", default=None,
                        help="checkpoint directory; repeat for several. Default: runs/qwm and runs/latent_only")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    split = args.split if args.split is not None else cfg_get(cfg, "eval.split", "val")
    require_dev_split(split)
    if split != "val":
        raise SystemExit("this evaluation is val only")
    check_horizons(HORIZONS)
    data_dir = Path(args.data) if args.data else Path(cfg_get(cfg, "eval.data_dir", "data"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    manifest = load_manifest(manifest_path(data_dir))
    device = get_device("auto")
    batch_size = int(cfg_get(cfg, "eval.batch_size", 16))
    run_root = ROOT / cfg_get(cfg, "eval.run_root", "runs")
    if args.run:
        run_dirs = [Path(p) if Path(p).is_absolute() else ROOT / p for p in args.run]
    else:
        run_dirs = [run_root / "qwm", run_root / "latent_only"]
    rows_by_model = {}
    payload = {"split": "val", "test_used": False, "horizons": list(HORIZONS), "models": {}}
    for run_dir in run_dirs:
        result = _eval_run(run_dir, data_dir, manifest, device, batch_size)
        rows_by_model[run_dir.name] = result["rows"]
        payload["models"][run_dir.name] = result
    text = format_table(rows_by_model)
    print(text)
    out = ROOT / "reports" / "horizon_val.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "reports" / "horizon_val.txt").write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
