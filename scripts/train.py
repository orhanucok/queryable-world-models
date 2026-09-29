"""Train one model on the 10k train split. Val picks the checkpoint. Test is not read.

    python scripts/train.py --config configs/model_qwm.yaml --data data
    python scripts/train.py --config configs/model_latent.yaml --data data

Both invocations share configs/train.yaml (steps, seed, sequence length, lr).
Batch 16 is attempted first. A CUDA OOM retries at batch 8 without changing
the model.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwm.data.dataset import SequenceDataset, make_loader          # noqa: E402
from qwm.data.splits import load_manifest, manifest_path           # noqa: E402
from qwm.evaluation.protocol import require_dev_split              # noqa: E402
from qwm.models import build_model, trainable_parameters           # noqa: E402
from qwm.training.checkpoints import save_checkpoint               # noqa: E402
from qwm.training.trainer import train_model                       # noqa: E402
from qwm.utils.config import load_config, merge, save_config       # noqa: E402
from qwm.utils.device import get_device                            # noqa: E402
from qwm.utils.logging import JsonlLogger, get_logger, stamp_run   # noqa: E402
from qwm.utils.seed import set_global_seed                         # noqa: E402

logger = get_logger("train")


def _ids(manifest, split: str):
    require_dev_split(split)
    return [int(i) for i in manifest["episode_ids"][split]]


def _run_once(cfg, model_cfg, train_ds, val_ds, device, batch_size: int, run_dir: Path, select_ds=None):
    set_global_seed(int(cfg["seed"]))
    model = build_model(model_cfg)
    n_params = trainable_parameters(model)
    kind = model_cfg["model"]["name"]
    seed = int(cfg["seed"])
    train_loader = make_loader(train_ds, batch_size, shuffle=True, seed=seed, num_workers=0)
    val_loader = make_loader(val_ds, batch_size, shuffle=False, seed=seed, num_workers=0)
    select_loader = None
    if select_ds is not None:
        select_loader = make_loader(select_ds, batch_size, shuffle=False, seed=seed, num_workers=0)
    log_path = run_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()
    logger_json = JsonlLogger(log_path)
    t0 = time.perf_counter()

    def on_step(record):
        logger_json.log(record)
        elapsed = time.perf_counter() - t0
        step = int(record["step"])
        if step == 0:
            logger.info("step 0 val total=%.4f", record["total"])
            return
        rate = step / max(elapsed, 1e-6)
        left = (int(cfg["train"]["steps"]) - step) / max(rate, 1e-6)
        logger.info(
            "step %d %s total=%.4f  %.2f step/s  eta %.0fs",
            step, record["split"], record["total"], rate, left,
        )

    try:
        result = train_model(
            model, train_loader, val_loader, cfg, device, kind,
            on_step=on_step, select_loader=select_loader,
        )
    finally:
        logger_json.close()
    meta = {
        "parameters": n_params,
        "batch_size": batch_size,
        "steps": result["steps"],
        "best_step": result["best_step"],
        "best_val": result["best_val"],
        "seed": seed,
        "kind": kind,
        "model_config": model_cfg,
        "test_used": False,
    }
    save_checkpoint(run_dir / "best.pt", model, meta)
    (run_dir / "parameters.txt").write_text(f"{n_params}\n", encoding="utf-8")
    logger.info(
        "done %s params=%d best_step=%d best_val_total=%.4f batch=%d",
        kind, n_params, result["best_step"], result["best_val"]["total"], batch_size,
    )
    return n_params, batch_size


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model_qwm.yaml")
    parser.add_argument("--train", default="configs/train.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--select", default=None)
    parser.add_argument("--rollout-k", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--own-prob", type=float, default=None)
    args = parser.parse_args()

    model_cfg = load_config(ROOT / args.config)
    train_cfg = load_config(ROOT / args.train)
    cfg = merge(train_cfg, model_cfg)
    if args.steps is not None:
        cfg["train"]["steps"] = int(args.steps)
    if args.select is not None:
        cfg["train"]["select"] = str(args.select)
    if args.rollout_k is not None:
        cfg["train"]["rollout_k"] = int(args.rollout_k)
    if args.seq_len is not None:
        cfg["train"]["sequence_length"] = int(args.seq_len)
    if args.own_prob is not None:
        cfg["train"]["own_prob_final"] = float(args.own_prob)
    kind = model_cfg["model"]["name"]
    if kind not in ("qwm", "latent_only"):
        raise SystemExit(f"unknown model {kind}")
    data_dir = Path(args.data) if args.data else Path(cfg["train"]["data_dir"])
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    cfg["train"]["data_dir"] = str(data_dir)
    manifest = load_manifest(manifest_path(data_dir))
    length = int(cfg["train"]["sequence_length"])
    train_ids = _ids(manifest, "train")
    val_ids = _ids(manifest, "val")
    train_ds = SequenceDataset(data_dir, train_ids, length)
    val_ds = SequenceDataset(data_dir, val_ids, length)
    select_ds = None
    if str(cfg["train"].get("select", "")) == "obj_l2_h8":
        select_ds = SequenceDataset(data_dir, val_ids, 33)
    device = get_device(str(cfg.get("device", "auto")))
    requested = int(args.batch_size if args.batch_size is not None else cfg["train"]["batch_size"])
    fallback = int(cfg["train"]["batch_size_fallback"])
    run_dir = ROOT / cfg["train"].get("run_root", "runs") / kind
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp_run(run_dir, cfg, extra={"kind": kind, "test_used": False, "parameters_note": "counted after build"})
    save_config(cfg, run_dir / "config.json")
    batches = [requested]
    if requested != fallback:
        batches.append(fallback)
    last_error = None
    for batch_size in batches:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            _run_once(cfg, model_cfg, train_ds, val_ds, device, batch_size, run_dir, select_ds)
            return 0
        except torch.cuda.OutOfMemoryError as exc:
            last_error = exc
            logger.warning("CUDA OOM at batch %d", batch_size)
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if batch_size == batches[-1]:
                break
            logger.warning("retrying at batch %d; model size is unchanged", fallback)
    raise SystemExit(f"training OOM at batch {batches[-1]}: {last_error}")


if __name__ == "__main__":
    raise SystemExit(main())
