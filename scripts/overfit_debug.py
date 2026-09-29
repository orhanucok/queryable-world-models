"""Overfit QWM on exactly 16 train-split sequences from the debug set.

    python scripts/overfit_debug.py --config configs/model_qwm.yaml --data data/debug

Prints parameter counts and the first/last per-head losses. Exits non-zero
if the weighted total does not fall to the configured fraction of its start.
Does not collect data and does not run the planner.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qwm.data.dataset import SequenceDataset          # noqa: E402
from qwm.data.splits import load_manifest, manifest_path  # noqa: E402
from qwm.models import build_model, trainable_parameters  # noqa: E402
from qwm.training.checkpoints import save_checkpoint  # noqa: E402
from qwm.training.trainer import overfit_qwm          # noqa: E402
from qwm.utils.config import get as cfg_get           # noqa: E402
from qwm.utils.config import load_config, merge       # noqa: E402
from qwm.utils.device import get_device               # noqa: E402
from qwm.utils.logging import get_logger              # noqa: E402
from qwm.utils.seed import set_global_seed            # noqa: E402

logger = get_logger("overfit_debug")

HEADS = ("grip_xyz", "obj_xyz", "upright", "contact", "on_table", "total")


def _format_losses(prefix: str, losses: dict) -> str:
    body = " ".join(f"{name}={losses[name]:.6f}" for name in HEADS)
    return f"{prefix}: {body}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/model_qwm.yaml")
    parser.add_argument("--train", default="configs/train.yaml")
    parser.add_argument("--data", default=None)
    args = parser.parse_args()

    model_cfg = load_config(ROOT / args.config)
    train_cfg = load_config(ROOT / args.train)
    if model_cfg.get("model", {}).get("name") != "qwm":
        raise SystemExit("the overfit gate trains QWM query heads; pass configs/model_qwm.yaml")
    cfg = merge(train_cfg, model_cfg)
    data_dir = Path(args.data) if args.data else Path(cfg_get(cfg, "data.dir"))
    if not data_dir.is_absolute():
        data_dir = ROOT / data_dir
    if cfg_get(cfg, "data.image_norm") != "uint8_div_255":
        raise SystemExit("only image_norm uint8_div_255 is implemented")

    seed = int(cfg["seed"])
    set_global_seed(seed)
    manifest = load_manifest(manifest_path(data_dir))
    length = int(cfg["overfit"]["sequence_length"])
    train_ids = manifest["episode_ids"]["train"]
    dataset = SequenceDataset(data_dir, train_ids, length)
    device = get_device(str(cfg.get("device", "auto")))
    model = build_model(model_cfg)
    latent = build_model(load_config(ROOT / "configs" / "model_latent.yaml"))
    qwm_n = trainable_parameters(model)
    latent_n = trainable_parameters(latent)
    cap = 20_000_000
    if qwm_n >= cap or latent_n >= cap:
        raise SystemExit(f"parameter cap {cap} exceeded: qwm={qwm_n} latent_only={latent_n}")
    if latent_n >= qwm_n:
        raise SystemExit("latent-only should be the QWM backbone without query heads")

    logger.info("device=%s qwm_params=%d latent_only_params=%d sequences=16 length=%d",
                device, qwm_n, latent_n, length)
    result = overfit_qwm(model, dataset, cfg, device)
    used_ids = sorted({int(dataset.window_episode_ids()[i]) for i in result["indices"]})
    forbidden = set(manifest["episode_ids"]["val"]) | set(manifest["episode_ids"]["test"])
    if set(used_ids) & forbidden:
        raise SystemExit(f"overfit sequences leaked outside train: {used_ids}")

    print(f"parameters qwm={qwm_n} latent_only={latent_n}")
    print(_format_losses("per-head loss first", result["first"]))
    print(_format_losses("per-head loss last", result["last"]))
    print(f"overfit_episode_ids={used_ids}")
    save_checkpoint(
        ROOT / "runs" / "overfit" / "qwm_overfit.pt",
        model,
        {
            "qwm_parameters": qwm_n,
            "latent_only_parameters": latent_n,
            "first": result["first"],
            "last": result["last"],
            "episode_ids": used_ids,
            "indices": result["indices"],
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
