"""Score open-loop query predictions at the frozen horizons. Val split only."""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch

from qwm.data import schema
from qwm.data.dataset import SequenceDataset, make_loader
from qwm.evaluation.protocol import (
    HORIZONS,
    binary_accuracy,
    binary_f1,
    grip_l2_m,
    obj_l2_m,
    position_error_m,
    require_dev_split,
)
from qwm.evaluation.rollout import open_loop


def _cat(chunks: List[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def evaluate_horizons(model, dataset: SequenceDataset, horizons, batch_size: int, device) -> List[Dict]:
    """Return one row per horizon. Models without query heads get null physical metrics."""
    require_dev_split("val")
    horizons = tuple(int(h) for h in horizons)
    if max(horizons) + 1 > dataset.sequence_length:
        raise ValueError(
            f"dataset sequence_length {dataset.sequence_length} is shorter than max horizon {max(horizons)}"
        )
    loader = make_loader(dataset, int(batch_size), shuffle=False, seed=0)
    model.eval()
    has_heads = getattr(model, "heads", None) is not None
    buckets = {
        h: {"pred": [], "target": [], "mask": []}
        for h in horizons
    }
    for batch in loader:
        obs0 = batch["obs"][:, 0].to(device)
        actions = batch["action"].to(device)
        query0 = batch["query"][:, 0].to(device)
        query = batch["query"].numpy()
        mask = batch["object_valid_mask"].numpy()
        rolled = open_loop(model, obs0, actions[:, : max(horizons)], query0)
        pred_q = rolled["query"].float().cpu().numpy() if has_heads else None
        for h in horizons:
            buckets[h]["target"].append(query[:, h])
            buckets[h]["mask"].append(mask[:, h])
            if pred_q is not None:
                buckets[h]["pred"].append(pred_q[:, h - 1])
    rows = []
    for h in horizons:
        target = _cat(buckets[h]["target"])
        mask = _cat(buckets[h]["mask"]).astype(bool)
        if not has_heads:
            rows.append({
                "horizon": h,
                "pos_err_m": None,
                "grip_l2_m": None,
                "obj_l2_m": None,
                "contact_f1": None,
                "upright_acc": None,
                "on_table_acc": None,
                "query_head": False,
            })
            continue
        pred = _cat(buckets[h]["pred"])
        rows.append({
            "horizon": h,
            "pos_err_m": position_error_m(pred, target, mask),
            "grip_l2_m": grip_l2_m(pred, target, mask),
            "obj_l2_m": obj_l2_m(pred, target, mask),
            "contact_f1": binary_f1(pred[:, schema.OBJ_CONTACT], target[:, schema.OBJ_CONTACT], mask),
            "upright_acc": binary_accuracy(pred[:, schema.OBJ_UPRIGHT], target[:, schema.OBJ_UPRIGHT], mask),
            "on_table_acc": binary_accuracy(pred[:, schema.OBJ_ON_TABLE], target[:, schema.OBJ_ON_TABLE], mask),
            "query_head": True,
            "n": int(pred.shape[0]),
        })
    return rows


def format_table(rows_by_model: Dict[str, List[Dict]]) -> str:
    """Plain table. Null metrics print as n/a."""
    header = (
        f"{'model':<14} {'H':>4} {'grip_l2_m':>10} {'obj_l2_m':>10} "
        f"{'contact_f1':>11} {'upright_acc':>12} {'on_table_acc':>13}"
    )
    lines = [
        "horizon eval on val only; test split was not read",
        header,
    ]
    for name, rows in rows_by_model.items():
        for row in rows:
            def cell(key, digits):
                value = row[key]
                return "n/a" if value is None else f"{value:.{digits}f}"
            lines.append(
                f"{name:<14} {row['horizon']:>4} {cell('grip_l2_m', 4):>10} {cell('obj_l2_m', 4):>10} "
                f"{cell('contact_f1', 4):>11} {cell('upright_acc', 4):>12} {cell('on_table_acc', 4):>13}"
            )
    if any(row["pos_err_m"] is None for rows in rows_by_model.values() for row in rows):
        lines.append("latent-only has no query heads, so position/contact/upright/on_table are not defined for it.")
    return "\n".join(lines)
