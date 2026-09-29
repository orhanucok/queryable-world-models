"""QWM overfit gate: train on exactly 16 sequences and require the loss to drop."""
from __future__ import annotations

from typing import Dict, List

import torch

from qwm.data import schema
from qwm.data.dataset import make_loader
from qwm.training.losses import HEAD_NAMES, query_losses
from qwm.utils.seed import make_rng

OVERFIT_SEQUENCES = 16


def select_overfit_indices(n_windows: int, n_sequences: int, seed: int) -> List[int]:
    """Deterministic subset. The gate is exactly 16 sequences."""
    if int(n_sequences) != OVERFIT_SEQUENCES:
        raise ValueError(f"overfit gate uses exactly {OVERFIT_SEQUENCES} sequences, got {n_sequences}")
    if int(n_windows) < OVERFIT_SEQUENCES:
        raise ValueError(f"need at least {OVERFIT_SEQUENCES} windows, got {n_windows}")
    chosen = make_rng(int(seed)).choice(int(n_windows), size=OVERFIT_SEQUENCES, replace=False)
    return sorted(int(i) for i in chosen)


def _to_device(batch: Dict, device: torch.device) -> Dict:
    return {key: value.to(device) for key, value in batch.items()}


def _loss_on_batch(model, batch, weights):
    prev = batch["query"][:, :-1] if getattr(model, "residual", False) else None
    pred = model(batch["obs"], batch["action"], prev)["query"]
    target = batch["query"][:, 1:]
    mask = batch["object_valid_mask"][:, 1:]
    return query_losses(pred, target, mask, weights)


def evaluate_losses(model, loader, device: torch.device, weights: Dict[str, float]) -> Dict[str, float]:
    """Mean unweighted head losses and weighted total, over every sequence once."""
    was_training = model.training
    model.eval()
    sums = {name: 0.0 for name in list(HEAD_NAMES) + ["total"]}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            _total, logs = _loss_on_batch(model, batch, weights)
            taken = int(batch["obs"].shape[0])
            for name, value in logs.items():
                sums[name] += value * taken
            count += taken
    if count != OVERFIT_SEQUENCES:
        raise RuntimeError(f"evaluation saw {count} sequences, expected {OVERFIT_SEQUENCES}")
    if was_training:
        model.train()
    return {name: sums[name] / count for name in sums}


def overfit_qwm(model, dataset, cfg: dict, device: torch.device) -> Dict[str, Dict[str, float]]:
    """Train ``model`` on exactly 16 windows. Raise if the total loss does not drop."""
    over = cfg["overfit"]
    weights = cfg["loss"]
    n_sequences = int(over["num_sequences"])
    indices = select_overfit_indices(len(dataset), n_sequences, int(cfg["seed"]))
    subset = torch.utils.data.Subset(dataset, indices)
    batch_size = int(over.get("batch_size", cfg["data"]["batch_size"]))
    eval_loader = make_loader(subset, batch_size, shuffle=False, seed=int(cfg["seed"]))
    train_loader = make_loader(subset, batch_size, shuffle=True, seed=int(cfg["seed"]))
    steps = int(over["steps"])
    if steps < 1:
        raise ValueError(f"overfit steps must be >= 1, got {steps}")
    ratio = float(over["min_drop_ratio"])
    if not (0.0 < ratio < 1.0):
        raise ValueError(f"min_drop_ratio must be in (0, 1), got {ratio}")

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(over["lr"]),
        weight_decay=float(over.get("weight_decay", 0.0)),
    )
    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    grad_clip = float(over.get("grad_clip", 1.0))

    first = evaluate_losses(model, eval_loader, device, weights)
    model.train()
    step = 0
    while step < steps:
        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                total, _logs = _loss_on_batch(model, batch, weights)
            if not torch.isfinite(total):
                raise RuntimeError(f"non-finite loss at overfit step {step}")
            scaler.scale(total).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            step += 1
            if step >= steps:
                break
    last = evaluate_losses(model, eval_loader, device, weights)
    if not (last["total"] <= first["total"] * ratio):
        raise RuntimeError(
            "overfit gate failed: "
            f"total {first['total']:.6f} -> {last['total']:.6f} "
            f"(required <= {ratio} times the start)"
        )
    return {"first": first, "last": last, "indices": indices}


def _repeat_loader(loader):
    while True:
        for batch in loader:
            yield batch


def _finite(loss: torch.Tensor, step: int) -> torch.Tensor:
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss at step {step}")
    return loss


def _query_batch_loss(model, batch, weights):
    k = int(getattr(model, "rollout_k", 0) or 0)
    if k > 0:
        if batch["obs"].shape[1] < k + 1:
            raise RuntimeError(
                f"rollout K={k} needs sequence length >= {k + 1}, got {batch['obs'].shape[1]}"
            )
        pred = model.rollout(
            batch["obs"][:, 0],
            batch["action"][:, :k],
            batch["query"][:, 0],
            teacher=batch["query"][:, :k],
            own_prob=float(getattr(model, "own_prob", 1.0)),
        )["query"]
        target = batch["query"][:, 1:k + 1]
        mask = batch["object_valid_mask"][:, 1:k + 1]
        return query_losses(pred, target, mask, weights)
    return _loss_on_batch(model, batch, weights)


def _latent_batch_loss(model, batch):
    from qwm.training.losses import latent_prediction_loss

    loss = latent_prediction_loss(model, batch["obs"], batch["action"])
    value = float(loss.detach())
    return loss, {"latent": value, "total": value}


def mean_obj_l2_h(model, loader, device, h: int = 1) -> float:
    """Closed-loop valid-object L2 in metres at horizon ``h``. Pads are excluded."""
    if h < 1:
        raise ValueError(f"horizon must be >= 1, got {h}")
    was_training = model.training
    model.eval()
    err_sum = 0.0
    count = 0
    scale = torch.tensor(schema.POS_SCALE, device=device, dtype=torch.float32)
    origin = torch.tensor(schema.POS_ORIGIN, device=device, dtype=torch.float32)
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            if batch["obs"].shape[1] < h + 1:
                raise RuntimeError(
                    f"val sequence length {batch['obs'].shape[1]} is shorter than H={h}"
                )
            pred = model.rollout(
                batch["obs"][:, 0], batch["action"][:, :h], batch["query"][:, 0]
            )["query"][:, h - 1]
            target = batch["query"][:, h]
            mask = batch["object_valid_mask"][:, h].to(dtype=torch.bool)
            pred_obj = pred[:, schema.OBJ_XYZ].reshape(pred.shape[0], schema.NMAX, 3) * scale + origin
            true_obj = target[:, schema.OBJ_XYZ].reshape(target.shape[0], schema.NMAX, 3) * scale + origin
            dist = torch.linalg.norm(pred_obj - true_obj, dim=-1)
            err_sum += float((dist * mask).sum())
            count += int(mask.sum())
    if count < 1:
        raise RuntimeError("validation set has no valid objects")
    if was_training:
        model.train()
    return err_sum / count


def mean_obj_l2_h1(model, loader, device) -> float:
    """Mean valid-object Euclidean error in meters at open-loop H=1.

    The first teacher-forced step uses only the current frame and action,
    so it is the H=1 rollout. Lower is better. Pads are excluded.
    """
    was_training = model.training
    model.eval()
    err_sum = 0.0
    count = 0
    scale = torch.tensor(schema.POS_SCALE, device=device, dtype=torch.float32)
    origin = torch.tensor(schema.POS_ORIGIN, device=device, dtype=torch.float32)
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            prev = batch["query"][:, :-1] if getattr(model, "residual", False) else None
            pred = model(batch["obs"], batch["action"], prev)["query"][:, 0]
            target = batch["query"][:, 1]
            mask = batch["object_valid_mask"][:, 1].to(dtype=torch.bool)
            pred_obj = pred[:, schema.OBJ_XYZ].reshape(pred.shape[0], schema.NMAX, 3) * scale + origin
            true_obj = target[:, schema.OBJ_XYZ].reshape(target.shape[0], schema.NMAX, 3) * scale + origin
            dist = torch.linalg.norm(pred_obj - true_obj, dim=-1)
            err_sum += float((dist * mask).sum())
            count += int(mask.sum())
    if count < 1:
        raise RuntimeError("validation set has no valid objects")
    if was_training:
        model.train()
    return err_sum / count


def mean_split_losses(model, loader, device, kind: str, weights=None) -> Dict[str, float]:
    """Average logged losses over a loader. Does not touch a test split."""
    was_training = model.training
    model.eval()
    sums: Dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            if kind == "qwm":
                _total, logs = _query_batch_loss(model, batch, weights)
            elif kind == "latent_only":
                _total, logs = _latent_batch_loss(model, batch)
            else:
                raise ValueError(f"unknown model kind {kind!r}")
            taken = int(batch["obs"].shape[0])
            for name, value in logs.items():
                sums[name] = sums.get(name, 0.0) + float(value) * taken
            count += taken
    if count < 1:
        raise RuntimeError("validation loader was empty")
    if was_training:
        model.train()
    return {name: value / count for name, value in sums.items()}


def train_model(model, train_loader, val_loader, cfg: dict, device, kind: str, on_step=None, select_loader=None) -> Dict:
    """Fixed-step training. Updates see only the train loader. Best checkpoint is by val total."""
    train_cfg = cfg["train"]
    steps = int(train_cfg["steps"])
    if steps < 1:
        raise ValueError(f"train.steps must be >= 1, got {steps}")
    weights = cfg.get("loss")
    if kind == "qwm" and not weights:
        raise KeyError("QWM training requires loss weights")
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    use_amp = bool(cfg.get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    val_every = int(train_cfg["val_every"])
    log_every = int(train_cfg.get("log_every", 50))

    select = str(train_cfg.get("select", "total"))
    if select not in ("total", "obj_l2_h1", "obj_l2_h8"):
        raise ValueError(f"train.select must be total, obj_l2_h1, or obj_l2_h8, got {select}")
    if select != "total" and kind != "qwm":
        raise ValueError(f"{select} selection needs query heads")
    model.rollout_k = int(train_cfg.get("rollout_k", 0) or 0)
    own_final = float(train_cfg.get("own_prob_final", 0.0) or 0.0)
    if own_final < 0.0 or own_final > 1.0:
        raise ValueError(f"own_prob_final must be in [0, 1], got {own_final}")
    # own_prob_final == 0 means pure closed loop (no scheduled sampling).
    model.own_prob = 0.0 if own_final > 0.0 else 1.0

    def val_losses():
        saved_own = float(getattr(model, "own_prob", 1.0))
        model.own_prob = 1.0
        try:
            current = mean_split_losses(model, val_loader, device, kind, weights)
            if select == "obj_l2_h1":
                current["obj_l2_h1"] = mean_obj_l2_h1(model, val_loader, device)
            if select == "obj_l2_h8":
                metric_loader = select_loader if select_loader is not None else val_loader
                current["obj_l2_h8"] = mean_obj_l2_h(model, metric_loader, device, h=8)
        finally:
            model.own_prob = saved_own
        return current

    history = []
    best = val_losses()
    best_step = 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history.append({"step": 0, "split": "val", **best})
    if on_step is not None:
        on_step({"step": 0, "split": "val", **best})

    model.train()
    window = {}
    window_n = 0
    stream = _repeat_loader(train_loader)
    for step in range(1, steps + 1):
        batch = _to_device(next(stream), device)
        if model.rollout_k > 0 and own_final > 0.0:
            model.own_prob = own_final * (step / steps)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if kind == "qwm":
                loss, logs = _query_batch_loss(model, batch, weights)
            else:
                loss, logs = _latent_batch_loss(model, batch)
        _finite(loss, step)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        taken = int(batch["obs"].shape[0])
        for name, value in logs.items():
            window[name] = window.get(name, 0.0) + float(value) * taken
        window_n += taken
        if step % log_every == 0 or step == steps:
            record = {"step": step, "split": "train", "own_prob": float(getattr(model, "own_prob", 0.0)), **{k: v / window_n for k, v in window.items()}}
            history.append(record)
            if on_step is not None:
                on_step(record)
            window = {}
            window_n = 0
        if step % val_every == 0 or step == steps:
            current = val_losses()
            record = {"step": step, "split": "val", **current}
            history.append(record)
            if on_step is not None:
                on_step(record)
            score = select if select != "total" else "total"
            if current[score] < best[score]:
                best = current
                best_step = step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            model.train()
    model.load_state_dict(best_state)
    return {"history": history, "best_step": best_step, "best_val": best, "steps": steps}
