"""Plain YAML config loading (no Hydra; YAML + argparse only).

Configs are plain mappings. ``merge`` performs a recursive dict merge used
for layering e.g. model configs over train defaults. ``save_config`` writes
an exact JSON snapshot used to stamp every run with the config it used.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import yaml

PathLike = Union[str, Path]


def load_config(path: PathLike) -> dict:
    """Load a YAML file and return its root mapping. Fail loud on problems."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise ValueError(f"config root must be a mapping, got {type(cfg).__name__}: {p}")
    return cfg


def merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Recursively merge ``override`` into ``base`` (dicts merged, leaves replaced)."""
    out: dict = copy.deepcopy(dict(base))
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def get(cfg: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    """Fetch a value by dotted path, e.g. ``get(cfg, "train.batch_size", 16)``."""
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return default
        node = node[part]
    return node


def save_config(cfg: Mapping[str, Any], path: PathLike) -> Path:
    """Write an exact, sorted JSON snapshot of the config (for run stamping)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(dict(cfg), f, indent=2, sort_keys=True, default=str)
    return p
