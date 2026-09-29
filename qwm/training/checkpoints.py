"""Minimal state-dict checkpoints. Metadata stays in a sidecar JSON."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Union

import torch
import torch.nn as nn

PathLike = Union[str, Path]


def save_checkpoint(path: PathLike, model: nn.Module, metadata: Dict) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    sidecar = out.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def load_checkpoint(path: PathLike, model: nn.Module, device: str = "cpu") -> Dict:
    blob = torch.load(Path(path), map_location=device, weights_only=True)
    model.load_state_dict(blob)
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.is_file():
        raise FileNotFoundError(f"checkpoint metadata missing: {sidecar}")
    return json.loads(sidecar.read_text(encoding="utf-8"))
