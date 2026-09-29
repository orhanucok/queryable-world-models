"""CUDA/CPU device selection.

torch is imported LAZILY inside ``get_device`` so that schema/data/env
tooling runs on machines without torch installed. Once a device is
requested, torch is required and failures are loud.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_device(prefer: str = "auto"):
    """Return a torch device.

    prefer:
      - "auto": CUDA if available, else CPU
      - "cuda" / "cuda:0": CUDA, loud RuntimeError if unavailable
      - "cpu": CPU
    """
    import torch  # lazy on purpose; fail loud if missing at this point

    prefer = str(prefer).lower()
    if prefer == "auto":
        prefer = "cuda" if torch.cuda.is_available() else "cpu"
    if prefer.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available() is False; "
                "check the CUDA build of torch and the driver"
            )
        device = torch.device(prefer)
        logger.info(
            "device: %s (%s)", device, torch.cuda.get_device_name(0)
        )
        return device
    if prefer == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device preference: {prefer!r}")
