"""Deterministic seeding utilities.

Rules:
- Every experiment/collection run receives ONE integer seed.
- All stochastic draws go through numpy Generator objects (PCG64) created by
  ``make_rng``; global RNG state is only seeded for third-party libraries
  that rely on it (python ``random``, ``numpy.random``, ``torch``).
- Episode-level seeds are DERIVED deterministically from a parent seed plus a
  domain label via SeedSequence spawning, so episode i always receives the
  same seed regardless of machine, order, or process.
- ``set_global_seed`` is idempotent and safe to call repeatedly.

The domain label uses CRC32 (stable across runs/processes), never ``hash()``
(which is randomized for strings by PYTHONHASHSEED).
"""
from __future__ import annotations

import os
import random
import zlib
from typing import List

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed python ``random``, ``numpy.random`` and (if installed) ``torch``.

    Idempotent: calling twice with the same seed is a no-op in effect.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:  # torch is optional until model training steps
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def make_rng(seed: int) -> np.random.Generator:
    """Create a deterministic PCG64 generator from an integer seed."""
    return np.random.default_rng(np.random.SeedSequence(int(seed)))


def spawn_seeds(seed: int, n: int, domain: str = "") -> List[int]:
    """Derive ``n`` child seeds deterministically from a parent seed.

    The domain label (e.g. ``"collect"``, ``"planning_eval"``) partitions
    seed spaces so the same parent seed produces different, non-colliding
    streams per usage domain.
    """
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    key = (zlib.crc32(domain.encode("utf-8")) & 0xFFFFFFFF,) if domain else ()
    children = np.random.SeedSequence(entropy=int(seed), spawn_key=key).spawn(n)
    return [
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in children
    ]
