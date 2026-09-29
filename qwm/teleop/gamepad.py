"""Optional Xbox/gamepad reader for the frozen ee-delta + gripper action.

Collection never requires a controller. ``controller_available`` returns
False when pygame is missing, SDL cannot start, or no joystick is connected.
``read_action`` fails loud: a teleop episode is never silently relabeled as
random or scripted.

Documented axis map (pygame Xbox-style layout, v0):
  axis 0 -> dx
  axis 1 -> dy
  axis 3 -> dz when at least 4 axes exist, otherwise dz = 0
  button 0 held -> gripper -1 (closed), otherwise gripper +1 (open)
Each component is clipped to [-1, 1].
"""
from __future__ import annotations

import os

import numpy as np

from qwm.data import schema

_PYGAME = None
_INIT_FAILED = False


def _ensure_pygame():
    """Import and init pygame once. Raise the original error on failure."""
    global _PYGAME, _INIT_FAILED
    if _PYGAME is not None:
        return _PYGAME
    if _INIT_FAILED:
        raise RuntimeError("pygame failed to initialize")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    try:
        import pygame
    except ImportError as exc:
        _INIT_FAILED = True
        raise RuntimeError("pygame is not installed") from exc
    try:
        pygame.init()
        pygame.joystick.init()
    except Exception:
        _INIT_FAILED = True
        raise
    _PYGAME = pygame
    return pygame


def controller_available() -> bool:
    """True only when a joystick is connected. Never raises."""
    try:
        pygame = _ensure_pygame()
        return int(pygame.joystick.get_count()) > 0
    except Exception:
        return False


def read_action() -> np.ndarray:
    """Poll joystick 0 once and return float32[4] in [-1, 1]."""
    try:
        pygame = _ensure_pygame()
    except Exception as exc:
        raise RuntimeError(f"no gamepad detected: {exc}") from exc
    pygame.event.pump()
    if pygame.joystick.get_count() < 1:
        raise RuntimeError("no gamepad detected")
    pad = pygame.joystick.Joystick(0)
    pad.init()
    if pad.get_numaxes() < 2:
        raise RuntimeError(f"gamepad {pad.get_name()!r} has fewer than 2 axes")
    action = np.zeros(schema.ACTION_DIM, dtype=np.float32)
    action[0] = float(pad.get_axis(0))
    action[1] = float(pad.get_axis(1))
    if pad.get_numaxes() > 3:
        action[2] = float(pad.get_axis(3))
    closed = pad.get_numbuttons() > 0 and bool(pad.get_button(0))
    action[3] = -1.0 if closed else 1.0
    return np.clip(action, schema.ACTION_LOW, schema.ACTION_HIGH).astype(np.float32)
