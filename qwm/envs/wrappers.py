"""Thin environment wrappers (no UI, no dashboards; pure bookkeeping).

SeedWrapper: forces reset(seed=...) to always pass an explicit seed and
records the last seed (fail-loud discipline against accidental unseeded
resets).

MonitorWrapper: accumulates per-episode stats (return, length, success,
per-step rewards) used by the collector for dataset statistics.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class SeedWrapper:
    """Ensure every reset is explicitly seeded (or reuses a recorded seed)."""

    def __init__(self, env, default_seed: int = 0):
        self.env = env
        self.default_seed = int(default_seed)
        self.last_seed: Optional[int] = None

    def reset(self, seed: Optional[int] = None, **kwargs) -> Dict:
        if seed is None:
            seed = self.last_seed if self.last_seed is not None else self.default_seed
        seed = int(seed)
        self.last_seed = seed
        return self.env.reset(seed=seed, **kwargs)

    def step(self, action):
        return self.env.step(action)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


class MonitorWrapper:
    """Track per-episode return/length/success for collection statistics."""

    def __init__(self, env):
        self.env = env
        self.episode_return = 0.0
        self.episode_length = 0
        self.success = False
        self.episode_rewards: list = []

    def reset(self, **kwargs) -> Dict:
        self.episode_return = 0.0
        self.episode_length = 0
        self.success = False
        self.episode_rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.episode_return += float(reward)
        self.episode_length += 1
        self.episode_rewards.append(float(reward))
        self.success = bool(info.get("success", self.success))
        return obs, reward, done, info

    def episode_stats(self) -> Dict:
        return {
            "return": float(self.episode_return),
            "length": int(self.episode_length),
            "success": bool(self.success),
            "mean_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)
