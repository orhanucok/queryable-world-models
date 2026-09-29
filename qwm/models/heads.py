"""Explicit query-state heads. Binary slices are logits, not probabilities.

The packed vector uses the frozen slices in qwm.data.schema. Those indices
are not redefined here.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from qwm.data import schema


class QueryHeads(nn.Module):
    """Map a recurrent state to the 123-dim query vector."""

    def __init__(self, hidden: int):
        super().__init__()
        hidden = int(hidden)
        self.grip = nn.Linear(hidden, 3)
        self.obj = nn.Linear(hidden, schema.NMAX * 3)
        self.upright = nn.Linear(hidden, schema.NMAX)
        self.contact = nn.Linear(hidden, schema.NMAX)
        self.on_table = nn.Linear(hidden, schema.NMAX)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim != 2:
            raise ValueError(f"query heads expect [B, H], got {tuple(state.shape)}")
        out = state.new_zeros(state.shape[0], schema.QUERY_DIM)
        out[:, schema.GRIP_XYZ] = self.grip(state)
        out[:, schema.OBJ_XYZ] = self.obj(state)
        out[:, schema.OBJ_UPRIGHT] = self.upright(state)
        out[:, schema.OBJ_CONTACT] = self.contact(state)
        out[:, schema.OBJ_ON_TABLE] = self.on_table(state)
        return out
