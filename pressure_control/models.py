from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .config import ControlConfig


@dataclass
class ControlAction:
    """Represents a single valve adjustment suggestion."""

    time: pd.Timestamp
    valve_command: float
    delta: float
    reason: str


@dataclass
class ContextFeatures:
    """Pre-computed metrics for a decision window."""

    center: float
    slope: float
    current_pressure: float
    cover_state: int


@dataclass
class ControlContext:
    """Context passed to a strategy for a single decision point."""

    now: pd.Timestamp
    window: pd.DataFrame
    config: ControlConfig
    last_action_time: Optional[pd.Timestamp]
    valve_command: float
    actions: List[ControlAction]
    features: ContextFeatures
    in_cooldown: bool


@dataclass
class ControlDecision:
    """Strategy output describing how to adjust the valve."""

    delta: float
    reason: str
