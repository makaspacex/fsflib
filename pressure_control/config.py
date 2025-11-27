from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ControlConfig:
    """Tunable control parameters shared by all strategies."""

    target: float = -80.0
    acceptable_min: float = -85.0
    acceptable_max: float = -75.0
    check_seconds: int = 5
    trend_window_seconds: int = 30
    cooldown_seconds: int = 20
    min_step: float = 0.2
    max_step: float = 1.0
    emergency_high: float = -50.0  # near atmospheric pressure
    emergency_low: float = -150.0  # deep vacuum limit
    preemptive_margin: float = 1.5  # how close to edge before preemptive adjust
    valve_min: float = 0.0
    valve_max: float = 100.0
