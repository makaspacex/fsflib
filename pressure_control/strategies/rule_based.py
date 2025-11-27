from __future__ import annotations

import pandas as pd

from ..config import ControlConfig
from ..models import ContextFeatures, ControlContext, ControlDecision
from .base import ControlStrategy, register_strategy


class RuleBasedStrategy(ControlStrategy):
    """Trend-aware rule-based strategy derived from operating guidance."""

    name = "rule_based"
    description = "Heuristic controls with hysteresis, trend checks, and emergencies"
    respects_cooldown = False

    def _band_center(self, window: pd.DataFrame) -> float:
        return window["logix.GXJ_A058"].mean()

    def _trend(self, window: pd.DataFrame) -> float:
        halfway = int(len(window) / 2) or 1
        first = window.iloc[:halfway]["logix.GXJ_A058"].mean()
        last = window.iloc[halfway:]["logix.GXJ_A058"].mean()
        duration = max((window.index[-1] - window.index[0]).total_seconds(), 1)
        return (last - first) / duration

    def _compute_step(self, deviation: float) -> float:
        step = max(self.config.min_step, min(self.config.max_step, abs(deviation) * 0.02))
        return step

    def build_features(self, window: pd.DataFrame) -> ContextFeatures:
        center = self._band_center(window)
        slope = self._trend(window)
        current_pressure = float(window.iloc[-1]["logix.GXJ_A058"])
        cover_state = int(window.iloc[-1]["logix._GXJ_C071"])
        return ContextFeatures(center=center, slope=slope, current_pressure=current_pressure, cover_state=cover_state)

    def decide(self, context: ControlContext) -> ControlDecision | None:
        cfg: ControlConfig = self.config
        f = context.features

        # Emergency handling during open cover
        if f.cover_state == 0:
            if f.current_pressure > cfg.emergency_high:
                delta = -min(cfg.max_step, cfg.min_step * 3)
                reason = "Emergency close during open cover (pressure too high)"
                return ControlDecision(delta=delta, reason=reason)
            if f.current_pressure < cfg.emergency_low:
                delta = min(cfg.max_step, cfg.min_step * 3)
                reason = "Emergency open during open cover (pressure too low)"
                return ControlDecision(delta=delta, reason=reason)
            return None

        if context.in_cooldown:
            return None

        # Preemptive adjustment when approaching boundary with wrong trend
        approaching_upper = f.center > (cfg.acceptable_max - cfg.preemptive_margin) and f.slope > 0
        approaching_lower = f.center < (cfg.acceptable_min + cfg.preemptive_margin) and f.slope < 0

        out_of_bounds = f.center > cfg.acceptable_max or f.center < cfg.acceptable_min

        if not (out_of_bounds or approaching_upper or approaching_lower):
            return None

        deviation = f.center - cfg.target
        step = self._compute_step(deviation)

        if f.center > cfg.acceptable_max or approaching_upper:
            delta = -step
            reason = "Close valve to reduce pressure"
        else:
            delta = step
            reason = "Open valve to relieve pressure"

        return ControlDecision(delta=delta, reason=reason)


register_strategy(RuleBasedStrategy)
