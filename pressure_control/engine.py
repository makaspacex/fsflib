from __future__ import annotations

from typing import List, Optional

import pandas as pd

from .config import ControlConfig
from .models import ControlAction, ControlContext
from .strategies import ControlStrategy


class ControlEngine:
    """Core runtime that feeds context into the chosen strategy."""

    def __init__(self, config: ControlConfig, strategy: ControlStrategy):
        self.config = config
        self.strategy = strategy
        self._last_action_time: Optional[pd.Timestamp] = None
        self._valve_command: float | None = None
        self.actions: List[ControlAction] = []

    def _prepare(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df["_time"] = pd.to_datetime(df["_time"])
        df = df.sort_values("_time").set_index("_time")
        df[["logix.GXJ_B0045", "logix._GXJ_C071", "logix.GXJ_A058"]] = df[
            ["logix.GXJ_B0045", "logix._GXJ_C071", "logix.GXJ_A058"]
        ].ffill().bfill()
        return df

    def _needs_cooldown(self, now: pd.Timestamp) -> bool:
        if self._last_action_time is None:
            return False
        return (now - self._last_action_time).total_seconds() < self.config.cooldown_seconds

    def _build_context(self, df: pd.DataFrame, now: pd.Timestamp) -> ControlContext | None:
        cfg = self.config
        window_start = now - pd.Timedelta(seconds=cfg.trend_window_seconds)
        window = df.loc[window_start:now]
        if window.empty:
            return None

        features = self.strategy.build_features(window)

        return ControlContext(
            now=now,
            window=window,
            config=cfg,
            last_action_time=self._last_action_time,
            valve_command=float(self._valve_command),
            actions=self.actions,
            features=features,
            in_cooldown=self._needs_cooldown(now),
        )

    def _normalize_delta(self, delta: float) -> float:
        if delta == 0:
            return 0.0
        sign = 1 if delta > 0 else -1
        magnitude = min(self.config.max_step, max(self.config.min_step, abs(delta)))
        return sign * magnitude

    def _apply_delta(self, delta: float) -> ControlAction | None:
        cfg = self.config
        normalized_delta = self._normalize_delta(delta)
        current_command = float(self._valve_command or 0.0)
        proposed = current_command + normalized_delta
        clamped_command = max(cfg.valve_min, min(cfg.valve_max, proposed))
        actual_delta = clamped_command - current_command

        if abs(actual_delta) < 1e-9:
            return None

        return ControlAction(time=pd.NaT, valve_command=clamped_command, delta=actual_delta, reason="")

    def run(self, data: pd.DataFrame) -> List[ControlAction]:
        df = self._prepare(data)
        if df.empty:
            return []

        self._valve_command = float(df.iloc[0]["logix.GXJ_B0045"])
        cfg = self.config
        decision_times = pd.date_range(df.index[0], df.index[-1], freq=f"{cfg.check_seconds}s")

        for current_time in decision_times:
            context = self._build_context(df, current_time)
            if context is None:
                continue

            if self.strategy.respects_cooldown and context.in_cooldown:
                continue

            decision = self.strategy.decide(context)
            if decision is None:
                continue

            action = self._apply_delta(decision.delta)
            if action is None:
                continue

            action.time = current_time
            action.reason = decision.reason
            self._valve_command = action.valve_command
            self._last_action_time = current_time
            self.actions.append(action)

        return self.actions
