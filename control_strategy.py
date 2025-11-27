"""Auto control simulation for pre-storage pressure using exhaust valve.

The module now offers a **pluggable control architecture**: you can pick
different strategy implementations via CLI parameters while reusing the
same controller runtime. A strategy receives a normalized context (windowed
data, computed trends, current valve command, and configuration) and
returns the next adjustment suggestion. The default strategy is the
rule-based controller designed from the provided behavioral notes:

* Sample pressure every 5 seconds.
* Target center pressure is -80 with an acceptable band of [-85, -75].
* Avoid adjustments while the cover is open (logix._GXJ_C071 == 0) unless
  an extreme pressure safety limit is reached.
* Include a 20-second cool-down after each valve change to wait for the
  process response.
* Each valve change is clamped between 0.2 and 1.0 units.

Run ``python control_strategy.py data.csv`` to see recommended adjustments,
or ``python control_strategy.py data.csv --strategy rule_based`` to
explicitly select a strategy.
"""
from __future__ import annotations

import argparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class ControlConfig:
    """Container for all tunable control parameters."""

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


class ControlStrategy(ABC):
    """Interface for pluggable control strategies."""

    name: str = "base"
    description: str = ""
    respects_cooldown: bool = True

    def __init__(self, config: ControlConfig):
        self.config = config

    @abstractmethod
    def decide(self, context: ControlContext) -> Optional[ControlDecision]:
        """Return a decision (delta + reason) or None to hold position."""
        raise NotImplementedError

    @abstractmethod
    def build_features(self, window: pd.DataFrame) -> ContextFeatures:
        """Compute context features from the decision window."""
        raise NotImplementedError


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

    def decide(self, context: ControlContext) -> Optional[ControlDecision]:
        cfg = self.config
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


STRATEGY_REGISTRY: Dict[str, type[ControlStrategy]] = {}


def register_strategy(strategy_cls: type[ControlStrategy]) -> None:
    STRATEGY_REGISTRY[strategy_cls.name] = strategy_cls


register_strategy(RuleBasedStrategy)


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

    def _build_context(self, df: pd.DataFrame, now: pd.Timestamp) -> Optional[ControlContext]:
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

    def _apply_delta(self, delta: float) -> Optional[ControlAction]:
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


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def available_strategies() -> List[str]:
    return sorted(STRATEGY_REGISTRY)


def build_strategy(name: str, config: ControlConfig) -> ControlStrategy:
    try:
        strategy_cls = STRATEGY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown strategy '{name}'. Available: {available_strategies()}") from exc
    return strategy_cls(config)


def run_controller(
    path: str, output: Optional[str], strategy_name: str = RuleBasedStrategy.name
) -> List[ControlAction]:
    data = load_data(path)
    config = ControlConfig()
    strategy = build_strategy(strategy_name, config)
    engine = ControlEngine(config, strategy)
    actions = engine.run(data)

    if output:
        pd.DataFrame(
            {
                "time": [a.time for a in actions],
                "valve_command": [a.valve_command for a in actions],
                "delta": [a.delta for a in actions],
                "reason": [a.reason for a in actions],
            }
        ).to_csv(output, index=False)

    return actions


def format_action(action: ControlAction) -> str:
    direction = "increase" if action.delta > 0 else "decrease"
    return (
        f"{action.time.isoformat()} | {direction} valve by {abs(action.delta):.2f} "
        f"-> command {action.valve_command:.2f} | {action.reason}"
    )


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Path to data.csv with pressure and valve signals")
    parser.add_argument(
        "--output",
        help="Optional path to save recommended valve actions as CSV",
    )
    parser.add_argument(
        "--strategy",
        default=RuleBasedStrategy.name,
        choices=available_strategies(),
        help="Control strategy to execute",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List available strategies and exit",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list_strategies:
        print("Available strategies:")
        for name in available_strategies():
            print(f"- {name}: {STRATEGY_REGISTRY[name].description}")
        return

    actions = run_controller(args.data, args.output, args.strategy)
    if not actions:
        print("No adjustments required for provided data.")
        return

    for action in actions:
        print(format_action(action))


if __name__ == "__main__":
    main()
