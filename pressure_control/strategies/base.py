from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Type

import pandas as pd

from ..config import ControlConfig
from ..models import ContextFeatures, ControlContext, ControlDecision


class ControlStrategy(ABC):
    """Interface for pluggable control strategies."""

    name: str = "base"
    description: str = ""
    respects_cooldown: bool = True

    def __init__(self, config: ControlConfig):
        self.config = config

    @abstractmethod
    def decide(self, context: ControlContext) -> ControlDecision | None:
        """Return a decision (delta + reason) or None to hold position."""

    @abstractmethod
    def build_features(self, window: pd.DataFrame) -> ContextFeatures:
        """Compute context features from the decision window."""


STRATEGY_REGISTRY: Dict[str, Type[ControlStrategy]] = {}


def register_strategy(strategy_cls: Type[ControlStrategy]) -> None:
    STRATEGY_REGISTRY[strategy_cls.name] = strategy_cls


def available_strategies() -> List[str]:
    return sorted(STRATEGY_REGISTRY)


def build_strategy(name: str, config: ControlConfig) -> ControlStrategy:
    try:
        strategy_cls = STRATEGY_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {available_strategies()}"
        ) from exc
    return strategy_cls(config)
