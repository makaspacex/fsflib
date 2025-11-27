from __future__ import annotations

from typing import Iterable, List

import pandas as pd

from .models import ControlAction


def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def actions_to_frame(actions: Iterable[ControlAction]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [a.time for a in actions],
            "valve_command": [a.valve_command for a in actions],
            "delta": [a.delta for a in actions],
            "reason": [a.reason for a in actions],
        }
    )


def format_action(action: ControlAction) -> str:
    direction = "increase" if action.delta > 0 else "decrease"
    return (
        f"{action.time.isoformat()} | {direction} valve by {abs(action.delta):.2f} "
        f"-> command {action.valve_command:.2f} | {action.reason}"
    )


def format_actions(actions: List[ControlAction]) -> str:
    return "\n".join(format_action(action) for action in actions)
