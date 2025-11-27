from __future__ import annotations

import argparse
from typing import Iterable, List, Optional

from .config import ControlConfig
from .data import actions_to_frame, format_action, load_data
from .engine import ControlEngine
from .models import ControlAction
from .strategies import DEFAULT_STRATEGY, available_strategies, build_strategy


def run_controller(
    path: str,
    output: str | None,
    strategy_name: str = DEFAULT_STRATEGY,
    config: ControlConfig | None = None,
) -> List[ControlAction]:
    data = load_data(path)
    config = config or ControlConfig()
    strategy = build_strategy(strategy_name, config)
    engine = ControlEngine(config, strategy)
    actions = engine.run(data)

    if output:
        actions_to_frame(actions).to_csv(output, index=False)

    return actions


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Auto control simulation for pre-storage pressure using exhaust valve.")
    parser.add_argument("data", help="Path to data.csv with pressure and valve signals")
    parser.add_argument(
        "--output",
        help="Optional path to save recommended valve actions as CSV",
    )
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        choices=available_strategies(),
        help="Control strategy to execute",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="List available strategies and exit",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list_strategies:
        print("Available strategies:")
        for name in available_strategies():
            from .strategies import base

            print(f"- {name}: {base.STRATEGY_REGISTRY[name].description}")
        return

    actions = run_controller(args.data, args.output, args.strategy)
    if not actions:
        print("No adjustments required for provided data.")
        return

    for action in actions:
        print(format_action(action))


__all__ = ["main", "run_controller", "build_arg_parser"]
