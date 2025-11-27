# fsflib

This repository contains sample furnace data (`data.csv`) and a pluggable
Python controller runtime that simulates automatic valve adjustments to keep
the pre-storage pressure near a target center of **-80** with a ±5 tolerance
band. Strategies are selected at runtime so you can swap heuristics without
changing the orchestration layer.

## Dataset

`data.csv` columns:

- `_time`: timestamp (UTC)
- `logix.GXJ_A058`: pre-storage pressure
- `logix.GXJ_B0045`: exhaust-valve opening
- `logix._GXJ_C071`: cover state (1 = closed, 0 = open)

## Controller architecture

The controller is packaged under `pressure_control/` so multiple strategies and
runtimes can evolve independently.

- `config.py`: shared tunable parameters for all strategies.
- `models.py`: dataclasses for actions, contexts, and strategy outputs.
- `engine.py`: the orchestration core that manages cadence, cooldown, clamping,
  and context construction.
- `data.py`: helpers for loading datasets and formatting actions.
- `strategies/`: pluggable strategy implementations. New strategies register
  themselves via `register_strategy`.
- `cli.py`: user-facing CLI helpers and the `run_controller` API.

`control_strategy.py` remains a tiny entrypoint that simply calls the CLI; it
can be used directly or imported by other scripts to embed the engine.

### Built-in strategy: `rule_based`

The default strategy follows the operating notes:

- Evaluates every 5 seconds using a 30-second trend window.
- Targets a pressure center of -80 and keeps the rolling center between -85 and
  -75.
- Waits 20 seconds after each adjustment before issuing another command.
- Avoids adjustments while the cover is open, except for emergency limits
  (pressure > -50 or < -150).
- Adjustment steps are clamped between 0.2 and 1.0 units and respond to both
  boundary violations and approaching-trend conditions.

## Extending strategies

1. Create a new module under `pressure_control/strategies/` and subclass
   `ControlStrategy`.
2. Implement `build_features` (to compute windowed metrics) and `decide` (to
   return a `ControlDecision` or `None`).
3. Call `register_strategy(NewStrategy)` at the bottom of the module. The CLI
   will automatically expose it via `--strategy` and `--list-strategies`.

## Usage

```bash
pip install -r requirements.txt
python control_strategy.py data.csv --output recommended_actions.csv

# List and select strategies
python control_strategy.py data.csv --list-strategies
python control_strategy.py data.csv --strategy rule_based
```

The script prints recommended valve adjustments and optionally writes them
to `recommended_actions.csv` for downstream analysis.
