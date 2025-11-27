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

`control_strategy.py` exposes a strategy interface that receives a normalized
control context (windowed data, computed trend features, the current valve
command, and the previous actions list) and returns the next delta to apply.
The orchestration layer (`ControlEngine`) handles sampling cadence, cooldowns,
and applying bounds. Strategies are registered in `STRATEGY_REGISTRY` and
selected via CLI arguments.

### Built-in strategy: `rule_based`

The default strategy follows the operating notes:

- Evaluates every 5 seconds using a 30-second trend window.
- Targets a pressure center of -80 and keeps the rolling center between -85 and -75.
- Waits 20 seconds after each adjustment before issuing another command.
- Avoids adjustments while the cover is open, except for emergency limits
  (pressure > -50 or < -150).
- Adjustment steps are clamped between 0.2 and 1.0 units and respond to
  both boundary violations and approaching-trend conditions.

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
