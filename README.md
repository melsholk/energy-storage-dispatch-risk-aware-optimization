# Grid-Scale Battery Dispatch Under Price Uncertainty (PJM) — Scenario Optimization + CVaR

This repo demonstrates a workflow for **grid-scale battery dispatch** using **real PJM real-time hourly LMPs** (Western Hub).  
It starts with a deterministic linear program (LP) baseline and then adds **scenario-based uncertainty** and **risk aversion** via **CVaR** to produce an explicit **risk–reward efficient frontier**.

## What this project shows
- Mathematical modeling of intertemporal energy storage constraints (SOC dynamics, power/energy limits, efficiency)
- Scenario analysis using historical price paths
- Risk-aware optimization (CVaR) implemented as a **linear program**
- Practical Python implementation with **Pyomo + HiGHS**
- Clear decision insights (expected value vs downside protection)

## Battery base case
- Power rating: **50 MW**
- Energy capacity: **200 MWh** (4-hour duration)
- Round-trip efficiency: **90%**
- Initial SOC = terminal SOC = **50%** (100 MWh)

## Data
Export PJM **Data Miner 2 → Real-Time Hourly LMPs** for:
- `Pricing Node Name`: **WESTERN HUB**
- Hourly timestamps (EPT)

Save the CSV to:
`data/raw/pjm_rt_hourly_western_hub_2025Q1.csv`

The notebooks expect at least these columns:
- `datetime_beginning_ept`
- `total_lmp_rt`

## Quickstart

```bash
pip install -r requirements.txt
```

Run notebooks in order:
1. `notebooks/01_data_and_deterministic_dispatch.ipynb`
2. `notebooks/02_scenarios_and_cvar_dispatch.ipynb`

## Notebooks
- **01 — Data + deterministic dispatch**: validates the LP and produces interpretable dispatch/SOC plots.
- **02 — Scenarios + CVaR**: builds **K=20** historical scenarios and solves EV vs **CVaR** plans; plots the efficient frontier.

## Key takeaway
Deterministic dispatch can maximize expected revenue while quietly accepting substantial tail risk.  
CVaR risk aversion provides a tunable knob (λ) to trade a small reduction in expected value for large improvements in worst-case outcomes.

## Repo layout
- `src/` reusable model/data utilities
- `notebooks/` end-to-end analysis and plots
- `data/` (input data; not tracked)
- `results/`, `figures/` (optional outputs)
