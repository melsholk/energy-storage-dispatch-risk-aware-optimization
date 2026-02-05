# Risk-Aware Optimization of Grid-Scale Battery Dispatch

This project models grid-scale battery dispatch under uncertain electricity prices using scenario-based stochastic optimization and CVaR risk aversion.

## Problem

Grid-scale battery operators must decide how to charge and discharge energy storage assets under highly volatile electricity prices. 
While deterministic optimization maximizes expected arbitrage revenue, it can expose operators to severe downside risk when prices deviate from forecasts.

This project studies how risk-aware optimization changes battery dispatch decisions and quantifies the tradeoff between expected value and downside protection.

## Modeling Approach

- Hourly real-time electricity prices from the PJM wholesale market
- **Linear optimization** model with intertemporal state-of-charge constraints
- **Scenario-based stochastic** programming using historical price samples
- Conditional Value-at-Risk (**CVaR**) to model downside risk aversion
- Implemented in Python using **Pyomo** and the **HiGHS** solver


## Assummed Battery base case
- Power rating: **50 MW**
- Energy capacity: **200 MWh** (4-hour duration)
- Round-trip efficiency: **90%**
- Initial SOC = terminal SOC = **50%** (100 MWh)

## Data
Export PJM **Data Miner 2 → Real-Time Hourly LMPs** for:
- `Pricing Node Name`: **WESTERN HUB**
- Hourly timestamps (EPT)

## Key Result: Efficient Frontier Between Value and Risk

![Efficient Frontier](figures/efficient_frontier.png)

The figure shows the efficient frontier between expected arbitrage revenue and worst-case performance.
Deterministic optimization achieves the highest expected value but exposes the system to large downside losses.
Introducing CVaR risk aversion dramatically improves worst-case outcomes with only modest reductions in mean revenue.
Beyond moderate risk aversion, additional downside protection exhibits diminishing returns.

| Policy | Mean Revenue | Worst Case | Std Dev |
|------|-------------|------------|---------|
| Deterministic | High | Very Negative | High |
| CVaR (λ = 0.1) | Slightly Lower | Positive | Lower |
| CVaR (λ = 1.0) | Moderate | Strongly Positive | Lower |

## Takeaways

- Deterministic optimization implicitly selects a high-risk operating policy
- Expected-value stochastic optimization alone does not change dispatch decisions in linear models
- CVaR risk aversion enables explicit control of downside risk
- Most downside protection is achieved with moderate risk aversion

## Repository Structure

- notebooks/01_data_and_deterministic_dispatch.ipynb  
- notebooks/02_scenarios_and_cvar_dispatch.ipynb  
- src/ – reusable optimization and evaluation code  

## How to Run

1. Install dependencies: `pip install -r requirements.txt`
2. Place PJM price CSV in `data/raw/`
3. Run notebooks in order
