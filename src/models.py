from __future__ import annotations
import numpy as np
import pandas as pd
import pyomo.environ as pyo

def build_price_scenarios(df, horizon = 24, K = 20, seed = 42):
    """Sample K historical daily price paths of length=horizon.

    df must have columns: timestamp (datetime), price (float).
    Returns: (scenarios, scenario_days)
      scenarios: np.ndarray shape (K, horizon)
      scenario_days: list[date]
    """
    tmp = df.copy()
    tmp["date"] = tmp["timestamp"].dt.date

    day_groups = tmp.groupby("date")
    valid_days = [d for d, g in day_groups if len(g) == horizon]

    if len(valid_days) < K:
        raise ValueError(f"Not enough complete days ({len(valid_days)}) to sample K={K} scenarios.")

    rng = np.random.default_rng(seed)
    sampled_days = rng.choice(valid_days, size=K, replace=False)

    scenarios = []
    for d in sampled_days:
        g = tmp[tmp["date"] == d].sort_values("timestamp")
        scenarios.append(g["price"].to_numpy())

    return np.asarray(scenarios), list(sampled_days)

def solve_deterministic_dispatch(
    prices_1d,
    P_MAX = 50.0,
    E_MAX = 200.0,
    SOC_INIT = 100.0,
    eta_rt = 0.90,
    DT = 1.0,
    solver_name = "highs",
):
    """Solve deterministic battery dispatch LP for a single price path."""
    prices_1d = np.asarray(prices_1d, dtype=float)
    T = len(prices_1d)
    ETA_C = float(np.sqrt(eta_rt))
    ETA_D = float(np.sqrt(eta_rt))

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T - 1)

    m.price = pyo.Param(m.T, initialize={t: float(prices_1d[t]) for t in range(T)})

    m.charge = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.discharge = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.soc = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    m.obj = pyo.Objective(
        expr=sum(m.price[t] * (m.discharge[t] - m.charge[t]) for t in m.T),
        sense=pyo.maximize,
    )

    def soc_balance(mm, t):
        if t == 0:
            return mm.soc[t] == SOC_INIT + ETA_C * mm.charge[t] * DT - (1 / ETA_D) * mm.discharge[t] * DT
        return mm.soc[t] == mm.soc[t - 1] + ETA_C * mm.charge[t] * DT - (1 / ETA_D) * mm.discharge[t] * DT

    m.soc_balance = pyo.Constraint(m.T, rule=soc_balance)
    m.charge_limit = pyo.Constraint(m.T, rule=lambda mm, t: mm.charge[t] <= P_MAX)
    m.discharge_limit = pyo.Constraint(m.T, rule=lambda mm, t: mm.discharge[t] <= P_MAX)
    m.soc_limit = pyo.Constraint(m.T, rule=lambda mm, t: mm.soc[t] <= E_MAX)
    m.terminal_soc = pyo.Constraint(expr=m.soc[T - 1] == SOC_INIT)

    solver = pyo.SolverFactory(solver_name)
    res = solver.solve(m, tee=False)
    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Solver did not find optimal solution: {res.solver.termination_condition}")

    plan = pd.DataFrame(
        {
            "hour": np.arange(T),
            "charge": [pyo.value(m.charge[t]) for t in m.T],
            "discharge": [pyo.value(m.discharge[t]) for t in m.T],
            "soc": [pyo.value(m.soc[t]) for t in m.T],
        }
    )
    return plan

def solve_expected_value_dispatch(price_scenarios, **kwargs):
    """Solve expected-value dispatch with a single here-and-now plan shared across scenarios."""
    price_scenarios = np.asarray(price_scenarios, dtype=float)
    mean_prices = price_scenarios.mean(axis=0)
    return solve_deterministic_dispatch(mean_prices, **kwargs)

def solve_cvar_dispatch(
    price_scenarios,
    alpha = 0.10,
    lam = 1.0,
    P_MAX = 50.0,
    E_MAX = 200.0,
    SOC_INIT = 100.0,
    eta_rt = 0.90,
    DT = 1.0,
    solver_name = "highs",
):
    """Risk-aware battery dispatch: maximize E[rev] - lam * CVaR_alpha(loss), loss=-rev.

    Decisions are a single here-and-now plan shared across scenarios.
    Returns: (plan_df, stats_dict)
    """
    price_scenarios = np.asarray(price_scenarios, dtype=float)
    K, T = price_scenarios.shape
    ETA_C = float(np.sqrt(eta_rt))
    ETA_D = float(np.sqrt(eta_rt))

    m = pyo.ConcreteModel()
    m.T = pyo.RangeSet(0, T - 1)
    m.S = pyo.RangeSet(0, K - 1)

    m.price = pyo.Param(
        m.S, m.T,
        initialize={(s, t): float(price_scenarios[s, t]) for s in range(K) for t in range(T)}
    )

    m.charge = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.discharge = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.soc = pyo.Var(m.T, domain=pyo.NonNegativeReals)

    def revenue_s(mm, s):
        return sum(mm.price[s, t] * (mm.discharge[t] - mm.charge[t]) for t in mm.T)
    m.rev = pyo.Expression(m.S, rule=revenue_s)

    m.zeta = pyo.Var(domain=pyo.Reals)
    m.u = pyo.Var(m.S, domain=pyo.NonNegativeReals)

    def cvar_slack(mm, s):
        return mm.u[s] >= (-mm.rev[s]) - mm.zeta
    m.cvar_slack = pyo.Constraint(m.S, rule=cvar_slack)

    m.cvar_loss = pyo.Expression(expr=m.zeta + (1.0 / (alpha * K)) * sum(m.u[s] for s in m.S))
    m.exp_rev = pyo.Expression(expr=(1.0 / K) * sum(m.rev[s] for s in m.S))

    m.obj = pyo.Objective(expr=m.exp_rev - lam * m.cvar_loss, sense=pyo.maximize)

    def soc_balance(mm, t):
        if t == 0:
            return mm.soc[t] == SOC_INIT + ETA_C * mm.charge[t] * DT - (1 / ETA_D) * mm.discharge[t] * DT
        return mm.soc[t] == mm.soc[t - 1] + ETA_C * mm.charge[t] * DT - (1 / ETA_D) * mm.discharge[t] * DT

    m.soc_balance = pyo.Constraint(m.T, rule=soc_balance)
    m.charge_limit = pyo.Constraint(m.T, rule=lambda mm, t: mm.charge[t] <= P_MAX)
    m.discharge_limit = pyo.Constraint(m.T, rule=lambda mm, t: mm.discharge[t] <= P_MAX)
    m.soc_limit = pyo.Constraint(m.T, rule=lambda mm, t: mm.soc[t] <= E_MAX)
    m.terminal_soc = pyo.Constraint(expr=m.soc[T - 1] == SOC_INIT)

    solver = pyo.SolverFactory(solver_name)
    res = solver.solve(m, tee=False)
    if res.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise RuntimeError(f"Solver did not find optimal solution: {res.solver.termination_condition}")

    plan = pd.DataFrame(
        {
            "hour": np.arange(T),
            "charge": [pyo.value(m.charge[t]) for t in m.T],
            "discharge": [pyo.value(m.discharge[t]) for t in m.T],
            "soc": [pyo.value(m.soc[t]) for t in m.T],
        }
    )
    stats = {
        "expected_revenue": float(pyo.value(m.exp_rev)),
        "cvar_loss": float(pyo.value(m.cvar_loss)),
        "zeta": float(pyo.value(m.zeta)),
        "alpha": float(alpha),
        "lambda": float(lam),
    }
    return plan, stats
