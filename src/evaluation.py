from __future__ import annotations
import numpy as np
import pandas as pd

def evaluate_plan_on_prices(plan, prices_1d):
    prices_1d = np.asarray(prices_1d, dtype=float)
    net = plan["discharge"].to_numpy() - plan["charge"].to_numpy()
    return float(np.sum(prices_1d * net))

def summarize_plan(plan, scenarios, name):
    revs = np.array([evaluate_plan_on_prices(plan, scenarios[s]) for s in range(scenarios.shape[0])])
    return {
        "plan": name,
        "mean_revenue": float(revs.mean()),
        "p10_revenue": float(np.percentile(revs, 10)),
        "worst_revenue": float(revs.min()),
        "std_revenue": float(revs.std(ddof=1)),
    }

def efficient_frontier_from_summary(summary_df):
    """Expect summary_df with columns plan, mean_revenue, worst_revenue and lambda labels in plan."""
    out = summary_df.loc[:, ["plan", "mean_revenue", "worst_revenue"]].copy()
    # try to parse lambda from plan strings like 'cvar_lam_1.0'
    def parse_lambda(s):
        if s == "deterministic_mean" or s == "deterministic":
            return "deterministic"
        if "lam" in s:
            return s.split("lam_")[-1]
        return s
    out["lambda"] = out["plan"].map(parse_lambda)
    return out
