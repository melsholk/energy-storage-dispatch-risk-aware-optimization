from __future__ import annotations
import pandas as pd

def load_pjm_prices_csv(path):
    """Load PJM Data Miner 2 Real-Time Hourly LMPs export.

    Expected columns include:
      - datetime_beginning_ept
      - total_lmp_rt

    Returns a tidy dataframe with columns:
      - timestamp (datetime)
      - price ($/MWh, float)
    """
    df = pd.read_csv(path)

    required = {"datetime_beginning_ept", "total_lmp_rt"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = (
        df.loc[:, ["datetime_beginning_ept", "total_lmp_rt"]]
        .rename(columns={"datetime_beginning_ept": "timestamp", "total_lmp_rt": "price"})
    )

    out["timestamp"] = pd.to_datetime(out["timestamp"])
    out = out.sort_values("timestamp").reset_index(drop=True)

    # basic sanity checks
    if out["price"].isna().any():
        raise ValueError("Found NaN prices after loading.")

    # check hourly spacing (allow daylight savings irregularities by checking most common diff)
    diffs = out["timestamp"].diff().dropna()
    if len(diffs) > 0:
        mode = diffs.mode().iloc[0]
        if mode != pd.Timedelta(hours=1):
            raise ValueError(f"Expected hourly spacing; most common diff is {mode}.")

    return out
