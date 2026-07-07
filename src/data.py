# Mortgage loan data with competing risks: default vs prepayment
import numpy as np
import pandas as pd

def make_synthetic(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    ltv = rng.uniform(60, 120, n)
    fico = rng.integers(500, 800, n)
    rate = rng.uniform(3, 12, n)
    balance = rng.lognormal(10, 1, n)
    # Default hazard
    h_def = np.exp(-8 + 0.02 * ltv - 0.03 * fico + 0.1 * rate + 0.1 * np.log(balance)) / 365
    # Prepay hazard
    h_pre = np.exp(-6 - 0.01 * ltv + 0.01 * fico - 0.05 * rate + 0.05 * np.log(balance)) / 365
    t_def = rng.exponential(1 / h_def).clip(30, 365 * 30)
    t_pre = rng.exponential(1 / h_pre).clip(30, 365 * 30)
    event = np.where(rng.random(n) < 0.5, 0, np.where(t_def < t_pre, 1, 2))
    obs = np.where(event == 0, np.minimum(t_def, t_pre),
                   np.where(event == 1, t_def, t_pre))
    df = pd.DataFrame({
        "ltv": ltv, "fico": fico, "rate": rate, "balance": balance,
        "time": obs, "event": event,
    })
    return {
        "df": df,
        "numerical_features": ["ltv", "fico", "rate", "balance"],
        "time_col": "time", "event_col": "event",
        "n_samples": n,
    }