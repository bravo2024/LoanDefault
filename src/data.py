# Mortgage loan data with competing risks: default vs prepayment
import numpy as np
import pandas as pd

def make_synthetic(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    ltv = rng.uniform(60, 120, n)
    fico = rng.integers(500, 800, n)
    rate = rng.uniform(3, 12, n)
    balance = rng.lognormal(10, 1, n)
    # Cause-specific hazards, centered so both causes actually occur:
    # high LTV / low FICO / high rate push default; the reverse pushes prepay.
    h_def = np.exp(-1.5 + 0.03 * (ltv - 80) - 0.01 * (fico - 650)
                   + 0.15 * (rate - 6) + 0.2 * (np.log(balance) - 10)) / 365
    h_pre = np.exp(-1.0 - 0.01 * (ltv - 80) + 0.005 * (fico - 650)
                   - 0.10 * (rate - 6)) / 365
    t_def = rng.exponential(1 / h_def)
    t_pre = rng.exponential(1 / h_pre)
    censor = rng.uniform(180, 365 * 10, n)  # observation window per loan
    obs = np.minimum(np.minimum(t_def, t_pre), censor).clip(30)
    event = np.where(censor <= np.minimum(t_def, t_pre), 0,
                     np.where(t_def < t_pre, 1, 2))
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