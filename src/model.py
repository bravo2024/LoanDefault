# Mortgage loan survival model: Cox PH for default risk with competing risks
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from src.core import concordance_index

def cox_neg_loglik(beta, X, t, e):
    risk = X @ np.asarray(beta, dtype=float)
    ll = 0.0
    for i in range(len(t)):
        if e[i] != 1:
            continue
        risk_set = t >= t[i]
        denom = np.log(np.sum(np.exp(risk[risk_set])))
        ll += risk[i] - denom
    return -ll

def fit_and_evaluate(data, seed=42):
    df = data["df"]
    feat = data["numerical_features"]
    X = df[feat].values.astype(float)
    t = df[data["time_col"]].values
    e = df[data["event_col"]].values
    X_tr, X_te, t_tr, t_te, e_tr, e_te = train_test_split(
        X, t, e, test_size=0.25, random_state=seed
    )
    res = minimize(
        cox_neg_loglik, np.zeros(len(feat)),
        args=(X_tr, t_tr, e_tr), method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-6},
    )
    risk = X_te @ res.x
    ci = concordance_index(t_te, e_te, risk)
    hr = np.exp(res.x).tolist()
    return (
        {"beta": res.x, "hazard_ratios": hr},
        {
            "c_index": ci,
            "hazard_ratios": dict(zip(feat, hr)),
            "n_events": int((e_te == 1).sum()),
            "n_samples": len(X_te),
        },
    )