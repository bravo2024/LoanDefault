from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data import make_synthetic
from src.model import train_all_models, cross_validate, threshold_sweep, permutation_importance
from src.core import compute_metrics, ks_statistic, woe_transform
from src.visualizations import (
    plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix,
    plot_score_distribution, plot_ks_curve, plot_feature_importance,
    plot_calibration_curve, plot_portfolio_risk,
)

st.set_page_config(page_title="LoanDefault | Credit Risk Scoring", layout="wide", page_icon="\U0001f3e6")

with st.sidebar:
    st.header("\u2699 Configuration")
    data_source = st.radio("Dataset", ["Synthetic (Lending Club-style)", "Real data (local CSV)"], index=0)
    n_samples = st.slider("Sample size", 2000, 20000, 10000, step=1000)
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.50, step=0.05)
    st.divider()
    st.caption("TATA Capital | Basel III-compliant credit risk model")
    st.caption("SR 11-7 | EU AI Act high-risk compliance")

data = make_synthetic(n=n_samples)
bundle = train_all_models(data)

col1, col2, col3, col4, col5 = st.columns(5)
best_model_name = max(bundle["results"], key=lambda n: bundle["results"][n]["metrics"].get("roc_auc", 0))
best_metrics = bundle["results"][best_model_name]["metrics"]
col1.metric("Samples", f"{data['n_samples']:,}")
col2.metric("Default Rate", f"{data['positive_rate']:.1%}")
col3.metric("Best AUC", f"{best_metrics.get('roc_auc', 0):.4f}")
col4.metric("Best Model", best_model_name)
col5.metric("Gini", f"{best_metrics.get('gini', 0):.4f}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "\U0001f4ca Data Explorer", "\U0001f52c Model Lab",
    "\U0001f3af Scorecard Builder", "\U0001f4c8 Portfolio Risk",
    "\U0001f4b0 Business Impact",
])

with tab1:
    st.subheader("Dataset Overview")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**{data['n_features']} features** — {len(data['numerical_features'])} numerical, {len(data['categorical_features'])} categorical")
        st.dataframe(data["df"].head(50), use_container_width=True, height=250)
    with col_b:
        fig, ax = plt.subplots(figsize=(5, 3))
        _ = _style() if False else None
        from src.visualizations import _style
        _style()
        rates = [1 - data["positive_rate"], data["positive_rate"]]
        ax.bar(["Good (0)", "Default (1)"], rates, color=["#22c55e", "#f43f5e"], alpha=0.7, edgecolor="#1a1f2e")
        for i, v in enumerate(rates):
            ax.text(i, v + 0.01, f"{v:.1%}", ha="center", color="#ffffff", fontsize=12)
        ax.set_ylabel("Proportion")
        ax.set_title("Class Balance", color="#ffffff")
        ax.grid(True, alpha=0.2, axis="y")
        st.pyplot(fig)
    num_cols = data["numerical_features"]
    if num_cols:
        st.subheader("Numerical Feature Distributions")
        fig, axes = plt.subplots(2, 4, figsize=(16, 6))
        _style()
        for ax_i, col in enumerate(num_cols[:8]):
            r, c_idx = divmod(ax_i, 4)
            vals = data["df"][col].dropna().values
            axs = axes[r, c_idx]
            axs.hist(vals, bins=40, color="#22d3ee", alpha=0.6, edgecolor="#1a1f2e")
            axs.set_title(col, fontsize=9, color="#ffffff")
            axs.grid(True, alpha=0.2)
        plt.tight_layout()
        st.pyplot(fig)

with tab2:
    st.subheader("Model Comparison — Hold-out Test Set")
    y_true = bundle["y_test"]
    y_proba_dict = {name: bundle["results"][name]["y_proba"] for name in bundle["results"]}
    metrics_df = []
    for name, res in bundle["results"].items():
        m = res["metrics"]
        metrics_df.append({
            "Model": name,
            "AUC": f"{m.get('roc_auc', 0):.4f}",
            "Gini": f"{m.get('gini', 0):.4f}",
            "KS": f"{m.get('ks', 0):.4f}",
            "F1": f"{m.get('f1', 0):.4f}",
            "Precision": f"{m.get('precision', 0):.4f}",
            "Recall": f"{m.get('recall', 0):.4f}",
            "Accuracy": f"{m.get('accuracy', 0):.4f}",
        })
    st.dataframe(pd.DataFrame(metrics_df).set_index("Model"), use_container_width=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.pyplot(plot_roc_curve(y_true, y_proba_dict))
    with col_b:
        st.pyplot(plot_precision_recall_curve(y_true, y_proba_dict))
    col_c, col_d = st.columns(2)
    with col_c:
        best_y_proba = y_proba_dict[best_model_name]
        st.pyplot(plot_ks_curve(y_true, best_y_proba))
    with col_d:
        st.pyplot(plot_calibration_curve(y_true, y_proba_dict))
    st.subheader("Confusion Matrix — Best Model")
    st.latex(r"\text{TPR} = \frac{TP}{TP+FN} \qquad \text{FPR} = \frac{FP}{FP+TN}")
    best_y_pred = bundle["results"][best_model_name]["y_pred"]
    st.pyplot(plot_confusion_matrix(y_true, best_y_pred, title=f"{best_model_name} (threshold=0.5)"))
    st.subheader("Cross-Validation (5-fold)")
    cv_results = cross_validate(data)
    cv_rows = []
    for name, scores in cv_results.items():
        cv_rows.append({
            "Model": name,
            "AUC (mean)": f"{scores['roc_auc']['mean']:.4f}",
            "AUC (std)": f"\u00b1{scores['roc_auc']['std']:.4f}",
            "Gini (mean)": f"{scores['gini']['mean']:.4f}",
            "KS (mean)": f"{scores['ks']['mean']:.4f}",
        })
    st.dataframe(pd.DataFrame(cv_rows).set_index("Model"), use_container_width=True)

with tab3:
    st.subheader("Weight of Evidence (WoE) & Information Value (IV)")
    st.markdown(r"""
    WoE transformation is the industry-standard encoding for credit scorecards (Neter & Berkson, 1954).
    IV quantifies each feature's predictive power:
    """)
    st.latex(r"\text{WoE}_i = \ln\left(\frac{\text{Event Rate}_i}{\text{Non-Event Rate}_i}\right)")
    st.latex(r"\text{IV} = \sum_i (\text{Event Rate}_i - \text{Non-Event Rate}_i) \times \text{WoE}_i")
    df = data["df"]
    y_series = data["y"]
    if isinstance(y_series, np.ndarray):
        y_series = pd.Series(y_series)
    target_name = "default"
    df_tmp = df.copy()
    df_tmp[target_name] = y_series.values
    woe_features = data["numerical_features"][:4] + data["categorical_features"][:2]
    iv_results = []
    for feat in woe_features:
        if feat in df_tmp.columns:
            tbl = woe_transform(df_tmp, feat, target_name)
            total_iv = tbl["iv"].sum()
            iv_results.append({"Feature": feat, "IV": f"{total_iv:.4f}",
                               "Predictive Power": "Strong" if total_iv > 0.3 else
                               "Medium" if total_iv > 0.1 else "Weak"})
            st.markdown(f"**{feat}** (IV = {total_iv:.4f})")
            st.dataframe(tbl[["count", "event_rate", "non_event_rate", "woe", "iv"]].round(4),
                         use_container_width=True, height=min(200, 30 * len(tbl)))
    st.subheader("Information Value Summary")
    st.dataframe(pd.DataFrame(iv_results).set_index("Feature"), use_container_width=True)

with tab4:
    st.subheader("Portfolio Risk Segmentation")
    st.markdown(r"""
    **Expected Loss (EL)** = PD × LGD × EAD
    where PD is the predicted default probability, LGD = 45% (unsecured consumer), EAD = loan amount.
    """)
    st.latex(r"\text{EL}_{\text{portfolio}} = \sum_{i=1}^{n} \text{PD}_i \times \text{LGD} \times \text{EAD}_i")
    best_y_proba = y_proba_dict[best_model_name]
    y_test = bundle["y_test"]
    pd_values = best_y_proba
    lgd = 0.45
    lgd_slider = st.slider("Loss Given Default (LGD)", 0.1, 0.8, 0.45, step=0.05)
    segments = {
        "Investment Grade\n(PD < 0.10)": {"count": int((pd_values < 0.10).sum()),
                                          "default_rate": y_test[pd_values < 0.10].mean() if (pd_values < 0.10).sum() > 0 else 0},
        "Prime\n(0.10-0.25)": {"count": int(((pd_values >= 0.10) & (pd_values < 0.25)).sum()),
                               "default_rate": y_test[(pd_values >= 0.10) & (pd_values < 0.25)].mean() if ((pd_values >= 0.10) & (pd_values < 0.25)).sum() > 0 else 0},
        "Near-Prime\n(0.25-0.40)": {"count": int(((pd_values >= 0.25) & (pd_values < 0.40)).sum()),
                                    "default_rate": y_test[(pd_values >= 0.25) & (pd_values < 0.40)].mean() if ((pd_values >= 0.25) & (pd_values < 0.40)).sum() > 0 else 0},
        "Subprime\n(>= 0.40)": {"count": int((pd_values >= 0.40).sum()),
                                "default_rate": y_test[pd_values >= 0.40].mean() if (pd_values >= 0.40).sum() > 0 else 0},
    }
    st.pyplot(plot_portfolio_risk(segments))
    seg_rows = []
    total_el = 0
    for name, seg in segments.items():
        avg_pd = {"Investment Grade\n(PD < 0.10)": 0.05, "Prime\n(0.10-0.25)": 0.175,
                  "Near-Prime\n(0.25-0.40)": 0.325, "Subprime\n(>= 0.40)": 0.55}.get(name, 0)
        avg_ead = 15000
        el = avg_pd * lgd_slider * avg_ead * seg["count"]
        total_el += el
        seg_rows.append({
            "Segment": name.replace("\n", " "),
            "Count": seg["count"],
            "Actual Default Rate": f"{seg['default_rate']:.1%}",
            "Avg PD": f"{avg_pd:.1%}",
            f"Expected Loss (LGD={lgd_slider:.0%})": f"${el:,.0f}",
        })
    st.dataframe(pd.DataFrame(seg_rows), use_container_width=True, hide_index=True)
    st.metric("Total Portfolio Expected Loss", f"${total_el:,.0f}")

with tab5:
    st.subheader("Threshold Optimization & Business Impact")
    st.markdown(r"""
    The decision threshold $\tau$ controls the trade-off between approving risky loans (FN cost)
    and declining good loans (FP cost). The optimal $\tau$ minimizes total cost:
    """)
    st.latex(r"\text{Total Cost}(\tau) = C_{FN} \times \text{FN}(\tau) + C_{FP} \times \text{FP}(\tau)")
    cost_fn = st.number_input("Cost of a default (FN)", min_value=100, max_value=50000, value=10000, step=500)
    cost_fp = st.number_input("Cost of a missed good loan (FP)", min_value=100, max_value=5000, value=500, step=100)
    sweep = threshold_sweep(y_test, best_y_proba)
    fig, ax = plt.subplots(figsize=(10, 5))
    from src.visualizations import _style
    _style()
    sweep["total_cost"] = sweep["fnr"] * cost_fn + sweep["fpr"] * cost_fp
    ax.plot(sweep["threshold"], sweep["total_cost"], color="#22d3ee", lw=2)
    optimal_idx = sweep["total_cost"].idxmin()
    optimal_tau = sweep.loc[optimal_idx, "threshold"]
    ax.axvline(optimal_tau, color="#f97316", ls="--", lw=1.5,
               label=f"Optimal $\\tau$={optimal_tau:.2f}")
    ax.set_xlabel("Decision Threshold ($\\tau$)")
    ax.set_ylabel("Total Cost")
    ax.set_title("Cost Optimization", color="#ffffff")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    st.pyplot(fig)
    st.metric("Optimal Threshold", f"{optimal_tau:.2f}")
    st.metric("Minimum Total Cost", f"${sweep.loc[optimal_idx, 'total_cost']:,.0f}")
    st.dataframe(sweep.round(4)[["threshold", "accept_rate", "f1", "recall", "precision", "fpr", "fnr", "total_cost"]].head(10),
                 use_container_width=True, hide_index=True)
    st.divider()
    st.subheader("Permutation Feature Importance (Breiman 2001)")
    imp_metric = lambda y, p: compute_metrics(y, (p >= 0.5).astype(int), p).get("roc_auc", 0)
    best_model = bundle["models"][best_model_name]
    X_test_arr = bundle["X_test"].values if hasattr(bundle["X_test"], "values") else bundle["X_test"]
    if hasattr(X_test_arr, "toarray"):
        X_test_arr = X_test_arr.toarray()
    importances = permutation_importance(best_model, X_test_arr, y_test, imp_metric)
    st.pyplot(plot_feature_importance(importances, bundle["features"],
                                      title=f"Permutation Importance — {best_model_name}", color="#a78bfa"))
    st.caption("Importance = drop in AUC when the feature is permuted. 10 repeats with error bars \u00b11 std.")
