from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from src.data import make_synthetic
from src.model import fit_and_evaluate
from src.core import cumulative_incidence

st.set_page_config(page_title="LoanDefault | Competing-Risks Credit Model", layout="wide", page_icon="\U0001f3e6")

plt.rcParams.update({
    "figure.facecolor": "#0f172a", "axes.facecolor": "#1e293b",
    "axes.edgecolor": "#334155", "axes.labelcolor": "#cbd5e1",
    "xtick.color": "#94a3b8", "ytick.color": "#94a3b8",
    "text.color": "#f1f5f9", "grid.color": "#334155", "grid.alpha": 0.4,
})

with st.sidebar:
    st.header("⚙ Configuration")
    n_samples = st.slider("Portfolio size", 500, 10000, 2000, step=500)
    seed = st.number_input("Random seed", 0, 999, 42)
    st.divider()
    st.caption("Cox PH · competing risks: default vs prepayment")

st.title("🏦 LoanDefault — Competing-Risks Loan Modeling")
st.markdown(
    "A mortgage leaves the book two ways: **default** (a credit loss) or **prepayment** "
    "(a lost income stream). Treating them as one 'bad' outcome misprices both. "
    "This model fits a Cox proportional-hazards model on the default cause while "
    "prepayment competes for the same loans.")


@st.cache_resource(show_spinner="Fitting Cox model…")
def get_fit(n: int, seed: int):
    data = make_synthetic(n=n, seed=seed)
    model, metrics = fit_and_evaluate(data, seed=seed)
    return data, model, metrics


data, model, metrics = get_fit(int(n_samples), int(seed))
df = data["df"]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Loans", f"{data['n_samples']:,}")
c2.metric("Defaults", f"{(df['event'] == 1).mean():.1%}")
c3.metric("Prepayments", f"{(df['event'] == 2).mean():.1%}")
c4.metric("C-index (default)", f"{metrics['c_index']:.3f}")

t1, t2, t3 = st.tabs(["📊 Portfolio", "🔬 Hazard Model", "📈 Cumulative Incidence"])

with t1:
    st.dataframe(df.head(50), use_container_width=True, height=220)
    fig, ax = plt.subplots(figsize=(6, 3))
    labels = {0: "Censored (active)", 1: "Default", 2: "Prepaid"}
    counts = df["event"].map(labels).value_counts()
    ax.bar(counts.index, counts.values, color=["#38bdf8", "#f43f5e", "#22c55e"][:len(counts)])
    ax.set_title("Loan outcomes"); ax.grid(True, alpha=.2)
    st.pyplot(fig)

with t2:
    st.markdown("Hazard ratios for the **default** cause (per unit of each covariate). "
                "HR > 1 raises default risk; HR < 1 is protective.")
    hr = metrics["hazard_ratios"]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    names, vals = list(hr.keys()), list(hr.values())
    colors = ["#f43f5e" if v > 1 else "#22c55e" for v in vals]
    ax.barh(names, vals, color=colors)
    ax.axvline(1.0, color="#94a3b8", ls="--", lw=1)
    ax.set_xlabel("Hazard ratio"); ax.set_title("Default-cause hazard ratios")
    ax.grid(True, alpha=.2)
    st.pyplot(fig)
    st.caption(f"Held-out concordance index {metrics['c_index']:.3f} on "
               f"{metrics['n_samples']:,} loans ({metrics['n_events']} default events). "
               "LTV and rate push risk up; FICO pulls it down — the signs come out of "
               "the data, they are not imposed.")

with t3:
    st.markdown("Cumulative incidence per cause — the honest replacement for 1−KM "
                "when risks compete (Fine & Gray 1999).")
    horizon = st.slider("Horizon (years)", 1, 10, 5)
    max_t = horizon * 365
    cif_def = np.cumsum(cumulative_incidence(df["time"], df["event"], event_type=1, max_time=max_t))
    cif_pre = np.cumsum(cumulative_incidence(df["time"], df["event"], event_type=2, max_time=max_t))
    months = np.arange(len(cif_def)) / 12 * 365 / 30 / 12
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(months, cif_def, color="#f43f5e", lw=2, label="Default")
    ax.plot(months, cif_pre, color="#22c55e", lw=2, label="Prepayment")
    ax.set_xlabel("Years on book"); ax.set_ylabel("Cumulative incidence")
    ax.set_title("Competing-risks cumulative incidence"); ax.legend(); ax.grid(True, alpha=.2)
    st.pyplot(fig)
