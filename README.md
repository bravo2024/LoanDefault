# LoanDefault

Loan modeling as a **competing-risks** problem: a mortgage leaves the book either by defaulting (a credit loss) or by prepaying (a lost income stream), and a model that lumps both into "bad" misprices each.

A Cox proportional-hazards model is fit on the *default* cause while prepayment competes for the same loans (cause-specific hazards; cumulative incidence in the spirit of Fine & Gray 1999). The optimizer is plain `scipy.optimize` on the Cox partial likelihood — no survival library.

## Data

Simulated loan portfolio (LTV, FICO, note rate, balance) with cause-specific hazards, because loan-level performance data with timing of defaults *and* prepayments sits behind registration walls (Freddie Mac / Fannie Mae loan-level datasets — free but gated). The generator's coefficients are the ground truth the model should recover, which is the point of the exercise:

| Covariate | True effect | Recovered HR |
|---|---|---|
| LTV (per point) | e^0.03 ≈ 1.030 | 1.032 |
| FICO (per point) | e^−0.01 ≈ 0.990 | 0.990 |
| Rate (per point) | e^0.15 ≈ 1.162 | 1.185 |

Held-out concordance index **0.76** on the default cause (264 default events in the test fold). Numbers regenerate with `python train.py`.

## Run it

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Dashboard

| Tab | What it shows |
|---|---|
| **Portfolio** | Loan table, outcome split (active / default / prepaid) |
| **Hazard Model** | Default-cause hazard ratios, held-out C-index |
| **Cumulative Incidence** | Per-cause incidence curves over a configurable horizon |

## Layout

```
src/         generator with cause-specific hazards, Cox PH fit, CIF + C-index
train.py     training pipeline
app.py       Streamlit dashboard
tests/       smoke tests
models/      saved model + metrics (gitignored)
```

MIT licensed.
