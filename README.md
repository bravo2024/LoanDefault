# LoanDefault

Credit risk scoring in the Basel III mold: a scorecard builder plus portfolio analytics.

Trains four classifiers on synthetic Lending Club-style loan data to predict default. Dashboard provides a full credit risk workflow: data exploration, multi-model comparison, Weight-of-Evidence scorecard construction, portfolio risk aggregation, and business impact simulation.

## Run it

```bash
pip install -r requirements.txt
python train.py
pytest -q
streamlit run app.py
```

## Layout

```
LoanDefault/
  src/         data, model, evaluate, persist, visualizations modules
  train.py     training pipeline (multi-model + CV)
  app.py       Streamlit dashboard
  tests/       pytest smoke test
  models/      saved model + metrics (gitignored)
```

## Results

Best model (Logistic Regression) holdout results:

| Metric | Value |
|---|---|
| ROC AUC | 0.916 |
| Gini | 0.831 |
| KS Statistic | 0.729 |
| F1 Score | 0.783 |
| Accuracy | 0.840 |

5-fold CV AUC: 0.923 ± 0.028. Four models compared.

### Dashboard tabs

| Tab | What it does |
|---|---|
| **Data Explorer** | Dataset overview, class balance, numerical feature distributions, feature correlations |
| **Model Lab** | Multi-model comparison, ROC/PR curves, KS curve, calibration, confusion matrix, CV results, permutation importance |
| **Scorecard Builder** | WOE transformation, scorecard points assignment, PD calibration, threshold sweep analysis |
| **Portfolio Risk** | Risk grade distribution, concentration analysis, expected loss calculation |
| **Business Impact** | Approval rate vs default rate trade-off, profit optimisation, regulatory capital impact |

## Data

Synthetic Lending Club-style loan dataset: loan amount, interest rate, term, grade, employment length, annual income, DTI ratio, credit history length, and default status.

## License

MIT
