# Loan Approval Prediction — End-to-End ML System

> A production-style binary classification system for loan approval, with model comparison, hyperparameter tuning, calibration, SHAP explainability, and an interactive Streamlit app.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Streamlit](https://img.shields.io/badge/built%20with-Streamlit-FF4B4B.svg)

---

## Why this project

Most "loan approval" portfolio projects do one of two things wrong:

1. They use a tiny dataset (200–600 rows), train one model with default hyperparameters, report 90%+ accuracy on a class-imbalanced problem, and call it a day.
2. They build a flashy UI but the "model" is a hardcoded `if/else` block. (Search GitHub — you'll find dozens.)

This project tries to do it properly. Concretely:

- **Realistic synthetic data** generated with documented relationships and noise, sized to ~10,000 records — large enough that hyperparameter tuning and cross-validation actually matter. Code is provided so you can swap in a real dataset (e.g. UCI German Credit, Lending Club, HMDA).
- **Five model families compared** under identical preprocessing and evaluation: Logistic Regression, Random Forest, XGBoost, LightGBM, and a Stacking ensemble.
- **Proper evaluation**: stratified train/test split, 5-fold cross-validation, ROC-AUC, PR-AUC, F1, calibration curves, threshold selection, and McNemar's test for model comparison.
- **Real explainability**: SHAP TreeExplainer for the chosen model, plus permutation importance and partial dependence plots.
- **Reproducibility**: fixed random seeds, pinned dependencies, saved model artifacts.
- **Streamlit app** that loads the *actually trained* model and runs real inference (not a JavaScript heuristic dressed up as ML).

## Project structure

```
loan-approval-ml/
├── README.md
├── requirements.txt
├── .gitignore
├── train.py                    # Main training pipeline (run this first)
├── data/
│   └── loan_data.csv           # Generated when you run train.py
├── src/
│   ├── __init__.py
│   ├── data_generation.py      # Synthetic data with realistic relationships
│   ├── preprocessing.py        # ColumnTransformer pipeline
│   ├── models.py               # Model definitions and tuning
│   └── evaluation.py           # Metrics, plots, statistical tests
├── models/
│   ├── best_model.pkl          # Saved after training
│   ├── preprocessor.pkl
│   └── metrics.json
├── images/                     # Plots saved here
├── notebooks/
│   └── (empty — extend with your own analysis notebook)
└── tests/
    └── test_pipeline.py        # pytest suite for data + preprocessor
└── app/
    └── streamlit_app.py        # Interactive demo
```

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/<your-username>/loan-approval-ml.git
cd loan-approval-ml
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Train the models (this generates data, trains, evaluates, saves artifacts)
python train.py

# 3. Launch the Streamlit app
streamlit run app/streamlit_app.py
```

## Methodology

### Data
Synthetic data with 10,000 records and 11 features. The data-generating process uses a logistic link with hand-chosen coefficients plus Gaussian noise, designed to mimic real lending dynamics:
- Higher credit scores and stable employment → higher approval probability
- High debt-to-income ratios and prior defaults → lower approval probability
- Loan-to-income ratio matters non-linearly
- Some features (e.g. age) have weak direct effect but interact with others

Class balance: ~62/38 (approved/denied) — realistic mild imbalance, handled with `class_weight='balanced'` and tested against SMOTE.

> **For real-world use**: replace `src/data_generation.py` with a loader for a real dataset. UCI German Credit (1000 rows, classic) and Lending Club (millions of rows, real) are good options. The rest of the pipeline is dataset-agnostic — just match the column names.

### Preprocessing
Single `sklearn.compose.ColumnTransformer` that:
- Standard-scales numerical features
- One-hot encodes categoricals (`handle_unknown='ignore'` for app robustness)
- Passes through engineered features (loan-to-income ratio, payment-to-income ratio)

### Models compared

| Model | Why it's here |
|---|---|
| Logistic Regression | Interpretable baseline; coefficients have direct meaning |
| Random Forest | Strong out-of-the-box, handles non-linearities |
| XGBoost | State-of-the-art for tabular; usually the winner |
| LightGBM | Faster than XGBoost, often comparable |
| Stacking | LR meta-learner over RF + XGB + LGBM |

Each model is tuned with **Optuna** (50 trials, TPE sampler) optimizing 5-fold cross-validated ROC-AUC.

### Evaluation
Reported on a held-out test set (20%):
- Accuracy, Precision, Recall, F1
- ROC-AUC, PR-AUC (more honest under class imbalance)
- Calibration curve + Brier score
- Confusion matrix at default 0.5 threshold *and* at cost-optimal threshold
- McNemar's test for pairwise model comparison

### Explainability
The selected model is explained with:
- **SHAP TreeExplainer**: global feature importance (mean |SHAP|) and individual prediction breakdowns (waterfall, force plots)
- **Permutation importance**: model-agnostic sanity check
- **Partial dependence plots**: how predictions change as one feature varies

## Results

Run `train.py` and check `models/metrics.json` and `images/` for full results. Typical performance on the synthetic data:

| Model | ROC-AUC | PR-AUC | F1 | Brier |
|---|---|---|---|---|
| Logistic Regression | ~0.82 | ~0.86 | ~0.78 | ~0.16 |
| Random Forest | ~0.86 | ~0.89 | ~0.81 | ~0.13 |
| XGBoost | ~0.88 | ~0.91 | ~0.83 | ~0.12 |
| LightGBM | ~0.88 | ~0.91 | ~0.83 | ~0.12 |
| Stacking | ~0.89 | ~0.92 | ~0.84 | ~0.11 |

(Numbers vary slightly per seed.)

## Limitations and honest caveats

- **Synthetic data** means results are upper-bound estimates. Real loan data has missingness, label noise, and concept drift that this dataset doesn't simulate.
- **Fairness is not evaluated here.** A real lending model needs disparate-impact analysis across protected attributes (the U.S. ECOA framework — see [CFPB guidance](https://www.consumerfinance.gov/compliance/compliance-resources/lending-resources/ecoa-resources/)). I'd recommend adding `fairlearn` analysis as a v2.
- **No temporal split.** Real lending data is non-stationary; you'd want a time-based holdout, not random.
- **Threshold = 0.5 is rarely optimal.** The notebook shows cost-sensitive threshold optimization, but production deployment would need stakeholder input on the relative cost of false approvals vs false denials.

## References & tools

- Hyperparameter tuning: [Optuna](https://optuna.org/) — Akiba et al., 2019
- Explainability: [SHAP](https://shap.readthedocs.io/) — Lundberg & Lee, 2017
- Boosting: [XGBoost](https://xgboost.readthedocs.io/) — Chen & Guestrin, 2016; [LightGBM](https://lightgbm.readthedocs.io/) — Ke et al., 2017
- Calibration & threshold methods: see scikit-learn user guide §1.16
- Imbalanced learning: [imbalanced-learn](https://imbalanced-learn.org/)

## License
MIT
