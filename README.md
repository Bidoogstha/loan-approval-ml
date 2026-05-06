# Loan Default Prediction — End-to-End ML System

A binary classification system for loan default prediction on real Lending Club data, with model comparison, hyperparameter tuning, fairness analysis across demographic groups, calibration, SHAP explainability, and an interactive Streamlit app with a What-if explorer.

**[Live demo →](https://loan-approval-ml-bidoog.streamlit.app/)**

![Hero screenshot](hero.png)

---

## What this project does

This is a portfolio piece, but I tried to build it the way a real lending team would scope the problem rather than the way most online tutorials do.

- **Real data, not synthetic.** ~1.3M loans from the Kaggle Lending Club dataset (2007–2018), filtered to applications with known outcomes (Fully Paid or Charged Off). The pipeline cleans 150+ columns down to 15 predictive features, dropping anything that would cause target leakage from post-origination fields.
- **Five model families compared under identical preprocessing**: Logistic Regression, Random Forest, XGBoost, LightGBM, and a stacking ensemble. Each tuned with Optuna over a 5-fold cross-validated ROC-AUC objective.
- **Fairness analysis using `fairlearn`.** I evaluate three mathematically distinct notions of fairness — demographic parity, equalized odds, and calibration by group — across two sensitive attributes (state and income bracket). The model passes the four-fifths rule for demographic parity but fails equalized odds. This trade-off is documented and discussed below, with references.
- **Interactive Streamlit app with a "What-if explorer"** that lets you snap to preset applicant scenarios (median, borderline, strong) and watch the model's risk prediction shift in real time as you drag any input. SHAP waterfall plots explain individual predictions.

---

## Live demo

Try the deployed app: **https://loan-approval-ml-bidoog.streamlit.app/**

![What-if explorer](whatif_explorer.png)
*Three preset scenarios snap the inputs to demonstrate dramatic differences in predicted default risk. Drag any slider to see the prediction update live, with a delta badge showing the change in approval likelihood.*

![SHAP waterfall](shap_waterfall.png)
*Per-applicant SHAP waterfall showing which features pushed the prediction up or down from the model's base rate.*

![Fairness audit](fairness_chart.png)
*Per-state breakdown of selection rates, with the four-fifths-rule threshold marked.*

---

## Project structure
loan-approval-ml/
├── README.md
├── requirements.txt
├── train.py                    # Main training pipeline
├── data/
│   ├── raw/                    # Raw Lending Club CSV (gitignored, see Quickstart)
│   └── lending_club_sample.csv # Cleaned sample for app inference
├── src/
│   ├── data_loader.py          # Lending Club cleaning + feature schema
│   ├── data_generation.py      # Original synthetic generator (kept as fallback)
│   ├── preprocessing.py        # ColumnTransformer pipeline
│   ├── models.py               # Model definitions and Optuna tuning
│   ├── evaluation.py           # Metrics, plots, statistical tests
│   ├── fairness.py             # Demographic parity, equalized odds, calibration by group
│   └── quick_train.py          # Reduced-trial training for fast iteration
├── models/
│   ├── best_model.pkl          # Saved after training
│   ├── all_pipelines.pkl       # All five tuned models
│   ├── metrics.json            # Per-model test metrics
│   └── fairness.json           # Full fairness audit
├── notebooks/
│   ├── 01_lending_club_eda.ipynb     # Initial EDA, feature selection
│   └── 02_fairness_eda.ipynb         # Disparity analysis by state and income
├── tests/
│   └── test_pipeline.py        # pytest suite
└── app/
└── streamlit_app.py        # Interactive demo with What-if explorer

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/Bidoogstha/loan-approval-ml.git
cd loan-approval-ml
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Download the raw Lending Club CSV from Kaggle:
#    https://www.kaggle.com/datasets/wordsforthewise/lending-club
#    Place accepted_2007_to_2018Q4.csv in data/raw/
#    (The file is ~1.3GB and not committed to the repo.)

# 3. Train the models (generates cleaned data, trains, evaluates, saves artifacts)
python train.py

# 4. Run the fairness audit
python -c "from src.fairness import run_full_audit; run_full_audit()"

# 5. Launch the Streamlit app
streamlit run app/streamlit_app.py
```

For a faster iteration cycle during development, `python src/quick_train.py` runs with reduced Optuna trials (~10 instead of 50).

---

## Methodology

### Data

Real Lending Club data, 2007–2018. Applications with status `Fully Paid` or `Charged Off` are kept; ongoing loans (Current, In Grace Period, Late) are dropped because the outcome is unknown. After cleaning: ~1.3M rows, 15 predictive features.

Class balance: ~80/20 (repaid/defaulted). Mild imbalance, handled with `class_weight='balanced'` in tree-based models.

Two sensitive attributes are carried through the pipeline as non-feature columns for fairness slicing only: `addr_state` (US state) and `income_bracket` (quartile of `annual_inc`).

### Preprocessing

A single `sklearn.compose.ColumnTransformer` that:
- Standard-scales numerical features
- One-hot encodes categoricals with `handle_unknown='ignore'` for app robustness on unseen values
- Passes through engineered ratios (loan-to-income, installment-to-income)

### Models

| Model | Why it's here |
|---|---|
| Logistic Regression | Interpretable baseline; coefficients have direct meaning |
| Random Forest | Strong out-of-the-box, handles non-linearities |
| XGBoost | State-of-the-art for tabular; usually wins |
| LightGBM | Faster than XGBoost, often comparable performance |
| Stacking | Logistic Regression meta-learner over RF + XGB + LGBM |

Each base model is tuned with Optuna (TPE sampler, 50 trials) optimizing 5-fold cross-validated ROC-AUC.

### Evaluation

Held-out test set (20% stratified): Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Brier score, calibration curve, confusion matrices at default and cost-optimal thresholds, and pairwise McNemar's test for model comparison.

### Explainability

SHAP TreeExplainer for the chosen model — global feature importance and per-prediction waterfall plots in the app. Permutation importance and partial dependence plots in the training output.

---

## Fairness analysis

I measure three distinct notions of fairness across two sensitive attributes (state, income bracket). These three metrics are mathematically incompatible to fully satisfy simultaneously (Pleiss et al., NeurIPS 2017; Kleinberg et al., ITCS 2017) — a real model has to trade them off, and the trade-off should be visible.

| Sensitive attribute | Metric | Disparity | Four-fifths rule |
|---|---|---|---|
| State (`addr_state`) | Demographic parity | 7.7pp | ✅ Pass (ratio = 0.92) |
| State (`addr_state`) | Calibration by group | 11.0pp | ❌ Fail |
| State (`addr_state`) | Equalized odds | **28.0pp** | ❌ Fail |
| Income bracket | Demographic parity | 4.7pp | ✅ Pass |
| Income bracket | Calibration by group | 0.9pp | ✅ Pass |
| Income bracket | Equalized odds | 11.4pp | ❌ Fail |

**What this means in plain English.** The model approves applicants at similar rates regardless of which state or income bracket they're in (demographic parity passes). But among applicants who actually repaid their loans, the model is much better at correctly identifying creditworthy borrowers in some states than in others — a 28-percentage-point true-positive-rate gap. The model also fails calibration by state: when it predicts "20% default risk," the actual default rate in some states is closer to 10% and in others closer to 30%.

This is the kind of finding any deployed lending model would face. The federal Equal Credit Opportunity Act (ECOA) and CFPB guidance emphasize disparate impact (the demographic parity story, which my model passes) — but the equalized-odds gap is what a fair-lending economist would flag in review. Resolving it would require either a different model class, post-hoc adjustment (e.g. group-conditional thresholds), or different training data.

Full per-group breakdowns are in `models/fairness.json` and the fairness EDA notebook (`notebooks/02_fairness_eda.ipynb`).

---

## Limitations and honest caveats

- **No temporal validation yet.** The current train/test split is random. Real lending data is non-stationary — borrower quality and default rates shift with the economy. A time-based split using `issue_d` would catch concept drift the random split hides. This is the highest-priority next improvement.
- **Equalized odds gap by state is unresolved.** The model passes the four-fifths rule but fails equalized odds with a 28pp gap. I document the disparity but do not remediate it. Realistic remediation strategies (group-conditional thresholds, fairness-constrained training) are out of scope for v1 but listed in the references below.
- **Threshold = 0.5 is rarely optimal in production.** The pipeline computes a cost-optimal threshold, but the relative cost of a false approval vs. a false denial is a stakeholder decision, not a math one. The app exposes both options as a toggle.
- **The pipeline migration from synthetic to real data took longer than I expected.** The original version of this project used 10K rows of synthetic data with hand-chosen feature relationships. Swapping to 1.3M rows of real Lending Club data exposed messiness the synthetic version didn't simulate — missing employment lengths encoded as multiple string formats, percent strings (`'10.65%'`) that needed parsing, post-origination fields that would cause target leakage if you didn't know to drop them. The refactor is documented in the git log under "Phase 1 complete: switch entire pipeline to Lending Club data."

---

## References & tools

- **Hyperparameter tuning:** Optuna — Akiba et al., 2019
- **Explainability:** SHAP — Lundberg & Lee, 2017
- **Boosting:** XGBoost — Chen & Guestrin, 2016; LightGBM — Ke et al., 2017
- **Calibration & threshold methods:** scikit-learn user guide §1.16
- **Fairness:** `fairlearn` library; Hardt, Price, Srebro 2016 (equalized odds); Kleinberg, Mullainathan, Raghavan 2017 (impossibility); Pleiss et al. 2017 (calibration trade-offs)
- **Imbalanced learning:** `imbalanced-learn`
- **Disparate impact doctrine:** Griggs v. Duke Power, 401 U.S. 424 (1971); EEOC Uniform Guidelines, 29 CFR §1607.4(D)

---

## License

MIT

---

Built by Bidoog Shrestha · [LinkedIn](https://www.linkedin.com/in/bidoog-shrestha-835093282) · [GitHub](https://github.com/Bidoogstha)
