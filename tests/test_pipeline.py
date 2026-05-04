"""Tests for the loan-approval pipeline.

Run from the project root:
    pytest tests/ -v

These are sanity tests, not exhaustive. They check:
- Data generator produces expected schema and reasonable distributions
- Preprocessor round-trips correctly
- Pipelines train and produce probabilities in [0, 1]
- Engineered features match training-time formulas
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_generation import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    generate_loan_data,
)
from src.preprocessing import build_preprocessor, get_feature_names


# ─── Data generation ──────────────────────────────────────────────────────

def test_data_shape_and_schema():
    df = generate_loan_data(n=500, seed=0)
    assert len(df) == 500
    expected_cols = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES + [TARGET])
    assert set(df.columns) == expected_cols


def test_target_is_binary():
    df = generate_loan_data(n=1000, seed=0)
    assert df[TARGET].isin([0, 1]).all()


def test_approval_rate_in_reasonable_range():
    """Should produce a meaningful imbalance, not 99/1 or 50/50 exactly."""
    df = generate_loan_data(n=5000, seed=0)
    rate = df[TARGET].mean()
    assert 0.45 < rate < 0.75, f"Approval rate {rate:.2%} outside expected range"


def test_credit_score_signal():
    """Higher credit scores should yield higher approval rates."""
    df = generate_loan_data(n=5000, seed=0)
    high_credit = df[df["credit_score"] >= 740][TARGET].mean()
    low_credit = df[df["credit_score"] <= 580][TARGET].mean()
    assert high_credit > low_credit + 0.20, (
        f"Credit-score signal too weak: high={high_credit:.2f}, low={low_credit:.2f}"
    )


def test_dti_signal():
    """Higher DTI should yield lower approval rates."""
    df = generate_loan_data(n=5000, seed=0)
    high_dti = df[df["dti_ratio"] > 50][TARGET].mean()
    low_dti = df[df["dti_ratio"] < 20][TARGET].mean()
    assert low_dti > high_dti + 0.20, (
        f"DTI signal too weak: low_dti={low_dti:.2f}, high_dti={high_dti:.2f}"
    )


def test_engineered_features_match_formulas():
    """loan_to_income == loan_amount / annual_income (within rounding)."""
    df = generate_loan_data(n=200, seed=0)
    np.testing.assert_allclose(
        df["loan_to_income"].values,
        (df["loan_amount"] / df["annual_income"]).round(3).values,
        rtol=1e-4,
    )


def test_seed_reproducibility():
    df1 = generate_loan_data(n=300, seed=7)
    df2 = generate_loan_data(n=300, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


# ─── Preprocessor ─────────────────────────────────────────────────────────

def test_preprocessor_fits_and_transforms():
    df = generate_loan_data(n=200, seed=0)
    pre = build_preprocessor()
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES]
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(df)
    # We have 8 numerical + 2 engineered + one-hot of 6+4+4 categorical = 24 cols
    assert Xt.shape[1] == 24


def test_preprocessor_handles_unseen_category():
    """One-hot encoder must not fail on unseen categories — important for the app."""
    df = generate_loan_data(n=200, seed=0)
    pre = build_preprocessor()
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES]
    pre.fit(X)
    # Inject a brand-new category in a single row
    X_new = X.iloc[[0]].copy()
    X_new["loan_purpose"] = "new_unseen_purpose"
    Xt = pre.transform(X_new)  # should not raise
    assert Xt.shape == (1, 24)


def test_get_feature_names_matches_output_dim():
    df = generate_loan_data(n=200, seed=0)
    pre = build_preprocessor()
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES]
    Xt = pre.fit_transform(X)
    names = get_feature_names(pre)
    assert len(names) == Xt.shape[1]


# ─── Full pipeline smoke test ─────────────────────────────────────────────

def test_random_forest_trains_and_predicts():
    """End-to-end smoke test on a tiny RF — should beat random."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    df = generate_loan_data(n=2000, seed=42)
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]

    assert ((proba >= 0) & (proba <= 1)).all(), "Probabilities out of bounds"
    auc = roc_auc_score(y_te, proba)
    assert auc > 0.70, f"AUC {auc:.3f} suspiciously low — pipeline may be broken"


def test_lending_club_loader_smoke():
    """Loader returns expected schema and reasonable default rate."""
    from src.data_loader import (
        load_lending_club,
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        ENGINEERED_FEATURES,
        TARGET,
    )
    from pathlib import Path

    if not Path("data/raw/accepted_2007_to_2018Q4.csv").exists():
        import pytest
        pytest.skip("Lending Club CSV not present — skipping integration test")

    df = load_lending_club(nrows=50_000)
    expected = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES + [TARGET]
    assert list(df.columns) == expected, "Column order/names off"
    assert 0.10 < df[TARGET].mean() < 0.30, "Default rate out of expected range"
    assert df.isnull().sum().sum() == 0, "No NaNs should remain after cleaning"