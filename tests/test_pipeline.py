"""Tests for the loan-approval pipeline.

Run from the project root:
    pytest tests/ -v

These are sanity tests, not exhaustive. They check:
- Lending Club loader returns expected schema and reasonable distributions
- Preprocessor fits, transforms, and handles unseen categories
- Engineered features match training-time formulas
- A simple Random Forest trains end-to-end and produces valid probabilities

Tests skip cleanly if the Lending Club CSV is not present (e.g., on Streamlit Cloud).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERICAL_FEATURES,
    SENSITIVE_FEATURES,
    TARGET,
    load_lending_club,
)
from src.preprocessing import build_preprocessor, get_feature_names

CSV_PATH = ROOT / "data" / "raw" / "accepted_2007_to_2018Q4.csv"


def _load_or_skip(nrows: int = 5000) -> pd.DataFrame:
    """Helper: load Lending Club data or skip the test."""
    if not CSV_PATH.exists():
        pytest.skip("Lending Club CSV not present — skipping integration test")
    return load_lending_club(nrows=nrows)


# ─── Data loader ──────────────────────────────────────────────────────────

def test_data_shape_and_schema():
    df = _load_or_skip(nrows=2000)
    expected_cols = set(
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        + ENGINEERED_FEATURES + SENSITIVE_FEATURES + [TARGET]
    )
    assert set(df.columns) == expected_cols


def test_target_is_binary():
    df = _load_or_skip(nrows=2000)
    assert df[TARGET].isin([0, 1]).all()


def test_default_rate_in_reasonable_range():
    """Lending Club has ~20% default rate — confirm we filtered/labeled correctly."""
    df = _load_or_skip(nrows=10_000)
    rate = df[TARGET].mean()
    assert 0.10 < rate < 0.30, f"Default rate {rate:.2%} outside expected range"


def test_credit_score_signal():
    """Higher FICO should yield lower default rate."""
    df = _load_or_skip(nrows=10_000)
    high_fico = df[df["fico_range_low"] >= 720][TARGET].mean()
    low_fico = df[df["fico_range_low"] < 680][TARGET].mean()
    assert low_fico > high_fico, (
        f"FICO signal wrong direction: high={high_fico:.2f}, low={low_fico:.2f}"
    )


def test_dti_signal():
    """Higher DTI should yield higher default rate."""
    df = _load_or_skip(nrows=10_000)
    high_dti = df[df["dti"] > 25][TARGET].mean()
    low_dti = df[df["dti"] < 15][TARGET].mean()
    assert high_dti > low_dti, (
        f"DTI signal wrong direction: low={low_dti:.2f}, high={high_dti:.2f}"
    )


def test_engineered_features_match_formulas():
    """loan_to_income == loan_amnt / annual_inc (within rounding)."""
    df = _load_or_skip(nrows=500)
    np.testing.assert_allclose(
        df["loan_to_income"].values,
        (df["loan_amnt"] / df["annual_inc"]).round(4).values,
        rtol=1e-3,
    )


def test_no_residual_nans():
    df = _load_or_skip(nrows=2000)
    assert df.isnull().sum().sum() == 0, "Loader should produce no NaNs"


# ─── Preprocessor ─────────────────────────────────────────────────────────

def test_preprocessor_fits_and_transforms():
    df = _load_or_skip(nrows=2000)
    pre = build_preprocessor()
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES]
    Xt = pre.fit_transform(X)
    assert Xt.shape[0] == len(df)
    # 10 numerical + 2 engineered + OHE of 5 categoricals = ~30+ cols
    assert Xt.shape[1] >= 20, f"Output dimension {Xt.shape[1]} suspiciously low"


def test_preprocessor_handles_unseen_category():
    """One-hot encoder must not fail on unseen categories — important for the app."""
    df = _load_or_skip(nrows=2000)
    pre = build_preprocessor()
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES]
    pre.fit(X)
    # Inject a brand-new category in a single row
    X_new = X.iloc[[0]].copy()
    X_new["purpose"] = "new_unseen_purpose"
    Xt = pre.transform(X_new)  # should not raise
    assert Xt.shape[0] == 1


def test_get_feature_names_matches_output_dim():
    df = _load_or_skip(nrows=2000)
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

    df = _load_or_skip(nrows=10_000)
    X = df[NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]

    assert ((proba >= 0) & (proba <= 1)).all(), "Probabilities out of bounds"
    auc = roc_auc_score(y_te, proba)
    # On real data, expect 0.65–0.75; floor at 0.60 for tiny-sample noise.
    assert auc > 0.60, f"AUC {auc:.3f} suspiciously low — pipeline may be broken"


def test_lending_club_loader_full_smoke():
    """Loader works at scale — smoke test."""
    if not CSV_PATH.exists():
        pytest.skip("Lending Club CSV not present")
    df = load_lending_club(nrows=50_000)
    expected = (
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        + ENGINEERED_FEATURES + SENSITIVE_FEATURES + [TARGET]
    )
    assert list(df.columns) == expected
    assert 0.10 < df[TARGET].mean() < 0.30
    # SENSITIVE_FEATURES (addr_state, income_bracket) shouldn't be NaN either.
    assert df.isnull().sum().sum() == 0

# ─── Fairness ─────────────────────────────────────────────────────────────

def test_fairness_demographic_parity_obvious_disparity():
    """Construct a toy case with clear group disparity; metric should detect it."""
    from src.fairness import demographic_parity
    # Group A: 200 people, all predicted "approved" (pred=0)
    # Group B: 200 people, all predicted "denied" (pred=1)
    # Selection rates 1.0 and 0.0 → disparity 1.0.
    y_pred = np.array([0]*200 + [1]*200)
    sensitive = pd.Series(["A"]*200 + ["B"]*200)
    rep = demographic_parity(y_pred, sensitive, "toy")
    assert rep.disparity > 0.95, f"Expected near-1 disparity, got {rep.disparity}"
    assert rep.four_fifths_pass is False


def test_fairness_demographic_parity_no_disparity():
    """Identical groups → near-zero disparity."""
    from src.fairness import demographic_parity
    rng = np.random.default_rng(0)
    y_pred = rng.integers(0, 2, size=1000)
    sensitive = pd.Series(["A"]*500 + ["B"]*500)
    rep = demographic_parity(y_pred, sensitive, "toy")
    assert rep.disparity < 0.10
    assert rep.four_fifths_pass is True


def test_fairness_calibration_detects_miscalibration():
    """Group A: predict 0.5, actual 0.5 (calibrated). Group B: predict 0.5, actual 0.9."""
    from src.fairness import calibration_by_group
    rng = np.random.default_rng(0)
    n = 500
    y_a = rng.binomial(1, 0.5, n)
    y_b = rng.binomial(1, 0.9, n)
    p_a = np.full(n, 0.5)
    p_b = np.full(n, 0.5)
    y_true = np.concatenate([y_a, y_b])
    y_proba = np.concatenate([p_a, p_b])
    sensitive = pd.Series(["A"]*n + ["B"]*n)
    rep = calibration_by_group(y_true, y_proba, sensitive, "toy")
    assert rep.disparity > 0.20, f"Should detect ~0.4 miscalibration, got {rep.disparity}"


def test_fairness_skips_tiny_groups():
    """Groups smaller than min_group_size are excluded."""
    from src.fairness import demographic_parity
    y_pred = np.array([0]*5 + [1]*200)  # group A is tiny (only 5)
    sensitive = pd.Series(["A"]*5 + ["B"]*200)
    rep = demographic_parity(y_pred, sensitive, "toy", min_group_size=50)
    # Only group B remains → fewer than 2 groups → skipped.
    assert "skipped" in rep.notes.lower()