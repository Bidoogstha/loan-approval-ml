"""
Real Lending Club data loader.

Replaces the synthetic generator with cleaned, real-world loan data.
The Kaggle 'wordsforthewise/lending-club' dataset has ~150 columns; we keep 15
predictive features after removing target leakage and post-origination fields.

The output schema is intentionally similar to src/data_generation.py so the rest
of the pipeline (preprocessor, training, app) needs minimal changes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ── Feature schema ──────────────────────────────────────────────────────
NUMERICAL_FEATURES = [
    "loan_amnt", "term_months", "int_rate", "installment", "annual_inc",
    "dti", "fico_range_low", "open_acc", "revol_util", "total_acc",
]
CATEGORICAL_FEATURES = [
    "grade", "home_ownership", "purpose", "verification_status", "emp_length",
]
ENGINEERED_FEATURES = [
    "loan_to_income", "installment_to_income",
]
TARGET = "defaulted"
# Sensitive/protected-attribute proxies for fairness analysis.
# These are NOT used as model features — they're carried through so we can
# slice predictions by group at evaluation time.
SENSITIVE_FEATURES = ["addr_state", "income_bracket"]

# Loans where the outcome is known. Anything else (Current, In Grace Period,
# Late, Default still in collections) is dropped because the label is uncertain.
KEEP_STATUSES = ["Fully Paid", "Charged Off"]


def _clean_term(s: pd.Series) -> pd.Series:
    """' 36 months' -> 36, ' 60 months' -> 60. Returns int."""
    return s.str.extract(r"(\d+)", expand=False).astype(float).astype("Int64")


def _clean_pct(s: pd.Series) -> pd.Series:
    """'10.65%' -> 10.65 (kept as percent, not fraction). Returns float."""
    return s.astype(str).str.rstrip("%").replace("nan", np.nan).astype(float)


def _clean_emp_length(s: pd.Series) -> pd.Series:
    """
    Normalize messy strings:
      '< 1 year', '1 year', '2 years', ..., '10+ years' -> bucketed strings.
    Missing values become 'unknown' (a real signal: people who skipped the field).
    """
    mapping = {
        "< 1 year": "lt_1",
        "1 year": "1",
        "2 years": "2",
        "3 years": "3",
        "4 years": "4",
        "5 years": "5",
        "6 years": "6",
        "7 years": "7",
        "8 years": "8",
        "9 years": "9",
        "10+ years": "10_plus",
    }
    return s.map(mapping).fillna("unknown")


def load_lending_club(
    path: Path | str = "data/raw/accepted_2007_to_2018Q4.csv",
    nrows: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Load and clean Lending Club data.

    Args:
        path: path to the raw CSV (or .csv.gz).
        nrows: if set, load only this many rows. Useful for fast iteration.
        seed: for any sampling.

    Returns:
        DataFrame with columns NUMERICAL_FEATURES + CATEGORICAL_FEATURES +
        ENGINEERED_FEATURES + [TARGET].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Lending Club CSV not found at {path}. "
            "Download from https://www.kaggle.com/datasets/wordsforthewise/lending-club "
            "and place at data/raw/accepted_2007_to_2018Q4.csv"
        )

    # Only read columns we need — saves memory drastically (150 cols -> 16).
    raw_cols = [
        "loan_amnt", "term", "int_rate", "installment", "annual_inc",
        "dti", "fico_range_low", "open_acc", "revol_util", "total_acc",
        "grade", "home_ownership", "purpose", "verification_status",
        "emp_length", "loan_status", "addr_state",
    ]
    df = pd.read_csv(path, low_memory=False, nrows=nrows, usecols=raw_cols)

    # Filter to known-outcome loans only.
    df = df[df["loan_status"].isin(KEEP_STATUSES)].copy()
    df[TARGET] = (df["loan_status"] == "Charged Off").astype(int)
    df = df.drop(columns=["loan_status"])

    # Clean the messy string columns.
    df["term_months"] = _clean_term(df["term"]).astype("float64")
    df = df.drop(columns=["term"])
    df["int_rate"] = _clean_pct(df["int_rate"])
    df["revol_util"] = _clean_pct(df["revol_util"])
    df["emp_length"] = _clean_emp_length(df["emp_length"])

    # Handle missingness with intent (not just dropna everywhere):
    #   dti: tiny number missing → drop those rows
    #   revol_util: missing means no revolving credit → fill with 0
    #   annual_inc: a handful of zeros and missings → drop (income of $0 is invalid)
    df = df.dropna(subset=["dti", "annual_inc"])
    df = df[df["annual_inc"] > 0]
    df["revol_util"] = df["revol_util"].fillna(0.0)

    # Engineered features (mirror what src/data_generation.py produces).
    df["loan_to_income"] = (df["loan_amnt"] / df["annual_inc"]).round(4)
    df["installment_to_income"] = (
        (df["installment"] * 12) / df["annual_inc"]
    ).round(4)

    # Income bracket — for fairness slicing. Quartiles based on this dataset.
    # ~25k, 50k, 80k roughly correspond to Q1/Q2/Q3 of US household income.
    df["income_bracket"] = pd.cut(
        df["annual_inc"],
        bins=[0, 40_000, 65_000, 100_000, np.inf],
        labels=["low", "lower_mid", "upper_mid", "high"],
    ).astype(str)

    # Reorder for consistency.
    cols = (
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES
        + SENSITIVE_FEATURES + [TARGET]
    )
    df = df[cols].reset_index(drop=True)

    # Final sanity: shouldn't be any NaN left in features we expect to be clean.
    n_before = len(df)
    df = df.dropna(subset=NUMERICAL_FEATURES + ENGINEERED_FEATURES)
    n_after = len(df)
    if n_before != n_after:
        print(f"Note: dropped {n_before - n_after} rows with residual NaN "
              f"({(n_before - n_after) / n_before:.2%} of data).")

    return df


if __name__ == "__main__":
    # Allows you to test the loader from the command line:
    #   python src/data_loader.py
    df = load_lending_club()
    print(f"Loaded {len(df):,} rows")
    print(f"Default rate: {df[TARGET].mean():.1%}")
    print(f"\nDtypes:\n{df.dtypes}")
    print(f"\nFirst row:\n{df.iloc[0]}")