"""
Run a fairness audit on the trained model and write results to
models/fairness.json. The Streamlit Fairness tab reads this file.

Usage:
    python audit_fairness.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import (
    CATEGORICAL_FEATURES, ENGINEERED_FEATURES, NUMERICAL_FEATURES,
    SENSITIVE_FEATURES, TARGET, load_lending_club,
)
from src.fairness import (
    calibration_by_group, demographic_parity, equalized_odds, report_to_dict,
)


def main():
    project_root = Path(__file__).resolve().parent
    model_path = project_root / "models" / "best_model.pkl"
    out_path = project_root / "models" / "fairness.json"

    print("Loading data…")
    # 100k rows is enough for a stable audit; full 1.3M takes too long.
    df = load_lending_club(nrows=100_000)

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES
    X = df[feature_cols]
    y = df[TARGET].values
    sensitive = df[SENSITIVE_FEATURES]

    # Use the same split as training (random_state=42, test_size=0.20).
    _, X_test, _, y_test, _, sens_test = train_test_split(
        X, y, sensitive, test_size=0.20, random_state=42, stratify=y,
    )
    print(f"Test set size: {len(X_test):,}")

    print("Loading trained model…")
    pipe = joblib.load(model_path)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.50).astype(int)

    print("\nOverall test-set metrics:")
    print(f"  Default rate (actual):     {y_test.mean():.3f}")
    print(f"  Default rate (predicted):  {y_pred.mean():.3f}")
    print(f"  Mean predicted probability: {y_proba.mean():.3f}")

    reports = []
    for attr in SENSITIVE_FEATURES:
        s = sens_test[attr].astype(str).reset_index(drop=True)
        print(f"\n=== Auditing by: {attr} ===")
        for fn, name in [
            (demographic_parity, "demographic_parity"),
            (calibration_by_group, "calibration_by_group"),
        ]:
            if name == "demographic_parity":
                rep = fn(y_pred, s, attr)
            else:
                rep = fn(y_test, y_proba, s, attr)
            print(f"  {name:24s} disparity = {rep.disparity:.3f}  "
                  f"{'PASS' if rep.four_fifths_pass else 'FAIL'}  "
                  f"({rep.notes})")
            reports.append(report_to_dict(rep))

        rep = equalized_odds(y_test, y_pred, s, attr)
        print(f"  {'equalized_odds':24s} disparity = {rep.disparity:.3f}  "
              f"{'PASS' if rep.four_fifths_pass else 'FAIL'}  "
              f"({rep.notes})")
        reports.append(report_to_dict(rep))

    with open(out_path, "w") as f:
        json.dump({"reports": reports}, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()