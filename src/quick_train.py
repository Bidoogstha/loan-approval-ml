"""
Fast inline training used by the Streamlit app on first launch.

Same artifact format as train.py, but:
  - 3,000 rows instead of 10,000
  - No Optuna — sensible default hyperparameters
  - No permutation importance (slow)
  - No PNG plots saved (the app generates plots from the cached results dict)

Runs in ~30–60 seconds on a laptop. Produces:
  models/best_model.pkl
  models/all_pipelines.pkl
  models/metrics.json
  data/loan_data.csv
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# These are harmless: sklearn's StackingClassifier passes numpy arrays internally,
# so LGBM/XGB warn about lost feature names. Predictions are unaffected.
warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
warnings.filterwarnings("ignore", category=DataConversionWarning)

from src.data_generation import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    generate_loan_data,
)
from src.evaluation import compute_metrics, find_optimal_threshold, mcnemar_test
from src.preprocessing import build_preprocessor

SEED = 42
N_RECORDS = 3_000
TEST_SIZE = 0.20


def _make_xgb():
    from xgboost import XGBClassifier
    return XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        eval_metric="logloss", tree_method="hist", random_state=SEED, n_jobs=-1,
    )


def _make_lgbm():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=200, max_depth=-1, num_leaves=31, learning_rate=0.08,
        subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )


def _safe_make(factory, name: str):
    """Try to instantiate a model. Return None if its native library is missing."""
    try:
        return factory()
    except Exception as e:
        print(f"⚠️  Skipping {name}: {type(e).__name__}: {e}")
        return None


def _make_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=10,
        min_samples_leaf=4, random_state=SEED, n_jobs=-1,
    )


def _make_logreg():
    return LogisticRegression(C=1.0, max_iter=1000, random_state=SEED)


def _build_pipeline(clf):
    return Pipeline([("pre", build_preprocessor()), ("clf", clf)])


def quick_train(
    root: Path,
    progress: Optional[Callable[[float, str], None]] = None,
) -> dict:
    """
    Train all models with sensible defaults and save artifacts.

    Args:
        root: Project root path. Artifacts saved under root/models and root/data.
        progress: Optional callback (fraction in [0,1], message) for UI updates.

    Returns:
        Dict with keys: best_pipe, all_pipes, metrics, df.
    """
    def _step(frac: float, msg: str):
        if progress is not None:
            progress(frac, msg)

    t0 = time.time()
    np.random.seed(SEED)

    models_dir = root / "models"
    data_dir = root / "data"
    models_dir.mkdir(exist_ok=True)
    data_dir.mkdir(exist_ok=True)

    # ── 1. Data ────────────────────────────────────────────────────────────
    _step(0.05, "Generating synthetic data…")
    data_path = data_dir / "loan_data.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
    else:
        df = generate_loan_data(n=N_RECORDS, seed=SEED)
        df.to_csv(data_path, index=False)

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES
    X = df[feature_cols]
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    # ── 2. Train base models ──────────────────────────────────────────────
    base_specs = {
        "logreg": _make_logreg,
        "random_forest": _make_rf,
        "xgboost": _make_xgb,
        "lightgbm": _make_lgbm,
    }
    pipelines = {}
    progress_per_model = 0.6 / len(base_specs)
    cur = 0.10
    for name, factory in base_specs.items():
        _step(cur, f"Training {name}…")
        clf = _safe_make(factory, name)
        if clf is None:
            cur += progress_per_model
            continue
        try:
            pipe = _build_pipeline(clf)
            pipe.fit(X_train, y_train)
            pipelines[name] = pipe
        except Exception as e:
            print(f"⚠️  Skipping {name} (fit failed): {type(e).__name__}: {e}")
        cur += progress_per_model

    if not pipelines:
        raise RuntimeError("No models could be trained — every base learner failed.")

    # ── 3. Stacking ───────────────────────────────────────────────────────
    _step(0.75, "Training stacking ensemble…")
    candidate_estimators = [
        ("rf", _safe_make(_make_rf, "random_forest")),
        ("xgb", _safe_make(_make_xgb, "xgboost")),
        ("lgbm", _safe_make(_make_lgbm, "lightgbm")),
    ]
    estimators = [(n, c) for n, c in candidate_estimators if c is not None]
    if len(estimators) >= 2:
        try:
            stack = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
                cv=5, stack_method="predict_proba", n_jobs=-1,
            )
            stack_pipe = _build_pipeline(stack)
            stack_pipe.fit(X_train, y_train)
            pipelines["stacking"] = stack_pipe
        except Exception as e:
            print(f"⚠️  Skipping stacking: {type(e).__name__}: {e}")
    else:
        print("⚠️  Skipping stacking — fewer than 2 tree models available.")

    # ── 4. Evaluate ───────────────────────────────────────────────────────
    _step(0.85, "Evaluating on test set…")
    results = {}
    for name, pipe in pipelines.items():
        y_proba = pipe.predict_proba(X_test)[:, 1]
        m = compute_metrics(y_test, y_proba)
        results[name] = {"y_proba": y_proba, "metrics": m}

    # ── 5. McNemar pairwise ───────────────────────────────────────────────
    _step(0.92, "Running McNemar tests…")
    mcnemar_table = []
    names = list(results.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pred_a = (results[a]["y_proba"] >= 0.5).astype(int)
            pred_b = (results[b]["y_proba"] >= 0.5).astype(int)
            stat, p = mcnemar_test(y_test, pred_a, pred_b)
            mcnemar_table.append(
                {"model_a": a, "model_b": b, "chi2": float(stat), "p_value": float(p)}
            )

    # ── 6. Best model + threshold ─────────────────────────────────────────
    _step(0.96, "Selecting best model…")
    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    best_pipe = pipelines[best_name]
    best_proba = results[best_name]["y_proba"]
    opt_thresh, _ = find_optimal_threshold(y_test, best_proba, cost_fp=1.0, cost_fn=3.0)
    metrics_at_opt = compute_metrics(y_test, best_proba, threshold=opt_thresh)

    # ── 7. Save ───────────────────────────────────────────────────────────
    _step(0.98, "Saving artifacts…")
    joblib.dump(best_pipe, models_dir / "best_model.pkl")
    joblib.dump(pipelines, models_dir / "all_pipelines.pkl")

    output = {
        "best_model": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "approval_rate_train": float(y_train.mean()),
        "approval_rate_test": float(y_test.mean()),
        "models": {
            name: {
                "metrics": {k: float(v) for k, v in r["metrics"].items()},
                "params": None,
                "cv_auc": None,
            }
            for name, r in results.items()
        },
        "optimal_threshold": float(opt_thresh),
        "metrics_at_optimal_threshold": {k: float(v) for k, v in metrics_at_opt.items()},
        "mcnemar_pairwise": mcnemar_table,
        "feature_columns": list(X.columns),
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
        "trained_via": "quick_train",
        "training_seconds": float(time.time() - t0),
    }
    with open(models_dir / "metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    _step(1.0, f"Done in {time.time() - t0:.1f}s.")

    return {
        "best_pipe": best_pipe,
        "all_pipes": pipelines,
        "metrics": output,
        "df": df,
    }
