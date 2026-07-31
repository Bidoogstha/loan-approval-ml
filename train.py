"""
Main training pipeline.

Run:
    python train.py

What it does:
    1. Generates synthetic loan data (or loads it if data/loan_data.csv exists).
    2. Stratified train/test split.
    3. Builds a preprocessing ColumnTransformer.
    4. Tunes Logistic Regression, Random Forest, XGBoost, LightGBM with Optuna.
    5. Builds a stacking ensemble from the tuned base learners.
    6. Evaluates everything on the held-out test set.
    7. Computes pairwise McNemar tests.
    8. Picks the best model by ROC-AUC, computes SHAP, finds optimal threshold.
    9. Saves: best_model.pkl, preprocessor.pkl, metrics.json, plots.

The script is verbose on purpose — for a portfolio piece you want the
console output to read like a story you can show off.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

from src.data_loader import (
    CATEGORICAL_FEATURES,
    ENGINEERED_FEATURES,
    NUMERICAL_FEATURES,
    TARGET,
    TEMPORAL_FEATURES,
    load_lending_club,
)
from src.preprocessing import build_preprocessor, get_feature_names
from src.models import tune_model, build_pipeline, build_stacking
from src.evaluation import (
    compute_metrics,
    find_optimal_threshold,
    mcnemar_test,
    plot_calibration,
    plot_confusion_matrices,
    plot_metrics_comparison,
    plot_pr_curves,
    plot_roc_curves,
)

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.20
N_TRIALS = 50  # Optuna trials per model — drop to 15-25 for a quick run
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "loan_data.csv"
MODELS_DIR = ROOT / "models"
IMAGES_DIR = ROOT / "images"
MODELS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)


def _section(title: str) -> None:
    print(f"\n{'─' * 70}\n {title}\n{'─' * 70}")


def main():
    t0 = time.time()
    np.random.seed(SEED)

    # ---------------------------------------------------------------- DATA
    _section("1. Loading data")
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        print(f"Loaded {DATA_PATH} ({len(df):,} rows)")
    else:
        df = load_lending_club()
        DATA_PATH.parent.mkdir(exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        print(f"Generated and saved {len(df):,} rows → {DATA_PATH}")

    print(f"Default rate: {df[TARGET].mean():.1%}")
    print(f"Class distribution:\n{df[TARGET].value_counts().to_string()}")

    feature_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + ENGINEERED_FEATURES
    X = df[feature_cols]
    y = df[TARGET].values

    # ------------------------------------------------------------- SPLIT
    _section("2. Train/test split (temporal, 80/20 by issue date)")
    df_sorted = df.sort_values("issue_d").reset_index(drop=True)
    cutoff_idx = int(len(df_sorted) * (1 - TEST_SIZE))
    train_df = df_sorted.iloc[:cutoff_idx]
    test_df = df_sorted.iloc[cutoff_idx:]

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_train = train_df[TARGET].values
    y_test = test_df[TARGET].values

    print(f"Train: {len(X_train):,} loans, issued "
          f"{train_df['issue_d'].min().date()} to {train_df['issue_d'].max().date()} "
          f"({y_train.mean():.1%} default rate)")
    print(f"Test : {len(X_test):,} loans, issued "
          f"{test_df['issue_d'].min().date()} to {test_df['issue_d'].max().date()} "
          f"({y_test.mean():.1%} default rate)")

    # ------------------------------------------------------ PREPROCESSOR
    preprocessor = build_preprocessor()

    # -------------------------------------------------------- TUNE BASES
    _section(f"3. Hyperparameter tuning ({N_TRIALS} Optuna trials per model)")
    base_results = {}
    for name in ["logreg", "random_forest", "xgboost", "lightgbm"]:
        t = time.time()
        print(f"\n→ Tuning {name} ...")
        best_params, best_cv = tune_model(
            name, X_train, y_train, build_preprocessor(),
            n_trials=N_TRIALS, seed=SEED,
        )
        print(f"  best CV ROC-AUC = {best_cv:.4f}   (took {time.time()-t:.1f}s)")
        print(f"  params = {best_params}")
        base_results[name] = {"params": best_params, "cv_auc": best_cv}

    # -------------------------------------------------------- FIT FINAL
    _section("4. Fitting final pipelines on full training set")
    pipelines = {}
    for name, info in base_results.items():
        pipe = build_pipeline(name, info["params"], build_preprocessor())
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe
        print(f"  fitted: {name}")

    # Stacking ensemble — uses the already-tuned hyperparameters
    print("  fitting: stacking (RF + XGB + LGBM with LR meta-learner) ...")
    stack_pipe = build_stacking(
        rf_params=base_results["random_forest"]["params"],
        xgb_params=base_results["xgboost"]["params"],
        lgbm_params=base_results["lightgbm"]["params"],
        preprocessor=build_preprocessor(),
    )
    stack_pipe.fit(X_train, y_train)
    pipelines["stacking"] = stack_pipe

    # ---------------------------------------------------------- EVALUATE
    _section("5. Test-set evaluation")
    results = {}
    for name, pipe in pipelines.items():
        y_proba = pipe.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_proba)
        results[name] = {"y_proba": y_proba, "metrics": metrics, "pipeline": pipe}
        print(f"  {name:14s}  AUC={metrics['roc_auc']:.4f}  "
              f"PR-AUC={metrics['pr_auc']:.4f}  F1={metrics['f1']:.4f}  "
              f"Brier={metrics['brier']:.4f}")

    # ---------------------------------------------- McNemar comparisons
    _section("6. Pairwise McNemar tests (statistical significance)")
    names = list(results.keys())
    mcnemar_table = []
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            pred_a = (results[a]["y_proba"] >= 0.5).astype(int)
            pred_b = (results[b]["y_proba"] >= 0.5).astype(int)
            stat, p = mcnemar_test(y_test, pred_a, pred_b)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            print(f"  {a:14s} vs {b:14s}  χ²={stat:6.2f}  p={p:.4f}  {sig}")
            mcnemar_table.append({"model_a": a, "model_b": b, "chi2": stat, "p_value": p})

    # ----------------------------------------------------- BEST MODEL
    _section("7. Selecting best model and finding optimal threshold")
    best_name = max(results, key=lambda k: results[k]["metrics"]["roc_auc"])
    best_pipe = results[best_name]["pipeline"]
    best_proba = results[best_name]["y_proba"]
    print(f"Best model: {best_name}")

    # Cost-optimal threshold (FN 3x as costly as FP — adjust for your business)
    opt_thresh, opt_cost = find_optimal_threshold(y_test, best_proba, cost_fp=1.0, cost_fn=3.0)
    metrics_at_opt = compute_metrics(y_test, best_proba, threshold=opt_thresh)
    print(f"Optimal threshold (cost FN=3, FP=1): {opt_thresh:.2f}")
    print(f"  At t=0.50: F1={results[best_name]['metrics']['f1']:.4f}  "
          f"Recall={results[best_name]['metrics']['recall']:.4f}")
    print(f"  At t={opt_thresh:.2f}: F1={metrics_at_opt['f1']:.4f}  "
          f"Recall={metrics_at_opt['recall']:.4f}")

    # ------------------------------------------------------------- PLOTS
    _section("8. Generating plots")
    plot_roc_curves(results, y_test, IMAGES_DIR / "roc_curves.png")
    plot_pr_curves(results, y_test, IMAGES_DIR / "pr_curves.png")
    plot_calibration(results, y_test, IMAGES_DIR / "calibration.png")
    plot_confusion_matrices(results, y_test, IMAGES_DIR / "confusion_matrices.png")
    plot_metrics_comparison(results, IMAGES_DIR / "metrics_comparison.png")
    print(f"  plots → {IMAGES_DIR}")

    # -------------------------------------------- FEATURE IMPORTANCE
    _section("9. Feature importance — permutation + SHAP")

    # Permutation importance is model-agnostic — works for any pipeline
    print("Computing permutation importance (this is slow)...")
    perm = permutation_importance(
        best_pipe, X_test, y_test, n_repeats=10, random_state=SEED,
        n_jobs=-1, scoring="roc_auc",
    )
    perm_df = (
        pd.DataFrame({
            "feature": X_test.columns,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        })
        .sort_values("importance_mean", ascending=False)
    )
    print(perm_df.to_string(index=False))
    perm_df.to_csv(MODELS_DIR / "permutation_importance.csv", index=False)

    # Plot it
    plt.figure(figsize=(9, 6))
    top = perm_df.head(15).iloc[::-1]
    plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"], color="#0ea5e9")
    plt.xlabel("Permutation importance (drop in ROC-AUC)")
    plt.title(f"Permutation Importance — {best_name}")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / "permutation_importance.png", dpi=120, bbox_inches="tight")
    plt.close()

    # SHAP — only meaningful for tree models
    if best_name in ("random_forest", "xgboost", "lightgbm"):
        print("\nComputing SHAP values...")
        # Transform once so we can pass a numerical matrix to TreeExplainer
        pre = best_pipe.named_steps["pre"]
        clf = best_pipe.named_steps["clf"]
        X_test_transformed = pre.transform(X_test)
        feature_names = get_feature_names(pre)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_transformed)

        # SHAP output format depends on version:
        # - Old (≤0.43): list of 2D arrays, one per class. Take index 1 for class 1.
        # - New (≥0.45): single 3D array (n_samples, n_features, n_classes). Slice [..., 1].
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif shap_values.ndim == 3:
            shap_values = shap_values[..., 1]

        # Global summary plot
        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values, X_test_transformed, feature_names=feature_names,
            show=False, plot_size=None,
        )
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values, X_test_transformed, feature_names=feature_names,
            plot_type="bar", show=False,
        )
        plt.tight_layout()
        plt.savefig(IMAGES_DIR / "shap_importance.png", dpi=120, bbox_inches="tight")
        plt.close()

        # Save mean |SHAP| for the app
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values("mean_abs_shap", ascending=False)
        shap_df.to_csv(MODELS_DIR / "shap_importance.csv", index=False)
        print(f"  SHAP plots → {IMAGES_DIR}")
    else:
        print(f"  Skipping SHAP — best model ({best_name}) is not a tree model.")

    # --------------------------------------------------------------- SAVE
    _section("10. Saving artifacts")
    joblib.dump(best_pipe, MODELS_DIR / "best_model.pkl")
    joblib.dump(pipelines, MODELS_DIR / "all_pipelines.pkl")
    print(f"  best model: {MODELS_DIR / 'best_model.pkl'}")
    print(f"  all pipelines: {MODELS_DIR / 'all_pipelines.pkl'}")

    output = {
        "best_model": best_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "approval_rate_train": float(y_train.mean()),
        "approval_rate_test": float(y_test.mean()),
        "models": {
            name: {
                "metrics": {k: float(v) for k, v in r["metrics"].items()},
                "params": base_results[name]["params"] if name in base_results else None,
                "cv_auc": base_results[name]["cv_auc"] if name in base_results else None,
            }
            for name, r in results.items()
        },
        "optimal_threshold": opt_thresh,
        "metrics_at_optimal_threshold": {k: float(v) for k, v in metrics_at_opt.items()},
        "mcnemar_pairwise": mcnemar_table,
        "feature_columns": list(X.columns),
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "engineered_features": ENGINEERED_FEATURES,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"  metrics: {MODELS_DIR / 'metrics.json'}")

    elapsed = time.time() - t0
    _section(f"Done in {elapsed:.1f}s")
    print(f"Best model: {best_name}  (ROC-AUC = {results[best_name]['metrics']['roc_auc']:.4f})")
    print(f"\nLaunch the app:  streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
