"""
Model definitions, hyperparameter search spaces, and Optuna-based tuning.

Why Optuna over GridSearchCV?
- Bayesian (TPE) sampling is more sample-efficient than grid for high-dim spaces.
- Easy pruning of unpromising trials.
- Same callback gives us a clean tuning history we can plot.
"""
from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

import lightgbm as lgb
import xgboost as xgb

# Optuna's default INFO logging is very chatty; quiet it for nice console output
optuna.logging.set_verbosity(optuna.logging.WARNING)


# -------------------- Objective functions --------------------------------- #

def _objective_logreg(trial, X_train, y_train, preprocessor, cv) -> float:
    # We omit `penalty` to use sklearn's default (deprecation warning in 1.8+
    # if set explicitly to 'l2'). Default solver 'lbfgs' supports L2.
    params = {
        "C": trial.suggest_float("C", 1e-3, 1e2, log=True),
        "max_iter": 2000,
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }
    pipe = Pipeline([("pre", preprocessor), ("clf", LogisticRegression(**params, random_state=42))])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def _objective_rf(trial, X_train, y_train, preprocessor, cv) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
        "class_weight": trial.suggest_categorical("class_weight", [None, "balanced"]),
    }
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(**params, random_state=42, n_jobs=-1)),
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def _objective_xgb(trial, X_train, y_train, preprocessor, cv) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", xgb.XGBClassifier(
            **params,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
            tree_method="hist",
        )),
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def _objective_lgbm(trial, X_train, y_train, preprocessor, cv) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
    }
    pipe = Pipeline([
        ("pre", preprocessor),
        ("clf", lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)),
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# -------------------- Public tuning API ----------------------------------- #

OBJECTIVES: Dict[str, Callable] = {
    "logreg": _objective_logreg,
    "random_forest": _objective_rf,
    "xgboost": _objective_xgb,
    "lightgbm": _objective_lgbm,
}


def tune_model(
    name: str,
    X_train,
    y_train,
    preprocessor,
    n_trials: int = 50,
    cv_splits: int = 5,
    seed: int = 42,
) -> Tuple[Dict, float]:
    """Run an Optuna study for the named model.

    Returns the best params and the best CV ROC-AUC.
    """
    if name not in OBJECTIVES:
        raise ValueError(f"Unknown model {name!r}. Choose from {list(OBJECTIVES)}")

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    sampler = TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: OBJECTIVES[name](trial, X_train, y_train, preprocessor, cv),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study.best_params, study.best_value


# -------------------- Build final pipelines ------------------------------- #

def build_pipeline(name: str, params: Dict, preprocessor) -> Pipeline:
    """Build a fitted-ready Pipeline (preprocessor + classifier) for the named model."""
    if name == "logreg":
        clf = LogisticRegression(**params, max_iter=2000, random_state=42)
    elif name == "random_forest":
        clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
    elif name == "xgboost":
        clf = xgb.XGBClassifier(
            **params, random_state=42, n_jobs=-1,
            eval_metric="logloss", tree_method="hist",
        )
    elif name == "lightgbm":
        clf = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1, verbose=-1)
    else:
        raise ValueError(f"Unknown model {name!r}")

    return Pipeline([("pre", preprocessor), ("clf", clf)])


def build_stacking(rf_params, xgb_params, lgbm_params, preprocessor) -> Pipeline:
    """LR meta-learner over tuned RF + XGB + LGBM."""
    estimators = [
        ("rf", RandomForestClassifier(**rf_params, random_state=42, n_jobs=-1)),
        ("xgb", xgb.XGBClassifier(
            **xgb_params, random_state=42, n_jobs=-1,
            eval_metric="logloss", tree_method="hist",
        )),
        ("lgbm", lgb.LGBMClassifier(**lgbm_params, random_state=42, n_jobs=-1, verbose=-1)),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=2000, random_state=42),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    return Pipeline([("pre", preprocessor), ("clf", stack)])
