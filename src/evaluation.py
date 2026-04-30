"""
Evaluation utilities: metrics, plots, calibration, threshold optimization,
and statistical comparison of models (McNemar's test).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute all the metrics we report. y_proba is P(approved=1)."""
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "brier": brier_score_loss(y_true, y_proba),
        "threshold": threshold,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    cost_fp: float = 1.0,
    cost_fn: float = 3.0,
) -> Tuple[float, float]:
    """Find threshold minimizing expected cost.

    Default cost ratio (FN three times as costly as FP) reflects a typical
    lender perspective where wrongly approving a defaulter costs more than
    wrongly denying a good borrower. ADJUST FOR YOUR SETTING — these are
    business choices, not statistical ones.
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    costs = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        costs.append(cost_fp * fp + cost_fn * fn)
    best_idx = int(np.argmin(costs))
    return float(thresholds[best_idx]), float(costs[best_idx])


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Tuple[float, float]:
    """McNemar's test for paired classifier comparison.

    Returns (chi-square statistic with continuity correction, p-value).
    Reference: McNemar, Q. (1947). Note on the sampling error of the
    difference between correlated proportions or percentages.
    """
    a_correct = (pred_a == y_true)
    b_correct = (pred_b == y_true)
    n01 = int(np.sum(a_correct & ~b_correct))   # A right, B wrong
    n10 = int(np.sum(~a_correct & b_correct))   # A wrong, B right
    if n01 + n10 == 0:
        return 0.0, 1.0
    stat = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    p = 1 - chi2.cdf(stat, df=1)
    return float(stat), float(p)


# -------------------- Plotting helpers ------------------------------------ #

def plot_roc_curves(results: Dict[str, Dict], y_test: np.ndarray, save_path: Path) -> None:
    """One ROC plot with all models overlaid."""
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {res['metrics']['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Test Set")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_pr_curves(results: Dict[str, Dict], y_test: np.ndarray, save_path: Path) -> None:
    """Precision-Recall curves — more informative than ROC under class imbalance."""
    plt.figure(figsize=(8, 6))
    baseline = y_test.mean()
    for name, res in results.items():
        prec, rec, _ = precision_recall_curve(y_test, res["y_proba"])
        plt.plot(rec, prec, lw=2, label=f"{name} (AP = {res['metrics']['pr_auc']:.3f})")
    plt.axhline(baseline, color="k", linestyle="--", lw=1, alpha=0.5,
                label=f"Baseline ({baseline:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves — Test Set")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_calibration(results: Dict[str, Dict], y_test: np.ndarray, save_path: Path) -> None:
    """Reliability diagram. A well-calibrated model lies on the diagonal."""
    plt.figure(figsize=(8, 6))
    for name, res in results.items():
        prob_true, prob_pred = calibration_curve(y_test, res["y_proba"], n_bins=10, strategy="quantile")
        plt.plot(prob_pred, prob_true, marker="o", lw=2,
                 label=f"{name} (Brier = {res['metrics']['brier']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration (Reliability) Curves")
    plt.legend(loc="upper left")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_confusion_matrices(results: Dict[str, Dict], y_test: np.ndarray, save_path: Path) -> None:
    """Grid of confusion matrices, one per model."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        y_pred = (res["y_proba"] >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Denied", "Approved"], yticklabels=["Denied", "Approved"],
                    cbar=False)
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(results: Dict[str, Dict], save_path: Path) -> None:
    """Bar chart of headline metrics across models."""
    metrics_to_plot = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    rows = []
    for name, res in results.items():
        for m in metrics_to_plot:
            rows.append({"model": name, "metric": m, "value": res["metrics"][m]})
    df = pd.DataFrame(rows)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="metric", y="value", hue="model")
    plt.title("Test-Set Metrics by Model")
    plt.ylim(0, 1)
    plt.legend(loc="lower right", ncol=2)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
