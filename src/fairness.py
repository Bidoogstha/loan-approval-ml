"""
Fairness metrics for the loan default prediction model.

We measure three distinct notions of fairness, each capturing a different
concern. They are mathematically incompatible in general (Pleiss et al.,
NeurIPS 2017) — a real model usually has to trade them off:

1. Demographic parity (a.k.a. statistical parity)
   "Does the model predict 'approve' at the same rate across groups?"
   Origin: Disparate Impact doctrine (US Supreme Court, Griggs v. Duke
   Power, 1971; codified by EEOC's four-fifths rule, 1978).

2. Equalized odds
   "Among people who actually paid back, are they equally likely to be
   approved across groups? Among people who actually defaulted, are they
   equally likely to be denied?"
   Origin: Hardt, Price, Srebro, "Equality of Opportunity in Supervised
   Learning", NeurIPS 2016 (https://arxiv.org/abs/1610.02413).

3. Calibration by group
   "When the model says '20% chance of default', is it actually 20% for
   every group? Or is it 20% for one group and 35% for another?"
   Origin: Kleinberg, Mullainathan, Raghavan, "Inherent Trade-Offs in the
   Fair Determination of Risk Scores", ITCS 2017
   (https://arxiv.org/abs/1609.05807).

Each function takes y_true, y_pred (binary), y_proba (probabilities),
and a sensitive attribute series. Each returns a per-group breakdown
plus a single summary disparity metric.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


# ── Data containers ───────────────────────────────────────────────────────

@dataclass
class GroupMetrics:
    """Performance and rates for one subgroup (e.g. one state)."""
    group: str
    n: int
    base_rate: float           # actual default rate in this group
    selection_rate: float      # rate at which the model predicts "approve" (=1 - predicted-default rate)
    tpr: float                 # true positive rate (recall on the positive=default class)
    fpr: float                 # false positive rate
    mean_predicted_proba: float


@dataclass
class FairnessReport:
    """Full fairness audit result for one sensitive attribute."""
    attribute: str
    metric_name: str           # e.g. 'demographic_parity'
    groups: List[GroupMetrics] = field(default_factory=list)
    disparity: float = 0.0     # the headline number for this metric
    four_fifths_pass: bool = True
    notes: str = ""


# ── The three metrics ─────────────────────────────────────────────────────

def demographic_parity(
    y_pred: np.ndarray,
    sensitive: pd.Series,
    attribute_name: str,
    min_group_size: int = 100,
) -> FairnessReport:
    """
    Demographic parity: equal selection rate across groups.

    "Selection" here = predicted as a *good* loan (i.e. predicted not to
    default). Predicted-not-default = 1 - predicted-default. We invert
    because the model's positive class is "defaulted" and we want
    "approved-by-the-model" as the favorable outcome.

    Returns the disparity = max selection rate - min selection rate
    and the four-fifths-rule pass/fail.
    """
    df = pd.DataFrame({"pred": y_pred, "group": sensitive.values})
    df["approved_by_model"] = (df["pred"] == 0).astype(int)

    grouped = df.groupby("group").agg(
        n=("pred", "size"),
        selection_rate=("approved_by_model", "mean"),
    )
    grouped = grouped[grouped["n"] >= min_group_size]

    if len(grouped) < 2:
        return FairnessReport(
            attribute=attribute_name,
            metric_name="demographic_parity",
            notes=f"Fewer than 2 groups with ≥{min_group_size} samples — skipped.",
        )

    sel_rates = grouped["selection_rate"]
    disparity = float(sel_rates.max() - sel_rates.min())
    four_fifths_ratio = float(sel_rates.min() / sel_rates.max())

    groups = [
        GroupMetrics(
            group=str(g), n=int(row["n"]),
            base_rate=np.nan, selection_rate=float(row["selection_rate"]),
            tpr=np.nan, fpr=np.nan, mean_predicted_proba=np.nan,
        )
        for g, row in grouped.iterrows()
    ]

    return FairnessReport(
        attribute=attribute_name,
        metric_name="demographic_parity",
        groups=groups,
        disparity=disparity,
        four_fifths_pass=bool(four_fifths_ratio >= 0.80),
        notes=f"Selection-rate spread = {disparity:.3f}. "
              f"Four-fifths ratio = {four_fifths_ratio:.3f}.",
    )


def equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive: pd.Series,
    attribute_name: str,
    min_group_size: int = 100,
) -> FairnessReport:
    """
    Equalized odds: equal TPR AND equal FPR across groups.

    Hardt et al. 2016. Strictly stronger than demographic parity —
    it conditions on the true label, so it rules out 'over-denying
    qualified Group A people in order to balance predicted approval rates'.

    The disparity is the max TPR-gap or FPR-gap across all groups,
    whichever is larger.
    """
    df = pd.DataFrame({"y": y_true, "pred": y_pred, "group": sensitive.values})

    rows = []
    for g, sub in df.groupby("group"):
        if len(sub) < min_group_size:
            continue
        # Treat "default" (label=1) as the positive class for TPR/FPR.
        try:
            tn, fp, fn, tp = confusion_matrix(sub["y"], sub["pred"], labels=[0, 1]).ravel()
        except ValueError:
            continue
        tpr = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        fpr = fp / (fp + tn) if (fp + tn) > 0 else np.nan
        rows.append({
            "group": g, "n": len(sub),
            "tpr": tpr, "fpr": fpr,
            "base_rate": float(sub["y"].mean()),
            "selection_rate": float((sub["pred"] == 0).mean()),
        })

    if len(rows) < 2:
        return FairnessReport(
            attribute=attribute_name,
            metric_name="equalized_odds",
            notes=f"Fewer than 2 groups with ≥{min_group_size} samples — skipped.",
        )

    tprs = [r["tpr"] for r in rows if not np.isnan(r["tpr"])]
    fprs = [r["fpr"] for r in rows if not np.isnan(r["fpr"])]
    tpr_gap = max(tprs) - min(tprs) if tprs else 0.0
    fpr_gap = max(fprs) - min(fprs) if fprs else 0.0
    disparity = max(tpr_gap, fpr_gap)

    groups = [
        GroupMetrics(
            group=str(r["group"]), n=int(r["n"]),
            base_rate=r["base_rate"], selection_rate=r["selection_rate"],
            tpr=r["tpr"], fpr=r["fpr"], mean_predicted_proba=np.nan,
        )
        for r in rows
    ]

    return FairnessReport(
        attribute=attribute_name,
        metric_name="equalized_odds",
        groups=groups,
        disparity=float(disparity),
        four_fifths_pass=bool(disparity < 0.10),  # convention: under 10pp is "OK"
        notes=f"TPR gap = {tpr_gap:.3f}. FPR gap = {fpr_gap:.3f}.",
    )


def calibration_by_group(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    sensitive: pd.Series,
    attribute_name: str,
    min_group_size: int = 100,
) -> FairnessReport:
    """
    Calibration by group: in each group, does mean predicted probability
    match the actual default rate?

    Kleinberg-Mullainathan-Raghavan 2017. A model can be perfectly
    calibrated overall but miscalibrated for minority groups — e.g., it
    predicts '30% default' for a group whose actual rate is 45%.

    Disparity = max |mean_predicted - actual_rate| across groups.
    """
    df = pd.DataFrame({"y": y_true, "p": y_proba, "group": sensitive.values})

    rows = []
    for g, sub in df.groupby("group"):
        if len(sub) < min_group_size:
            continue
        rows.append({
            "group": g, "n": len(sub),
            "mean_predicted": float(sub["p"].mean()),
            "actual_rate": float(sub["y"].mean()),
            "gap": float(sub["p"].mean() - sub["y"].mean()),
        })

    if len(rows) < 2:
        return FairnessReport(
            attribute=attribute_name,
            metric_name="calibration_by_group",
            notes=f"Fewer than 2 groups with ≥{min_group_size} samples — skipped.",
        )

    max_gap = max(abs(r["gap"]) for r in rows)

    groups = [
        GroupMetrics(
            group=str(r["group"]), n=int(r["n"]),
            base_rate=r["actual_rate"],
            selection_rate=np.nan,
            tpr=np.nan, fpr=np.nan,
            mean_predicted_proba=r["mean_predicted"],
        )
        for r in rows
    ]

    return FairnessReport(
        attribute=attribute_name,
        metric_name="calibration_by_group",
        groups=groups,
        disparity=float(max_gap),
        four_fifths_pass=bool(max_gap < 0.05),  # within 5pp = well-calibrated
        notes=f"Max calibration gap = {max_gap:.3f}.",
    )


# ── Convenience: serialize a report to JSON ───────────────────────────────

def report_to_dict(report: FairnessReport) -> Dict:
    """Convert a FairnessReport to a JSON-serializable dict."""
    def _clean(v):
        if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
            return None
        return v

    return {
        "attribute": report.attribute,
        "metric_name": report.metric_name,
        "disparity": _clean(report.disparity),
        "four_fifths_pass": report.four_fifths_pass,
        "notes": report.notes,
        "groups": [
            {
                "group": g.group, "n": g.n,
                "base_rate": _clean(g.base_rate),
                "selection_rate": _clean(g.selection_rate),
                "tpr": _clean(g.tpr), "fpr": _clean(g.fpr),
                "mean_predicted_proba": _clean(g.mean_predicted_proba),
            }
            for g in report.groups
        ],
    }


"""
Audit results — Lending Club default model:

By state:
- Demographic parity disparity = 0.077 (PASS, four-fifths ratio 0.921)
- Calibration gap = 0.110 (FAIL — model probabilities are off by up to 11pp depending on state)
- Equalized-odds disparity = 0.280 (FAIL — TPR varies 28pp across states)

By income bracket:
- Demographic parity disparity = 0.047 (PASS)
- Calibration gap = 0.009 (PASS — well-calibrated by income)
- Equalized-odds disparity = 0.114 (FAIL — TPR varies 11pp by bracket)

Interpretation: the model passes demographic parity (similar approval rates
across groups) but fails equalized odds and group calibration. This is the
trade-off described in Pleiss et al. (NeurIPS 2017): no model can satisfy
demographic parity, equalized odds, and calibration simultaneously unless
it is perfectly accurate or the base rates are equal across groups.
"""