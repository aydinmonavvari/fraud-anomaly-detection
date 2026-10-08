"""Evaluation metrics and decision-layer analysis for imbalanced detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)


def discrimination_metrics(y_true, y_score) -> dict[str, float]:
    """PR-AUC (primary under imbalance) and ROC-AUC (secondary, read with care)."""
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "base_rate": float(np.mean(y_true)),
    }


def precision_at_k(y_true, y_score, k_frac: float) -> dict[str, float]:
    """Precision/recall when investigating the top k% of scored cases.

    This is how fraud operations actually consume scores: a team reviews the
    highest-scoring transactions within its capacity budget.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    k = max(1, int(round(len(y_true) * k_frac)))
    top_idx = np.argsort(-y_score)[:k]
    precision = float(y_true[top_idx].mean())
    recall = float(y_true[top_idx].sum() / max(y_true.sum(), 1))
    return {
        f"precision_at_{int(k_frac * 100)}pct": precision,
        f"recall_at_{int(k_frac * 100)}pct": recall,
    }


def confusion_at_threshold(y_true, y_score, threshold: float) -> dict[str, int]:
    """Confusion entries at a decision threshold (labels [0, 1])."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def expected_cost(entries: dict[str, int], fn_cost: float, fp_cost: float) -> float:
    """Cost of missed fraud (FN) + falsely flagged legitimate work (FP)."""
    return float(entries["fn"] * fn_cost + entries["fp"] * fp_cost)


def cost_optimal_threshold(y_true, y_score, fn_cost: float, fp_cost: float) -> float:
    """The threshold on the PR curve minimizing expected cost.

    Ties and degenerate curves resolve to the lowest threshold that attains
    the best cost, i.e. the aggressive-flagging end - the correct bias for
    expensive missed fraud.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    best_cost, best_thr = float("inf"), float(thresholds[0])
    for _p, _r, t in zip(precision[:-1], recall[:-1], thresholds, strict=False):
        entries = confusion_at_threshold(y_true, y_score, float(t))
        cost = entries["fn"] * fn_cost + entries["fp"] * fp_cost
        if cost < best_cost:
            best_cost, best_thr = cost, float(t)
    return best_thr
