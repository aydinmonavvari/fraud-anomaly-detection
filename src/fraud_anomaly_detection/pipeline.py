"""End-to-end orchestration for the fraud-detection study."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split

from .config import PROJECT_ROOT, FraudConfig
from .data import load_data
from .evaluation import (
    confusion_at_threshold,
    cost_optimal_threshold,
    discrimination_metrics,
    expected_cost,
    precision_at_k,
)
from .models import (
    fit_isolation_forest,
    isolation_scores,
    make_supervised,
)

logger = logging.getLogger(__name__)


def run_pipeline(config: FraudConfig | None = None, save_outputs: bool = True) -> dict:
    config = config or FraudConfig()
    config.ensure_dirs()

    frame, provenance = load_data(config)
    logger.info("source: %s | rows=%d fraud_rate=%.4f%%",
                provenance, len(frame), 100 * frame["Class"].mean())

    X = frame.drop(columns=["Class"])
    y = frame["Class"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, stratify=y, random_state=config.random_state
    )

    scores: dict[str, np.ndarray] = {}
    fallbacks: dict[str, str] = {}

    # ---- unsupervised: Isolation Forest (never sees labels) ----------------
    iso = fit_isolation_forest(X_train, config.contamination, config.random_state)
    scores["isolation_forest"] = isolation_scores(iso, X_test)

    # ---- supervised --------------------------------------------------------
    supervised = make_supervised(config.random_state)
    for name in ("logistic_regression", "random_forest", "gradient_boosting"):
        try:
            model = supervised[name]
            model.fit(X_train, y_train)
            scores[name] = model.predict_proba(X_test)[:, 1]
        except Exception as exc:  # record failure honestly, keep others running
            logger.error("model %s failed: %s", name, exc)
            fallbacks[name] = str(exc)

    # ---- evaluation --------------------------------------------------------
    metric_rows = []
    for name, y_score in scores.items():
        metrics = discrimination_metrics(y_test, y_score)
        metrics.update(precision_at_k(y_test, y_score, 0.001))
        metrics.update(precision_at_k(y_test, y_score, 0.01))
        thr = cost_optimal_threshold(y_test, y_score, config.fn_cost, config.fp_cost)
        entries = confusion_at_threshold(y_test, y_score, thr)
        metrics.update(
            {
                "model": name,
                "cost_threshold": thr,
                "expected_cost": expected_cost(entries, config.fn_cost, config.fp_cost),
                **{f"{k}_at_thr": v for k, v in entries.items()},
            }
        )
        metric_rows.append(metrics)
    metrics_tbl = pd.DataFrame(metric_rows).set_index("model")

    if save_outputs:
        _save(config, metrics_tbl, scores, y_test, provenance, fallbacks)

    return {"metrics": metrics_tbl, "provenance": provenance, "scores": scores,
            "y_test": y_test, "fallbacks": fallbacks}


def _save(config, metrics_tbl, scores, y_test, provenance, fallbacks) -> None:
    payload = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "project": "fraud-anomaly-detection",
        "provenance": provenance,
        "fallbacks": fallbacks,
        "config": {
            k: (str(v).replace(str(PROJECT_ROOT) + "/", "") if isinstance(v, Path) else v)
            for k, v in asdict(config).items()
        },
        "metrics": metrics_tbl.round(6).reset_index().to_dict("records"),
    }
    (config.reports_dir / "fraud_results.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    metrics_tbl.round(6).to_csv(config.reports_dir / "fraud_metrics.csv")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, y_score in scores.items():
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        ax[0].plot(recall, precision, lw=1.7, label=name)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        ax[1].plot(fpr, tpr, lw=1.7, label=name)
    base = float(np.mean(y_test))
    ax[0].axhline(base, color="grey", ls="--", lw=0.9, label=f"base rate {base:.4f}")
    ax[0].set_title("Precision-Recall (the honest curve under imbalance)")
    ax[0].set_xlabel("recall")
    ax[0].set_ylabel("precision")
    ax[0].legend()
    ax[1].plot([0, 1], [0, 1], color="grey", ls="--", lw=0.9)
    ax[1].set_title("ROC (flattering under imbalance - read with care)")
    ax[1].set_xlabel("false positive rate")
    ax[1].set_ylabel("true positive rate")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(config.figures_dir / "pr_roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    metric_tbl_plot = metrics_tbl[["pr_auc", "roc_auc"]].sort_values("pr_auc")
    metric_tbl_plot.plot(kind="barh", ax=ax)
    ax.set_title("PR-AUC vs ROC-AUC under extreme imbalance")
    ax.set_xlabel("score")
    fig.tight_layout()
    fig.savefig(config.figures_dir / "metric_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    run_pipeline()
    print("done")
