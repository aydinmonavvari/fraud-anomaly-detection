"""Model factories for the fraud-detection study."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_ORDER = (
    "isolation_forest",
    "logistic_regression",
    "random_forest",
    "gradient_boosting",
)


def make_supervised(seed: int) -> dict[str, Pipeline]:
    """Supervised models with class imbalance handled by class weights."""
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    GradientBoostingClassifier(
                        learning_rate=0.08,
                        max_depth=3,
                        n_estimators=250,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def fit_isolation_forest(X_train, contamination: float, seed: int) -> IsolationForest:
    """Fit the unsupervised anomaly detector on training features only
    (labels are never seen - that is the point of the unsupervised arm)."""
    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    iso.fit(X_train)
    return iso


def isolation_scores(iso: IsolationForest, X) -> np.ndarray:
    """Higher = more anomalous (negate sklearn's negative score)."""
    return -iso.score_samples(X)
