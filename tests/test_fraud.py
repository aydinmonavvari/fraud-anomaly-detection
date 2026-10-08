"""Tests for the fraud-detection study (offline, synthetic data only)."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_anomaly_detection.data import load_data, make_synthetic
from fraud_anomaly_detection.evaluation import (
    confusion_at_threshold,
    cost_optimal_threshold,
    discrimination_metrics,
    expected_cost,
    precision_at_k,
)


def test_synthetic_generator_shape_and_rate():
    frame = make_synthetic(n=20_000, contamination=0.002, seed=1)
    assert len(frame) == 20_000
    assert set(frame.columns) == {f"V{i}" for i in range(1, 29)} | {"Amount", "Class"}
    assert 0 < frame["Class"].mean() < 0.01
    assert frame["Class"].isin([0, 1]).all()


def test_synthetic_signal_exists():
    """Fraud rows must differ in the shifted components - otherwise the study
    would run on noise. V1..V3 are deterministic signal dims."""
    frame = make_synthetic(n=40_000, seed=7)
    legit = frame[frame["Class"] == 0]["V1"]
    fraud = frame[frame["Class"] == 1]["V1"]
    assert abs(fraud.mean() - legit.mean()) > 1.0, "fraud mean shift expected"
    assert fraud.std() > legit.std() * 0.5


def test_kaggle_mode_requires_user_file(tmp_path):
    cfg = tmp_path / "raw"
    cfg.mkdir()
    from fraud_anomaly_detection.config import FraudConfig

    config = FraudConfig(source="kaggle", raw_dir=cfg)
    with pytest.raises(FileNotFoundError, match="Kaggle"):
        load_data(config)


def test_pr_auc_and_base_rate():
    y = np.array([0, 0, 0, 1, 1])
    perfect = np.array([0.0, 0.1, 0.2, 0.8, 0.9])
    m = discrimination_metrics(y, perfect)
    assert m["pr_auc"] == pytest.approx(1.0)
    assert m["base_rate"] == pytest.approx(0.4)
    null = np.full(5, 0.5)
    m_null = discrimination_metrics(y, null)
    assert m_null["pr_auc"] == pytest.approx(0.4)  # = base rate for constant scores


def test_precision_at_k_math():
    y = np.array([1, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
    # top 10% = 1 case -> it is a fraud
    res = precision_at_k(y, score, 0.1)
    assert res["precision_at_10pct"] == 1.0
    assert res["recall_at_10pct"] == pytest.approx(0.5)
    # top 50% = 5 cases -> 2 frauds
    res5 = precision_at_k(y, score, 0.5)
    assert res5["precision_at_50pct"] == pytest.approx(0.4)
    assert res5["recall_at_50pct"] == pytest.approx(1.0)


def test_cost_optimal_threshold_prefers_missing_less_fraud():
    """With FN 10x FP the optimum must flag aggressively (low threshold)."""
    rng = np.random.default_rng(0)
    y = (rng.uniform(size=4000) < 0.02).astype(int)
    score = np.clip(rng.normal(0.3 * y + 0.2, 0.1), 0, 1)
    thr = cost_optimal_threshold(y, score, fn_cost=10.0, fp_cost=1.0)
    entries = confusion_at_threshold(y, score, thr)
    # at the optimum, recall must be substantial - a high threshold would be
    # dominated by simply flagging more cases
    recall = entries["tp"] / max(entries["tp"] + entries["fn"], 1)
    assert recall > 0.5
    cost = expected_cost(entries, fn_cost=10.0, fp_cost=1.0)
    higher = confusion_at_threshold(y, score, min(thr + 0.2, 0.99))
    assert cost <= expected_cost(higher, 10.0, 1.0) + 1e-9


def test_confusion_at_threshold_labels():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.6, 0.7, 0.9])
    e = confusion_at_threshold(y, score, 0.5)
    assert e == {"tn": 1, "fp": 1, "fn": 0, "tp": 2}
