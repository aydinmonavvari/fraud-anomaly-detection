"""Tests for the fraud-detection study (offline, synthetic data only)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import roc_auc_score

from fraud_anomaly_detection import pipeline as pipeline_module
from fraud_anomaly_detection.config import FraudConfig
from fraud_anomaly_detection.data import load_data, make_synthetic
from fraud_anomaly_detection.evaluation import (
    confusion_at_threshold,
    cost_optimal_threshold,
    discrimination_metrics,
    expected_cost,
    precision_at_k,
)

ALL_COLUMNS = [f"V{i}" for i in range(1, 29)] + ["Amount"]


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


# ---------------------------------------------------------------------------
# Stealth-fraud generator properties (data.py contract)
# ---------------------------------------------------------------------------

def _stealth_sample(seed: int = 42):
    """Enriched-stealth sample: ~3,000 stealth rows for statistical power.

    The committed configuration (~0.17% fraud, 15% stealth) leaves only ~15
    stealth rows - far too few to test a distributional property. The property
    under test (stealth rows are i.i.d. legitimate draws) is independent of
    the contamination/stealth-fraction settings, so the tests use a sample
    enriched in stealth rows and the same fixed seed for reproducibility.
    """
    return make_synthetic(
        n=200_000,
        contamination=0.03,
        stealth_fraction=0.5,
        seed=seed,
        return_stealth_mask=True,
    )


def test_stealth_rows_match_legitimate_distribution():
    """Stealth fraud rows must be statistically indistinguishable from
    legitimate rows on EVERY column (features + Amount)."""
    frame, stealth_mask = _stealth_sample()
    n_stealth = int(stealth_mask.sum())
    assert n_stealth > 1_000, "test needs a large stealth sample for power"

    legit = frame.loc[frame["Class"] == 0, ALL_COLUMNS]
    stealth = frame.loc[stealth_mask, ALL_COLUMNS]

    # 29 simultaneous KS tests: the per-column threshold is Bonferroni-corrected
    # (family-wise alpha = 0.01). An uncorrected per-column p > 0.01 would fail
    # ~25% of the time even when every column is drawn from the same
    # distribution - the correction is what makes this guard robust.
    ks_alpha = 0.01 / len(ALL_COLUMNS)

    for col in ALL_COLUMNS:
        a = stealth[col].to_numpy()
        b = legit[col].to_numpy()
        smd = abs(a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
        p = stats.ks_2samp(a, b).pvalue
        assert smd < 0.1, f"stealth vs legit |SMD| too large for {col}: {smd:.3f}"
        assert p > ks_alpha, (
            f"stealth vs legit distribution differs for {col}: KS p={p:.2e}"
        )


def test_stealth_univariate_detectability_is_capped():
    """No single feature separates stealth frauds from legitimate rows:
    univariate AUC must be ~0.5 (within chance range at this seed)."""
    frame, stealth_mask = _stealth_sample()
    keep = (frame["Class"] == 0) | stealth_mask
    y = frame.loc[keep, "Class"].to_numpy()  # stealth rows are Class==1
    for col in ("V1", "Amount"):
        auc = roc_auc_score(y, frame.loc[keep, col].to_numpy())
        assert 0.45 <= auc <= 0.55, f"univariate AUC for {col} is {auc:.3f}, not ~0.5"


# ---------------------------------------------------------------------------
# Three-way split / threshold-selection guard (pipeline.py contract)
# ---------------------------------------------------------------------------

def test_threshold_is_never_selected_on_test(monkeypatch, tmp_path):
    """The cost-optimal threshold must be computed from VALIDATION labels and
    scores - never from the test split - and the recorded metadata must
    reflect the validation-selected threshold."""
    n = 8_000  # -> stratified 49/21/30 split: 3,920 / 1,680 / 2,400
    config = FraudConfig(
        n_synthetic=n,
        raw_dir=tmp_path / "raw",
        processed_dir=tmp_path / "processed",
        figures_dir=tmp_path / "figures",
        reports_dir=tmp_path / "reports",
    )

    calls: list[dict] = []
    real_threshold = cost_optimal_threshold  # module-level binding = real one

    def spy(y_true, y_score, fn_cost, fp_cost):
        calls.append(
            {
                "y_true": np.asarray(y_true).copy(),
                "y_score": np.asarray(y_score).copy(),
            }
        )
        return real_threshold(y_true, y_score, fn_cost, fp_cost)

    monkeypatch.setattr(pipeline_module, "cost_optimal_threshold", spy)
    result = pipeline_module.run_pipeline(config, save_outputs=False)
    monkeypatch.undo()

    # one threshold selection per scored model (IF + LR + RF + GB = 4)
    assert len(calls) == len(result["metrics"]) == 4

    # every selection ran on the validation labels, never on the test labels
    y_val = np.asarray(result["y_val"])
    y_test = np.asarray(result["y_test"])
    assert len(y_val) == 1_680 and len(y_test) == 2_400  # validation-sized arrays
    for call in calls:
        assert np.array_equal(call["y_true"], y_val)
        assert not np.array_equal(call["y_true"], y_test)

    # metadata: recorded threshold == threshold recomputed from validation data
    for row in result["metrics"].reset_index().to_dict("records"):
        name = row["model"]
        assert row["threshold_source"] == "validation"
        assert result["split_sizes"]["threshold_source"] == "validation"
        val_thr = real_threshold(
            result["y_val"], result["val_scores"][name], config.fn_cost, config.fp_cost
        )
        assert row["cost_threshold"] == pytest.approx(val_thr)

    # a hypothetical (wrong) test-selected threshold would differ from the
    # recorded validation-selected one for at least one model at this seed -
    # the metadata demonstrably distinguishes the two procedures
    test_selected = {
        name: real_threshold(result["y_test"], score, config.fn_cost, config.fp_cost)
        for name, score in result["scores"].items()
    }
    recorded = result["metrics"]["cost_threshold"].to_dict()
    assert any(
        test_selected[name] != pytest.approx(thr) for name, thr in recorded.items()
    ), "test-selected thresholds coincidentally all equal validation-selected"
