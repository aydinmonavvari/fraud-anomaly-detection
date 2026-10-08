"""Configuration for the fraud-detection study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class FraudConfig:
    """Configuration for one run of the fraud-detection study.

    Attributes
    ----------
    source:
        ``"synthetic"`` (default; fully reproducible, clearly labeled as
        synthetic) or ``"kaggle"`` (the ULB credit-card dataset; requires the
        user to place ``creditcard.csv`` in ``data/raw/`` themselves - see
        README section 7. The repository never redistributes that file).
    contamination:
        Expected fraud fraction used by Isolation Forest.
    test_size:
        Fraction held out (stratified) for final evaluation only; the test set
        is scored exactly once, at the validation-selected threshold.
    val_size:
        Fraction held out (stratified) for model-free decisions - i.e. the
        cost-optimal threshold. With the defaults, the three-way split is
        train 49% / validation 21% / test 30%.
    random_state:
        Global seed.
    fn_cost, fp_cost:
        Costs of missed fraud vs falsely flagged legitimate transaction.
    """

    source: str = "synthetic"
    contamination: float = 0.0017  # ~ the real dataset's 0.173% rate
    test_size: float = 0.30
    val_size: float = 0.21
    random_state: int = 42
    fn_cost: float = 10.0
    fp_cost: float = 1.0
    n_synthetic: int = 60_000

    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    processed_dir: Path = PROJECT_ROOT / "data" / "processed"
    figures_dir: Path = PROJECT_ROOT / "figures"
    reports_dir: Path = PROJECT_ROOT / "reports"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.figures_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)
