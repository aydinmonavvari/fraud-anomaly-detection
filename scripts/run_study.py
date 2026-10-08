"""CLI: run the full fraud-detection study.

Default source is the clearly-labeled synthetic generator (no network, no
gated data). To run on the real ULB dataset, download creditcard.csv from
Kaggle (mlg-ulb/creditcardfraud) into data/raw/ and pass --source kaggle.
"""

import argparse
import logging

from fraud_anomaly_detection.config import FraudConfig
from fraud_anomaly_detection.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["synthetic", "kaggle"], default="synthetic")
    parser.add_argument("--n-synthetic", type=int, default=60_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = FraudConfig(source=args.source, n_synthetic=args.n_synthetic)
    results = run_pipeline(config)
    print("\n=== metrics (provenance:", results["provenance"], ") ===")
    print(results["metrics"].round(4).to_string())


if __name__ == "__main__":
    main()
