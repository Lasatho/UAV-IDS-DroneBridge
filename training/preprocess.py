"""
Standalone data-preprocessing entry point.

Runs discovery -> mission-level split -> alignment -> normalization and
writes the .npy cache, without training a model:

    python preprocess.py --config configs/default.yaml
    python preprocess.py --config configs/default.yaml --experiment configs/flypaw_test.yaml

The resulting cache directory (dataset/cache/{freq}hz_{feature_hash}/ —
one .npy per mission + normalizer.json + splits.json) is a self-contained
training artifact: it can be copied to another machine (e.g. uploaded to
cloud storage and downloaded on the training machine) without the raw
telemetry CSVs. See data/dataset.py build_datasets() cache-only path and
the "Cache-Only Deploy" section in the top-level README.
"""

from __future__ import annotations

import argparse
import logging

from config import load_config
from data import build_datasets

logger = logging.getLogger("preprocess")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--experiment", default=None)
    parser.add_argument("overrides", nargs="*", help="a.b.c=value")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = load_config(args.config, args.experiment, args.overrides)
    datasets, _ = build_datasets(cfg["data"])
    for split, ds in datasets.items():
        logger.info("%s: %d missions, %d windows", split, len(ds.missions), len(ds))


if __name__ == "__main__":
    main()
