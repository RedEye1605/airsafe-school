#!/usr/bin/env python3
"""Run lag-dataset Kriging spatial interpolation pipeline.

Reads friend's temporal lag datasets (h6/h12/h24), runs Kriging per-timestamp
to produce school-level PM2.5 estimates, and saves to disk.

Usage:
    python scripts/run_lag_kriging.py --lag-hours 6 [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
    python scripts/run_lag_kriging.py --all-lags [--output-format csv|parquet]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.spatial.kriging import KrigingConfig
from src.spatial.lag_kriging import run_lag_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = project_root / "data"
LAG_DIR = DATA_DIR / "friend_model" / "Data Lag"


def main() -> None:
    parser = argparse.ArgumentParser(description="Lag Kriging pipeline")
    parser.add_argument(
        "--lag-hours", type=int, default=None,
        help="Lag horizon: 6, 12, or 24. Mutually exclusive with --all-lags.",
    )
    parser.add_argument(
        "--all-lags", action="store_true",
        help="Run all three lag datasets (h6, h12, h24).",
    )
    parser.add_argument(
        "--start-date", default=None,
        help="Start date (inclusive, default: all available)",
    )
    parser.add_argument(
        "--end-date", default=None,
        help="End date (inclusive, default: all available)",
    )
    parser.add_argument(
        "--output-format", choices=["csv", "parquet"], default="csv",
        help="Output format (default: csv for streaming)",
    )
    parser.add_argument(
        "--corrector-path", default=None,
        help="Path to trained ResidualCorrector pickle",
    )
    parser.add_argument(
        "--chunk-timestamps", type=int, default=100,
        help="Flush to disk every N timestamps (default: 100)",
    )
    args = parser.parse_args()

    if not args.all_lags and args.lag_hours is None:
        parser.error("Specify --lag-hours or --all-lags")

    if args.all_lags:
        lag_hours_list = [6, 12, 24]
    else:
        lag_hours_list = [args.lag_hours]

    config = KrigingConfig(nlags=4, min_sensors=3)
    school_path = DATA_DIR / "processed" / "schools" / "schools_geocoded.csv"

    for lag_h in lag_hours_list:
        lag_file = LAG_DIR / f"dataset_h{lag_h}.csv"
        if not lag_file.exists():
            logger.error("Lag dataset not found: %s", lag_file)
            continue

        logger.info("=== Processing h%d ===", lag_h)
        output_path = run_lag_pipeline(
            lag_path=lag_file,
            school_path=school_path,
            start_date=args.start_date,
            end_date=args.end_date,
            lag_hours=lag_h,
            config=config,
            corrector_path=args.corrector_path,
            output_format=args.output_format,
            chunk_timestamps=args.chunk_timestamps,
        )
        logger.info("h%d output: %s", lag_h, output_path)

    logger.info("=== All pipelines complete ===")


if __name__ == "__main__":
    main()
