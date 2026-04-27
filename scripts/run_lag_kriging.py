#!/usr/bin/env python3
"""Run lag-dataset Kriging spatial interpolation pipeline.

Reads friend's temporal lag datasets (h6/h12/h24), runs Kriging per-timestamp
to produce school-level PM2.5 estimates, and saves to disk.

Usage:
    python scripts/run_lag_kriging.py --lag-hours 6 --start-date 2025-01-01
    python scripts/run_lag_kriging.py --all-lags --batch-months
    python scripts/run_lag_kriging.py --lag-hours 6 --resume data/processed/lag_kriging_h6/
"""

from __future__ import annotations

import argparse
import gc
import logging
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.spatial.kriging import KrigingConfig
from src.spatial.lag_kriging import load_lag_dataset, run_lag_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = project_root / "data"
LAG_DIR = DATA_DIR / "friend_model" / "Data Lag"


def _get_month_batches(
    lag_path: Path,
    start_date: str | None,
    end_date: str | None,
) -> list[tuple[str, str]]:
    """Return list of (month_start, month_end) date strings for batching."""
    df = pd.read_csv(lag_path, usecols=["datetime"], parse_dates=["datetime"])
    ts = df["datetime"]
    if start_date:
        ts = ts[ts >= pd.Timestamp(start_date)]
    if end_date:
        ts = ts[ts <= pd.Timestamp(end_date)]

    months = sorted(ts.dt.to_period("M").unique())
    return [
        (m.start_time.strftime("%Y-%m-01"), m.end_time.strftime("%Y-%m-%d"))
        for m in months
    ]


def run_batched(
    lag_hours: int,
    lag_path: Path,
    school_path: Path,
    start_date: str | None,
    end_date: str | None,
    output_dir: Path,
    config: KrigingConfig,
    corrector_path: str | None,
    chunk_timestamps: int,
    resume: bool,
) -> Path:
    """Run pipeline month-by-month, writing one Parquet per month.

    Keeps memory flat (~3M rows per month) by flushing each month to disk
    and releasing it before starting the next.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_path = output_dir.parent / f"lag_kriging_h{lag_hours}.parquet"

    batches = _get_month_batches(lag_path, start_date, end_date)
    logger.info(
        "Batched mode: %d months for h%d (%s to %s)",
        len(batches), lag_hours,
        batches[0][0] if batches else "?",
        batches[-1][1] if batches else "?",
    )

    monthly_files: list[Path] = []
    for i, (ms, me) in enumerate(batches):
        month_tag = ms[:7]  # "2025-01"
        month_file = output_dir / f"h{lag_hours}_{month_tag}.parquet"

        if resume and month_file.exists():
            logger.info(
                "[%d/%d] %s already exists — skipping", i + 1, len(batches), month_tag,
            )
            monthly_files.append(month_file)
            continue

        logger.info("[%d/%d] Processing h%d %s ...", i + 1, len(batches), lag_hours, month_tag)
        run_lag_pipeline(
            lag_path=lag_path,
            school_path=school_path,
            output_path=month_file,
            start_date=ms,
            end_date=me,
            lag_hours=lag_hours,
            config=config,
            corrector_path=corrector_path,
            output_format="parquet",
            chunk_timestamps=chunk_timestamps,
        )
        monthly_files.append(month_file)
        gc.collect()

    # Concatenate all monthly Parquets into one final file
    logger.info("Concatenating %d monthly files → %s", len(monthly_files), final_path)
    chunks = []
    for f in monthly_files:
        chunks.append(pd.read_parquet(f))
    full = pd.concat(chunks, ignore_index=True)
    full.to_parquet(final_path, index=False)
    logger.info("Final output: %s (%d rows)", final_path, len(full))

    # Cleanup monthly files only after successful concatenation
    for f in monthly_files:
        f.unlink(missing_ok=True)
    try:
        output_dir.rmdir()
    except OSError:
        pass  # directory not empty or other issue, leave it

    return final_path


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
        "--output-format", choices=["csv", "parquet"], default="parquet",
        help="Output format (default: parquet)",
    )
    parser.add_argument(
        "--corrector-path", default=None,
        help="Path to trained ResidualCorrector pickle",
    )
    parser.add_argument(
        "--chunk-timestamps", type=int, default=100,
        help="Flush to disk every N timestamps (default: 100)",
    )
    parser.add_argument(
        "--batch-months", action="store_true",
        help="Process month-by-month for memory efficiency (recommended for full runs).",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already-completed monthly batches (use with --batch-months).",
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

        if args.batch_months:
            output_dir = DATA_DIR / "processed" / f"lag_kriging_h{lag_h}_batch"
            output_path = run_batched(
                lag_hours=lag_h,
                lag_path=lag_file,
                school_path=school_path,
                start_date=args.start_date,
                end_date=args.end_date,
                output_dir=output_dir,
                config=config,
                corrector_path=args.corrector_path,
                chunk_timestamps=args.chunk_timestamps,
                resume=args.resume,
            )
        else:
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
