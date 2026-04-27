#!/usr/bin/env python3
"""Run the hourly Kriging spatial interpolation pipeline.

Produces school-level PM2.5 estimates via Kriging + LightGBM residual
correction. Uses the friend's lag dataset (dataset_h6.csv) as source since
it has complete hourly PM2.5 (0% missing vs 48.5% in raw SPKU). Output is a
minimal spatial layer (npsn, datetime, pm25_kriging, pm25_corrected, kriging_std)
for the ML engineer to join with features.

Steps:
1. Load friend's lag dataset (PM2.5 already imputed by Adit).
2. Run hourly LOSOCV.
3. Train LightGBM residual corrector on LOSOCV residuals.
4. Run full hourly Kriging with correction for all schools.
5. Save minimal Parquet output.

Usage:
    python scripts/run_hourly_kriging.py
    python scripts/run_hourly_kriging.py --batch-months --resume
    python scripts/run_hourly_kriging.py --start-date 2025-01-01 --end-date 2025-06-30
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.spatial.hourly_kriging import (
    ISPU_STATION_COORDS,
    load_hourly_data,
    run_hourly_pipeline,
)
from src.spatial.hourly_losocv import hourly_losocv
from src.spatial.kriging import KrigingConfig
from src.spatial.residual_corrector import ResidualCorrector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = project_root / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = project_root / "models"
CONFIG = KrigingConfig(nlags=4, min_sensors=3)


def _get_month_batches(
    data_path: Path,
    start_date: str | None,
    end_date: str | None,
) -> list[tuple[str, str]]:
    """Return list of (month_start, month_end) date strings for batching."""
    cols = ["datetime"]
    df = pd.read_csv(data_path, usecols=cols, parse_dates=["datetime"])
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


def train_corrector(
    data_path: Path,
    max_losocv_hours: int | None = 500,
) -> tuple[ResidualCorrector, dict]:
    """Run LOSOCV + train residual corrector. Returns (corrector, metrics)."""
    hourly_df = load_hourly_data(data_path)
    logger.info(
        "Loaded: %d rows, %d stations, %s to %s",
        len(hourly_df), hourly_df["station_name"].nunique(),
        hourly_df["datetime"].min(), hourly_df["datetime"].max(),
    )

    loso_results, loso_metrics = hourly_losocv(
        hourly_df, config=CONFIG, max_hours=max_losocv_hours,
    )

    loso_path = PROCESSED_DIR / "hourly_losocv_results.csv"
    loso_results.to_csv(loso_path, index=False)
    logger.info(
        "LOSOCV: MAE=%.3f RMSE=%.3f R²=%.4f (%d predictions)",
        loso_metrics.mae, loso_metrics.rmse, loso_metrics.r_squared,
        loso_metrics.n_predictions,
    )

    sensor_df = pd.DataFrame([
        {"station_name": name, "latitude": lat, "longitude": lon}
        for name, (lat, lon) in ISPU_STATION_COORDS.items()
    ])

    corrector = ResidualCorrector(
        lgbm_params={
            "objective": "regression_l2",
            "metric": "mae",
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.05,
            "num_leaves": 8,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.5,
            "reg_lambda": 2.0,
            "random_state": 42,
            "verbose": -1,
        },
        feature_columns=[
            "kriging_prediction", "kriging_std", "n_sensors",
            "dist_nearest", "dist_2nd_nearest", "sensor_density_10km",
            "latitude", "longitude", "lat_x_lon",
            "variogram_spherical", "variogram_exponential",
            "variogram_gaussian", "variogram_linear",
        ],
    )
    metrics = corrector.train(loso_results, sensor_df)

    model_path = MODELS_DIR / "hourly_residual_corrector.pkl"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    corrector.save(str(model_path))

    logger.info(
        "Corrector trained: CV MAE=%.3f ± %.3f, in-sample R² %.4f → %.4f",
        metrics["cv_mae_mean"], metrics["cv_mae_std"],
        metrics["in_sample_r2_before"], metrics["in_sample_r2_after"],
    )

    return corrector, {
        "losocv": {
            "mae": loso_metrics.mae,
            "rmse": loso_metrics.rmse,
            "r_squared": loso_metrics.r_squared,
            "bias": loso_metrics.bias,
            "n_predictions": loso_metrics.n_predictions,
            "n_hours": loso_metrics.n_hours,
            "per_sensor_mae": loso_metrics.per_sensor_mae,
        },
        "corrector": metrics,
    }


def run_batched(
    data_path: Path,
    school_path: Path,
    start_date: str | None,
    end_date: str | None,
    output_dir: Path,
    corrector_path: str | None,
    chunk_hours: int,
    resume: bool,
) -> Path:
    """Run pipeline month-by-month for memory efficiency."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir.parent / "school_pm25_kriging.parquet"

    batches = _get_month_batches(data_path, start_date, end_date)
    logger.info(
        "Batched: %d months (%s to %s)",
        len(batches),
        batches[0][0] if batches else "?",
        batches[-1][1] if batches else "?",
    )

    monthly_files: list[Path] = []
    for i, (ms, me) in enumerate(batches):
        month_tag = ms[:7]
        month_file = output_dir / f"pm25_{month_tag}.parquet"

        if resume and month_file.exists():
            logger.info("[%d/%d] %s exists — skipping", i + 1, len(batches), month_tag)
            monthly_files.append(month_file)
            continue

        logger.info("[%d/%d] Processing %s ...", i + 1, len(batches), month_tag)
        run_hourly_pipeline(
            data_path=data_path,
            school_path=school_path,
            output_path=month_file,
            start_date=ms,
            end_date=me,
            config=CONFIG,
            corrector_path=corrector_path,
            output_format="parquet",
        )
        monthly_files.append(month_file)
        gc.collect()

    logger.info("Concatenating %d months → %s", len(monthly_files), final_path)
    chunks = [pd.read_parquet(f) for f in monthly_files]
    full = pd.concat(chunks, ignore_index=True)
    full.to_parquet(final_path, index=False)
    logger.info("Final: %s (%d rows)", final_path, len(full))

    for f in monthly_files:
        f.unlink(missing_ok=True)
    if output_dir.exists() and not any(output_dir.iterdir()):
        output_dir.rmdir()

    return final_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly Kriging pipeline")
    parser.add_argument("--start-date", default="2025-01-01")
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--max-losocv-hours", type=int, default=500,
        help="Max hours for LOSOCV subsampling (0 = all)",
    )
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip LOSOCV + corrector training (use existing model)")
    parser.add_argument("--corrector-path", default=None,
                        help="Path to existing corrector (skip training)")
    parser.add_argument("--batch-months", action="store_true",
                        help="Process month-by-month for memory efficiency")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed monthly batches")
    parser.add_argument("--chunk-hours", type=int, default=200)
    args = parser.parse_args()

    data_path = DATA_DIR / "friend_model" / "Data Lag" / "dataset_h6.csv"
    school_path = PROCESSED_DIR / "schools" / "schools_geocoded.csv"
    max_losocv = args.max_losocv_hours if args.max_losocv_hours > 0 else None

    # Step 1: Train or load corrector
    if args.skip_training or args.corrector_path:
        corrector_path = args.corrector_path or str(MODELS_DIR / "hourly_residual_corrector.pkl")
        logger.info("Loading existing corrector: %s", corrector_path)
    else:
        logger.info("=== Training residual corrector ===")
        corrector, train_metrics = train_corrector(data_path, max_losocv)
        corrector_path = str(MODELS_DIR / "hourly_residual_corrector.pkl")

        summary_path = PROCESSED_DIR / "hourly_pipeline_summary.json"
        with open(summary_path, "w") as f:
            json.dump(train_metrics, f, indent=2, default=str)

    # Step 2: Run Kriging with correction
    logger.info("=== Running Kriging interpolation ===")

    if args.batch_months:
        output_dir = PROCESSED_DIR / "school_pm25_kriging_batch"
        output_path = run_batched(
            data_path=data_path,
            school_path=school_path,
            start_date=args.start_date,
            end_date=args.end_date,
            output_dir=output_dir,
            corrector_path=corrector_path,
            chunk_hours=args.chunk_hours,
            resume=args.resume,
        )
    else:
        output_path = run_hourly_pipeline(
            data_path=data_path,
            school_path=school_path,
            start_date=args.start_date,
            end_date=args.end_date,
            config=CONFIG,
            corrector_path=corrector_path,
            output_format="parquet",
        )

    logger.info("=== Done: %s ===", output_path)


if __name__ == "__main__":
    main()
