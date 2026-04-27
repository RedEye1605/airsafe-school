#!/usr/bin/env python3
"""Run the hourly Kriging interpolation pipeline.

Steps:
1. Load friend's hourly SPKU dataset.
2. Run hourly LOSOCV (leave-one-sensor-out across hours).
3. Train LightGBM residual corrector on LOSOCV results.
4. Run full hourly Kriging with correction for all schools.
5. Save output to data/processed/.

Usage:
    python scripts/run_hourly_kriging.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]
                                         [--max-losocv-hours N] [--output-format csv|parquet]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly Kriging pipeline")
    parser.add_argument(
        "--start-date", default="2025-01-01",
        help="Start date (inclusive, default: 2025-01-01)",
    )
    parser.add_argument(
        "--end-date", default=None,
        help="End date (inclusive, default: all available)",
    )
    parser.add_argument(
        "--max-losocv-hours", type=int, default=500,
        help="Max hours for LOSOCV subsampling (default: 500). "
             "Set to 0 for all hours.",
    )
    parser.add_argument(
        "--output-format", choices=["csv", "parquet"], default="parquet",
        help="Output format (default: parquet)",
    )
    args = parser.parse_args()

    data_path = DATA_DIR / "dataset_master_spku_weather.csv"
    school_path = PROCESSED_DIR / "schools" / "schools_geocoded.csv"
    max_losocv = args.max_losocv_hours if args.max_losocv_hours > 0 else None

    # Step 1: Load hourly data
    logger.info("=== Step 1: Loading hourly data ===")
    hourly_df = load_hourly_data(
        data_path,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    logger.info(
        "Loaded: %d rows, %d stations, %s to %s",
        len(hourly_df), hourly_df["station_name"].nunique(),
        hourly_df["datetime"].min(), hourly_df["datetime"].max(),
    )

    # Step 2: Run hourly LOSOCV
    logger.info("=== Step 2: Running hourly LOSOCV ===")
    config = KrigingConfig(nlags=4, min_sensors=3)
    loso_results, loso_metrics = hourly_losocv(
        hourly_df, config=config, max_hours=max_losocv,
    )

    loso_path = PROCESSED_DIR / "hourly_losocv_results.csv"
    loso_results.to_csv(loso_path, index=False)
    logger.info("LOSOCV results saved: %s (%d rows)", loso_path, len(loso_results))

    # Step 3: Train residual corrector
    logger.info("=== Step 3: Training residual corrector ===")
    sensor_df = pd.DataFrame([
        {
            "station_name": name,
            "latitude": lat,
            "longitude": lon,
            "pm25": hourly_df[hourly_df["station_name"] == name]["pm25"].mean(),
        }
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

    metrics_path = PROCESSED_DIR / "hourly_residual_corrector_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Step 4: Run full hourly pipeline with correction
    logger.info("=== Step 4: Running full hourly Kriging ===")
    output_path = run_hourly_pipeline(
        data_path=data_path,
        school_path=school_path,
        start_date=args.start_date,
        end_date=args.end_date,
        config=config,
        corrector_path=model_path,
        output_format=args.output_format,
    )

    # Save summary
    summary = {
        "data_source": str(data_path),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "n_ispu_stations": len(ISPU_STATION_COORDS),
        "losocv": {
            "mae": loso_metrics.mae,
            "rmse": loso_metrics.rmse,
            "r_squared": loso_metrics.r_squared,
            "bias": loso_metrics.bias,
            "n_predictions": loso_metrics.n_predictions,
            "n_hours": loso_metrics.n_hours,
            "n_fallback": loso_metrics.n_fallback,
            "per_sensor_mae": loso_metrics.per_sensor_mae,
        },
        "corrector": metrics,
        "output": str(output_path),
    }
    summary_path = PROCESSED_DIR / "hourly_pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info("=== Pipeline complete ===")
    logger.info("Output: %s", output_path)
    logger.info(
        "LOSOCV MAE=%.3f, Corrector CV MAE=%.3f",
        loso_metrics.mae, metrics.get("cv_mae_mean", "N/A"),
    )


if __name__ == "__main__":
    main()
