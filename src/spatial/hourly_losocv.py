"""
Temporal Leave-One-Sensor-Out Cross-Validation for hourly Kriging.

For each hour with valid PM2.5 data across the 5 ISPU stations, holds
out each station in turn and predicts its PM2.5 using the remaining 4
stations via Kriging. This produces O(5 * n_hours) predictions — far
more training data than the static 98-sensor LOSOCV.

The variogram is refit for each fold using only training stations,
ensuring honest cross-validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.spatial.kriging import KrigingConfig, kriging_interpolate

logger = logging.getLogger(__name__)

_HOURLY_LOSOCV_CONFIG = KrigingConfig(nlags=4, min_sensors=3)


@dataclass(frozen=True)
class HourlyLosoMetrics:
    """Aggregate metrics for hourly LOSOCV."""

    mae: float
    rmse: float
    r_squared: float
    median_ae: float
    bias: float
    n_predictions: int
    n_hours: int
    n_fallback: int
    per_sensor_mae: dict[str, float]


def _compute_hourly_metrics(results: pd.DataFrame) -> HourlyLosoMetrics:
    actual = results["actual_pm25"].to_numpy(dtype=float)
    predicted = results["predicted_pm25"].to_numpy(dtype=float)
    residuals = actual - predicted

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))
    r_squared = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    per_sensor = {}
    for sid, grp in results.groupby("sensor_id"):
        res = grp["actual_pm25"].to_numpy() - grp["predicted_pm25"].to_numpy()
        per_sensor[sid] = round(float(np.mean(np.abs(res))), 3)

    return HourlyLosoMetrics(
        mae=float(np.mean(np.abs(residuals))),
        rmse=float(np.sqrt(np.mean(residuals ** 2))),
        r_squared=r_squared,
        median_ae=float(np.median(np.abs(residuals))),
        bias=float(np.mean(residuals)),
        n_predictions=len(results),
        n_hours=results["datetime"].nunique(),
        n_fallback=int((results["variogram_used"] == "fallback_idw").sum()),
        per_sensor_mae=per_sensor,
    )


def hourly_losocv(
    hourly_df: pd.DataFrame,
    config: Optional[KrigingConfig] = None,
    max_hours: Optional[int] = None,
    seed: int = 42,
) -> tuple[pd.DataFrame, HourlyLosoMetrics]:
    """Run LOSOCV across hours for the 5-sensor ISPU network.

    For each station, holds it out and predicts its PM2.5 at every hour
    using the remaining stations. Produces O(n_stations * n_hours)
    predictions.

    Args:
        hourly_df: From load_hourly_data(). Must have datetime,
            station_name, latitude, longitude, pm25.
        config: KrigingConfig for the 5-sensor regime.
        max_hours: If set, subsample to this many hours for speed.
        seed: Random seed for subsampling.

    Returns:
        Tuple of (per-prediction DataFrame, aggregate HourlyLosoMetrics).
    """
    config = config or _HOURLY_LOSOCV_CONFIG

    station_names = sorted(hourly_df["station_name"].unique())
    n_stations = len(station_names)
    if n_stations < 2:
        raise ValueError(
            f"Need at least 2 stations for LOSOCV, got {n_stations}"
        )

    # Find hours where ALL stations have valid data
    counts_per_hour = hourly_df.groupby("datetime")["station_name"].nunique()
    complete_hours = counts_per_hour[counts_per_hour == n_stations].index

    if complete_hours.empty:
        raise ValueError("No hours with all stations having valid PM2.5.")

    if max_hours and len(complete_hours) > max_hours:
        rng = np.random.default_rng(seed)
        complete_hours = sorted(rng.choice(complete_hours, max_hours, replace=False))
        logger.info("Subsampled to %d hours (from %d)", max_hours, len(counts_per_hour))

    logger.info(
        "Running hourly LOSOCV: %d stations, %d hours, config=%s",
        n_stations, len(complete_hours), config,
    )

    records: list[dict] = []
    for i, dt in enumerate(complete_hours):
        hour_data = hourly_df[hourly_df["datetime"] == dt].reset_index(drop=True)

        for held_idx in range(len(hour_data)):
            held = hour_data.iloc[[held_idx]]
            mask = np.ones(len(hour_data), dtype=bool)
            mask[held_idx] = False
            remaining = hour_data.iloc[mask]

            target = pd.DataFrame({
                "npsn": [held["station_name"].iloc[0]],
                "latitude": [held["latitude"].iloc[0]],
                "longitude": [held["longitude"].iloc[0]],
            })

            try:
                result = kriging_interpolate(
                    remaining, target,
                    value_col="pm25",
                    config=config,
                )
                predicted = float(result.iloc[0]["pm25_kriging"])
                model_used = str(result.iloc[0]["variogram_model"])
                kriging_std = result.iloc[0].get("kriging_std", np.nan)
                n_sensors_fold = int(result.iloc[0].get("n_sensors", len(remaining)))
            except Exception as exc:
                logger.debug("LOSOCV failed for %s @ %s: %s",
                             held["station_name"].iloc[0], dt, exc)
                predicted = np.nan
                model_used = "error"
                kriging_std = np.nan
                n_sensors_fold = 0

            actual = float(held["pm25"].iloc[0])
            records.append({
                "datetime": dt,
                "sensor_id": held["station_name"].iloc[0],
                "latitude": held["latitude"].iloc[0],
                "longitude": held["longitude"].iloc[0],
                "actual_pm25": actual,
                "predicted_pm25": predicted,
                "abs_error": abs(actual - predicted) if not np.isnan(predicted) else np.nan,
                "squared_error": (actual - predicted) ** 2 if not np.isnan(predicted) else np.nan,
                "variogram_used": model_used,
                "kriging_std": kriging_std,
                "n_sensors_fold": n_sensors_fold,
            })

        if (i + 1) % 100 == 0:
            logger.info("LOSOCV progress: %d / %d hours", i + 1, len(complete_hours))

    results = pd.DataFrame(records)
    valid = results.dropna(subset=["predicted_pm25"])

    if valid.empty:
        raise ValueError("All hourly LOSOCV folds failed.")

    metrics = _compute_hourly_metrics(valid)

    logger.info(
        "Hourly LOSOCV complete: MAE=%.3f RMSE=%.3f R²=%.3f bias=%.3f "
        "(%d predictions, %d hours, %d fallback)",
        metrics.mae, metrics.rmse, metrics.r_squared, metrics.bias,
        metrics.n_predictions, metrics.n_hours, metrics.n_fallback,
    )

    return results, metrics
