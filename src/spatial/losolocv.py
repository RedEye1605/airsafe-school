"""
Leave-One-Sensor-Out Cross-Validation (LOSOCV) for Kriging interpolation.

For each sensor, holds it out and predicts its PM2.5 using the remaining
sensors via kriging_interpolate(). Computes per-sensor error metrics and
aggregate statistics (MAE, RMSE, R², bias).

The variogram is refit from scratch for each fold (using only the
training sensors) to avoid information leakage. This is the correct
practice for honest cross-validation but makes the procedure O(N)
relative to a single kriging run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.spatial.kriging import KrigingConfig, kriging_interpolate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LosoMetrics:
    """Aggregate cross-validation metrics."""

    mae: float
    rmse: float
    r_squared: float
    median_ae: float
    bias: float
    n_sensors: int
    n_fallback: int


def _compute_metrics(errors: pd.DataFrame) -> LosoMetrics:
    actual = errors["actual_pm25"].to_numpy(dtype=float)
    predicted = errors["predicted_pm25"].to_numpy(dtype=float)
    residuals = actual - predicted

    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))

    if ss_tot > 0:
        r_squared = float(1 - ss_res / ss_tot)
    else:
        r_squared = 0.0 if ss_res == 0 else float("nan")

    return LosoMetrics(
        mae=float(np.mean(np.abs(residuals))),
        rmse=float(np.sqrt(np.mean(residuals ** 2))),
        r_squared=r_squared,
        median_ae=float(np.median(np.abs(residuals))),
        bias=float(np.mean(residuals)),
        n_sensors=len(errors),
        n_fallback=int((errors["variogram_used"] == "fallback_idw").sum()),
    )


def losocv_validate(
    sensor_df: pd.DataFrame,
    value_col: str = "pm25",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    sensor_id_col: str = "station_name",
    config: Optional[KrigingConfig] = None,
) -> tuple[pd.DataFrame, LosoMetrics]:
    """Run leave-one-sensor-out cross-validation.

    For each sensor, holds it out and uses the remaining sensors to
    predict PM2.5 at the held-out sensor location via kriging_interpolate().

    The variogram is refit for each fold using only training sensors,
    ensuring honest cross-validation without information leakage.

    Args:
        sensor_df: DataFrame with sensor coordinates and PM2.5 values.
        value_col: Column name for the value to interpolate.
        lat_col: Latitude column name.
        lon_col: Longitude column name.
        sensor_id_col: Column identifying each sensor uniquely.
        config: Kriging configuration.

    Returns:
        Tuple of (per-sensor results DataFrame, aggregate LosoMetrics).
        The DataFrame has columns: sensor_id, latitude, longitude,
        actual_pm25, predicted_pm25, abs_error, squared_error, variogram_used.

    Raises:
        ValueError: If fewer than 2 valid sensors are provided.
    """
    config = config or KrigingConfig()
    n = len(sensor_df)
    if n < 2:
        raise ValueError(
            f"Need at least 2 sensors for LOSOCV, got {n}"
        )

    records: list[dict] = []

    for idx in range(n):
        # Use positional boolean mask to avoid duplicate-index bugs
        mask = np.ones(n, dtype=bool)
        mask[idx] = False
        held = sensor_df.iloc[[idx]]
        remaining = sensor_df.iloc[mask]

        target = pd.DataFrame({
            "npsn": [str(held[sensor_id_col].iloc[0])],
            lat_col: [held[lat_col].iloc[0]],
            lon_col: [held[lon_col].iloc[0]],
        })

        try:
            result = kriging_interpolate(
                remaining, target,
                value_col=value_col, lat_col=lat_col, lon_col=lon_col,
                config=config,
            )
            predicted = float(result.iloc[0][f"{value_col}_kriging"])
            model_used = str(result.iloc[0]["variogram_model"])
            kriging_var = result.iloc[0].get("kriging_variance", np.nan)
            kriging_std = result.iloc[0].get("kriging_std", np.nan)
            fold_n_sensors = int(result.iloc[0].get("n_sensors", len(remaining)))
        except Exception as exc:
            logger.warning(
                "LOSOCV failed for sensor %s: %s",
                held[sensor_id_col].iloc[0], exc,
            )
            predicted = np.nan
            model_used = "error"
            kriging_var = np.nan
            kriging_std = np.nan
            fold_n_sensors = 0

        actual = float(held[value_col].iloc[0])
        records.append({
            "sensor_id": str(held[sensor_id_col].iloc[0]),
            "latitude": float(held[lat_col].iloc[0]),
            "longitude": float(held[lon_col].iloc[0]),
            "actual_pm25": actual,
            "predicted_pm25": predicted,
            "abs_error": abs(actual - predicted) if not np.isnan(predicted) else np.nan,
            "squared_error": (actual - predicted) ** 2 if not np.isnan(predicted) else np.nan,
            "variogram_used": model_used,
            "kriging_variance": kriging_var,
            "kriging_std": kriging_std,
            "n_sensors_fold": fold_n_sensors,
        })

        logger.debug(
            "Sensor %s: actual=%.2f predicted=%.2f error=%.2f (%s)",
            records[-1]["sensor_id"], actual, predicted,
            records[-1]["abs_error"], model_used,
        )

    results = pd.DataFrame(records)
    valid = results.dropna(subset=["predicted_pm25"])

    if valid.empty:
        raise ValueError(
            "All LOSOCV folds failed — no valid predictions."
        )

    metrics = _compute_metrics(valid)

    logger.info(
        "LOSOCV complete: MAE=%.3f RMSE=%.3f R²=%.3f bias=%.3f "
        "(%d sensors, %d fallback)",
        metrics.mae, metrics.rmse, metrics.r_squared, metrics.bias,
        metrics.n_sensors, metrics.n_fallback,
    )

    return results, metrics
