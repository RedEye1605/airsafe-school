"""
Leave-One-Sensor-Out Cross-Validation (LOSOCV) for Kriging interpolation.

For each sensor, holds it out and predicts its PM2.5 using the remaining
sensors via kriging_interpolate(). Computes per-sensor error metrics and
aggregate statistics (MAE, RMSE, R², bias).
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

    return LosoMetrics(
        mae=float(np.mean(np.abs(residuals))),
        rmse=float(np.sqrt(np.mean(residuals ** 2))),
        r_squared=float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
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
    """
    config = config or KrigingConfig()
    records: list[dict] = []

    for idx in range(len(sensor_df)):
        held = sensor_df.iloc[[idx]]
        remaining = sensor_df.drop(sensor_df.index[idx])

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
        except Exception as exc:
            logger.warning(
                "LOSOCV failed for sensor %s: %s",
                held[sensor_id_col].iloc[0], exc,
            )
            predicted = np.nan
            model_used = "error"

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
        })

        logger.info(
            "Sensor %s: actual=%.2f predicted=%.2f error=%.2f (%s)",
            records[-1]["sensor_id"], actual, predicted,
            records[-1]["abs_error"], model_used,
        )

    results = pd.DataFrame(records)
    valid = results.dropna(subset=["predicted_pm25"])
    metrics = _compute_metrics(valid)

    logger.info(
        "LOSOCV complete: MAE=%.3f RMSE=%.3f R²=%.3f bias=%.3f "
        "(%d sensors, %d fallback)",
        metrics.mae, metrics.rmse, metrics.r_squared, metrics.bias,
        metrics.n_sensors, metrics.n_fallback,
    )

    return results, metrics
