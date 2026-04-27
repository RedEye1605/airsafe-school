"""
Kriging spatial interpolation for PM2.5 estimation at school locations.

Uses Ordinary Kriging via PyKrige to interpolate sensor PM2.5 readings
to arbitrary target points (schools). Falls back to IDW when too few
sensors are available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KrigingConfig:
    variogram_models: tuple[str, ...] = ("spherical", "exponential", "gaussian", "linear")
    coordinates_type: str = "geographic"
    nlags: int = 6
    weight: bool = True
    min_sensors: int = 3
    idw_power: float = 2.0
    max_output_pm25: float = 500.0


def _require_columns(df: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _clean_points(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    value_col: Optional[str] = None,
) -> pd.DataFrame:
    needed = [lat_col, lon_col] + ([value_col] if value_col else [])
    out = df[list(set(needed))].copy()
    for col in needed:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=needed)
    out = out[out[lat_col].between(-90, 90) & out[lon_col].between(-180, 360)]
    # PyKrige geographic mode expects positive longitude in [0, 360]
    out[lon_col] = np.where(out[lon_col] < 0, out[lon_col] + 360, out[lon_col])
    return out


def _dedupe_sensors(
    sensor_df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    value_col: str,
) -> pd.DataFrame:
    return (
        sensor_df
        .groupby([lat_col, lon_col], as_index=False)[value_col]
        .mean()
    )


def _idw_interpolate(
    sensor_df: pd.DataFrame,
    target_df: pd.DataFrame,
    value_col: str,
    lat_col: str,
    lon_col: str,
    power: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    sensor_xy = sensor_df[[lon_col, lat_col]].to_numpy(dtype=float)
    target_xy = target_df[[lon_col, lat_col]].to_numpy(dtype=float)
    values = sensor_df[value_col].to_numpy(dtype=float)

    preds = []
    variances = []
    for point in target_xy:
        distances = np.sqrt(np.sum((sensor_xy - point) ** 2, axis=1))
        if np.any(distances == 0):
            matched = values[distances == 0]
            preds.append(float(np.mean(matched)))
            variances.append(0.0)
            continue
        weights = 1.0 / np.power(distances, power)
        weights = weights / weights.sum()
        pred = float(np.sum(weights * values))
        variance = float(np.sum(weights * (values - pred) ** 2))
        preds.append(pred)
        variances.append(variance)
    return np.array(preds), np.array(variances)


def kriging_interpolate(
    sensor_df: pd.DataFrame,
    target_df: pd.DataFrame,
    value_col: str = "pm25",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    target_id_col: str = "npsn",
    config: Optional[KrigingConfig] = None,
) -> pd.DataFrame:
    """Estimate PM2.5 at target locations using Ordinary Kriging.

    Args:
        sensor_df: DataFrame with sensor coordinates and PM2.5 values.
        target_df: DataFrame with target coordinates (schools).
        value_col: Column name for the value to interpolate.
        lat_col: Latitude column name.
        lon_col: Longitude column name.
        target_id_col: ID column in target_df.
        config: Kriging configuration.

    Returns:
        target_df augmented with kriging estimates, variance, and metadata.
    """
    config = config or KrigingConfig()
    _require_columns(sensor_df, [lat_col, lon_col, value_col], "sensor_df")
    _require_columns(target_df, [target_id_col, lat_col, lon_col], "target_df")

    sensors = _clean_points(sensor_df, lat_col, lon_col, value_col)
    sensors = _dedupe_sensors(sensors, lat_col, lon_col, value_col)

    # Clean targets but track valid indices to preserve all original columns
    target_clean = _clean_points(target_df, lat_col, lon_col)
    valid_idx = target_clean.index
    targets = target_clean

    if sensors.empty:
        raise ValueError("No valid sensor rows available for interpolation.")
    if targets.empty:
        raise ValueError("No valid target rows available for interpolation.")

    result = target_df.loc[valid_idx].copy()
    n_sensors = len(sensors)

    # Fallback to IDW for very few sensors
    if n_sensors < config.min_sensors:
        logger.warning(
            "Only %d sensors available. Falling back to IDW interpolation.",
            n_sensors,
        )
        pred, var = _idw_interpolate(
            sensors, targets, value_col=value_col,
            lat_col=lat_col, lon_col=lon_col, power=config.idw_power,
        )
        result[f"{value_col}_kriging"] = np.clip(pred, 0, config.max_output_pm25)
        result["kriging_variance"] = np.maximum(var, 0)
        result["kriging_std"] = np.sqrt(result["kriging_variance"])
        result["variogram_model"] = "fallback_idw"
        result["n_sensors"] = n_sensors
        return result

    x = sensors[lon_col].to_numpy(dtype=float)
    y = sensors[lat_col].to_numpy(dtype=float)
    z = sensors[value_col].to_numpy(dtype=float)
    tx = targets[lon_col].to_numpy(dtype=float)
    ty = targets[lat_col].to_numpy(dtype=float)

    last_error: Optional[Exception] = None
    for model in config.variogram_models:
        try:
            logger.info("Trying OrdinaryKriging with variogram_model=%s", model)
            ok = OrdinaryKriging(
                x, y, z,
                variogram_model=model,
                nlags=config.nlags,
                weight=config.weight,
                verbose=False,
                enable_plotting=False,
                coordinates_type=config.coordinates_type,
                pseudo_inv=True,
            )
            pred, var = ok.execute("points", tx, ty, backend="loop")
            pred = np.asarray(pred, dtype=float)
            var = np.asarray(var, dtype=float)

            result[f"{value_col}_kriging"] = np.clip(pred, 0, config.max_output_pm25)
            result["kriging_variance"] = np.maximum(var, 0)
            result["kriging_std"] = np.sqrt(result["kriging_variance"])
            result["variogram_model"] = model
            result["n_sensors"] = n_sensors
            logger.info("Kriging succeeded with variogram_model=%s", model)
            return result
        except Exception as exc:
            logger.warning("Kriging failed for variogram_model=%s: %s", model, exc)
            last_error = exc

    # All models failed — fall back to IDW
    logger.warning(
        "All Kriging models failed. Falling back to IDW. Last error: %s",
        last_error,
    )
    pred, var = _idw_interpolate(
        sensors, targets, value_col=value_col,
        lat_col=lat_col, lon_col=lon_col, power=config.idw_power,
    )
    result[f"{value_col}_kriging"] = np.clip(pred, 0, config.max_output_pm25)
    result["kriging_variance"] = np.maximum(var, 0)
    result["kriging_std"] = np.sqrt(result["kriging_variance"])
    result["variogram_model"] = "fallback_idw"
    result["n_sensors"] = n_sensors
    return result
