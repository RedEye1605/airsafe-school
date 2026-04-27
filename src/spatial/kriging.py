"""
Kriging spatial interpolation for PM2.5 estimation at school locations.

Uses Ordinary Kriging via PyKrige to interpolate sensor PM2.5 readings
to arbitrary target points (schools). Falls back to IDW when too few
sensors are available.

Notes on units:
    With coordinates_type="geographic" (default), PyKrige computes
    variogram distances in **degrees** (great-circle angular distance).
    The variogram range parameter is therefore also in degrees. At
    Jakarta's latitude (~-6.2 deg), 1 degree ≈ 111 km.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging

logger = logging.getLogger(__name__)

_VALID_COORDINATES_TYPES = ("geographic", "euclidean")
_VALID_VARIOGRAM_MODELS = ("spherical", "exponential", "gaussian", "linear")


def _haversine_dist(
    lon1: np.ndarray, lat1: np.ndarray,
    lon2: np.ndarray, lat2: np.ndarray,
    radius_km: float = 6371.0,
) -> np.ndarray:
    """Vectorised haversine distance in km between arrays of points."""
    lon1, lat1, lon2, lat2 = (
        np.radians(a) for a in (lon1, lat1, lon2, lat2)
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return radius_km * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


@dataclass(frozen=True)
class KrigingConfig:
    """Configuration for Kriging interpolation.

    Attributes:
        variogram_models: Candidate models evaluated; the one with lowest
            variogram-fit residual (cR) is selected.
        coordinates_type: "geographic" (lat/lon) or "euclidean".
        nlags: Number of lag bins for the empirical variogram.
            10-15 is recommended for ~100 sensors.
        weight: If True, weight lag bins inversely by count (recommended).
        min_sensors: Minimum sensors for Kriging; below this, use IDW.
        idw_power: Inverse-distance power for the IDW fallback.
        max_output_pm25: Upper clip for predicted PM2.5 values.
    """

    variogram_models: tuple[str, ...] = (
        "spherical", "exponential", "gaussian", "linear",
    )
    coordinates_type: str = "geographic"
    nlags: int = 12
    weight: bool = True
    min_sensors: int = 3
    idw_power: float = 2.0
    max_output_pm25: float = 1000.0

    def __post_init__(self) -> None:
        if not self.variogram_models:
            raise ValueError("variogram_models must be non-empty.")
        bad = [
            m for m in self.variogram_models
            if m not in _VALID_VARIOGRAM_MODELS
        ]
        if bad:
            raise ValueError(
                f"Invalid variogram model(s): {bad}. "
                f"Valid: {_VALID_VARIOGRAM_MODELS}"
            )
        if self.coordinates_type not in _VALID_COORDINATES_TYPES:
            raise ValueError(
                f"coordinates_type must be one of {_VALID_COORDINATES_TYPES}, "
                f"got {self.coordinates_type!r}"
            )
        if self.nlags < 2:
            raise ValueError(f"nlags must be >= 2, got {self.nlags}")
        if self.min_sensors < 1:
            raise ValueError(
                f"min_sensors must be >= 1, got {self.min_sensors}"
            )
        if self.idw_power <= 0:
            raise ValueError(
                f"idw_power must be > 0, got {self.idw_power}"
            )
        if self.max_output_pm25 <= 0:
            raise ValueError(
                f"max_output_pm25 must be > 0, got {self.max_output_pm25}"
            )


def _require_columns(
    df: pd.DataFrame, columns: Iterable[str], name: str,
) -> None:
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
    out = df[needed].copy()
    for col in needed:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=needed)
    out = out[out[lat_col].between(-90, 90) & out[lon_col].between(-180, 180)]
    if value_col and value_col in out.columns:
        out = out[out[value_col] >= 0]
    # PyKrige geographic mode expects positive longitude in [0, 360]
    out[lon_col] = np.where(
        out[lon_col] < 0, out[lon_col] + 360, out[lon_col],
    )
    out = out.reset_index(drop=True)
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
    lat_orig: np.ndarray,
    lon_orig: np.ndarray,
    power: float = 2.0,
) -> np.ndarray:
    """IDW interpolation using haversine distance (km).

    Uses original (pre-conversion) lat/lon for haversine to avoid
    artifacts near the antimeridian.
    """
    s_lat = sensor_df[lat_col].to_numpy(dtype=float)
    s_lon = sensor_df[lon_col].to_numpy(dtype=float)
    # Convert sensor longitudes back to [-180, 180] for haversine
    s_lon = np.where(s_lon > 180, s_lon - 360, s_lon)
    values = sensor_df[value_col].to_numpy(dtype=float)

    t_lat = lat_orig.copy()
    t_lon = np.where(lon_orig > 180, lon_orig - 360, lon_orig)

    preds = np.empty(len(t_lat))
    for i in range(len(t_lat)):
        distances = _haversine_dist(
            s_lon, s_lat,
            np.full_like(s_lon, t_lon[i]),
            np.full_like(s_lat, t_lat[i]),
        )
        if np.any(distances == 0):
            matched = values[distances == 0]
            preds[i] = float(np.mean(matched))
            continue
        weights = 1.0 / np.power(distances, power)
        weights = weights / weights.sum()
        preds[i] = float(np.sum(weights * values))

    return preds


def _pick_best_variogram(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    config: KrigingConfig,
) -> tuple[OrdinaryKriging, str]:
    """Fit all candidate variogram models and return the one with lowest
    variogram-fit residual (cR).

    cR is the sum of squared differences between the empirical
    semivariogram and the fitted model — lower is better.
    """
    candidates: list[tuple[float, OrdinaryKriging, str]] = []
    last_error: Optional[Exception] = None

    for model in config.variogram_models:
        try:
            ok = OrdinaryKriging(
                x, y, z,
                variogram_model=model,
                nlags=config.nlags,
                weight=config.weight,
                verbose=False,
                enable_plotting=False,
                coordinates_type=config.coordinates_type,
                pseudo_inv=True,
                enable_statistics=True,
            )
            cR = float(ok.cR)
            candidates.append((cR, ok, model))
            logger.info("Variogram %s: cR=%.4f", model, cR)
        except Exception as exc:
            logger.warning(
                "Kriging failed for variogram_model=%s: %s", model, exc,
            )
            last_error = exc

    if candidates:
        candidates.sort(key=lambda t: t[0])
        best_cr, best_ok, best_model = candidates[0]
        logger.info(
            "Selected variogram: %s (cR=%.4f, %d candidates)",
            best_model, best_cr, len(candidates),
        )
        return best_ok, best_model

    raise RuntimeError(
        f"All variogram models failed. Last error: {last_error}"
    )


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
        Rows with invalid coordinates are dropped.  When the IDW fallback
        is used, ``kriging_variance`` and ``kriging_std`` are set to NaN
        (IDW does not produce statistically-comparable uncertainty).
    """
    config = config or KrigingConfig()
    _require_columns(sensor_df, [lat_col, lon_col, value_col], "sensor_df")
    _require_columns(target_df, [target_id_col, lat_col, lon_col], "target_df")

    # Compute valid-target mask before coordinate conversion
    target_coords = target_df[[lat_col, lon_col]].copy()
    for c in [lat_col, lon_col]:
        target_coords[c] = pd.to_numeric(target_coords[c], errors="coerce")
    valid_mask = (
        target_coords[lat_col].notna()
        & target_coords[lon_col].notna()
        & target_coords[lat_col].between(-90, 90)
        & target_coords[lon_col].between(-180, 180)
    )
    valid_positions = np.where(valid_mask)[0]

    # Keep original (pre-conversion) target coords for IDW haversine
    orig_lat = target_coords.iloc[valid_positions][lat_col].to_numpy(
        dtype=float,
    )
    orig_lon = target_coords.iloc[valid_positions][lon_col].to_numpy(
        dtype=float,
    )

    sensors = _clean_points(sensor_df, lat_col, lon_col, value_col)
    sensors = _dedupe_sensors(sensors, lat_col, lon_col, value_col)

    targets = _clean_points(target_df, lat_col, lon_col)

    if sensors.empty:
        raise ValueError(
            "No valid sensor rows available for interpolation."
        )
    if targets.empty:
        raise ValueError(
            "No valid target rows available for interpolation."
        )

    # Use iloc for positional indexing to avoid duplicate-index issues
    result = target_df.iloc[valid_positions].copy().reset_index(drop=True)

    n_sensors = len(sensors)

    # Fallback to IDW for very few sensors
    if n_sensors < config.min_sensors:
        logger.warning(
            "Only %d sensors available. Falling back to IDW interpolation.",
            n_sensors,
        )
        pred = _idw_interpolate(
            sensors, targets, value_col=value_col,
            lat_col=lat_col, lon_col=lon_col,
            lat_orig=orig_lat, lon_orig=orig_lon,
            power=config.idw_power,
        )
        result[f"{value_col}_kriging"] = np.clip(
            pred, 0, config.max_output_pm25,
        )
        result["kriging_variance"] = np.nan
        result["kriging_std"] = np.nan
        result["variogram_model"] = "fallback_idw"
        result["n_sensors"] = n_sensors
        return result

    x = sensors[lon_col].to_numpy(dtype=float)
    y = sensors[lat_col].to_numpy(dtype=float)
    z = sensors[value_col].to_numpy(dtype=float)
    tx = targets[lon_col].to_numpy(dtype=float)
    ty = targets[lat_col].to_numpy(dtype=float)

    try:
        ok, best_model = _pick_best_variogram(x, y, z, config)
        pred, var = ok.execute("points", tx, ty, backend="loop")
        pred = np.asarray(pred, dtype=float)
        var = np.asarray(var, dtype=float)

        result[f"{value_col}_kriging"] = np.clip(
            pred, 0, config.max_output_pm25,
        )
        result["kriging_variance"] = np.maximum(var, 0)
        result["kriging_std"] = np.sqrt(result["kriging_variance"])
        result["variogram_model"] = best_model
        result["n_sensors"] = n_sensors
        return result
    except RuntimeError:
        pass  # All models failed — fall through to IDW

    # All Kriging models failed — fall back to IDW
    logger.warning("All Kriging models failed. Falling back to IDW.")
    pred = _idw_interpolate(
        sensors, targets, value_col=value_col,
        lat_col=lat_col, lon_col=lon_col,
        lat_orig=orig_lat, lon_orig=orig_lon,
        power=config.idw_power,
    )
    result[f"{value_col}_kriging"] = np.clip(
        pred, 0, config.max_output_pm25,
    )
    result["kriging_variance"] = np.nan
    result["kriging_std"] = np.nan
    result["variogram_model"] = "fallback_idw"
    result["n_sensors"] = n_sensors
    return result
