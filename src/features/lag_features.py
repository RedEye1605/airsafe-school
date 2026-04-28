"""Feature engineering for PM2.5 prediction — matches Adit's prepo-modelling-data.ipynb.

Builds lag features, rolling statistics, temporal features, wind components,
missing flags, and imputed values for LightGBM model input. The pipeline
produces the same feature columns as the training datasets (dataset_h6/h12/h24.csv).

Usage:
    from src.features.lag_features import build_prediction_features
    features = build_prediction_features(historical_df, horizon=6)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

TARGET_COL = "pm25"
POLLUTANT_COLS = ["pm10", "so2", "co", "o3", "no2"]
WEATHER_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]

WEATHER_LAG_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "rain",
    "surface_pressure",
    "wind_speed_10m",
    "wind_u",
    "wind_v",
]

# Per-horizon lag/rolling config (from Adit's prepo-modelling-data.ipynb)
HORIZON_CONFIG = {
    6: {
        "pm25_lags": [1, 2, 3, 6, 12, 24],
        "weather_lags": [1, 3, 6, 12, 24],
        "pollutant_lags": [1, 3, 6, 12, 24],
        "roll_windows": [3, 6, 12, 24],
    },
    12: {
        "pm25_lags": [1, 2, 3, 6, 12, 24, 48],
        "weather_lags": [1, 3, 6, 12, 24],
        "pollutant_lags": [1, 3, 6, 12, 24],
        "roll_windows": [3, 6, 12, 24],
    },
    24: {
        "pm25_lags": [1, 3, 6, 12, 24, 48, 72],
        "weather_lags": [1, 3, 6, 12, 24, 48],
        "pollutant_lags": [1, 3, 6, 12, 24, 48],
        "roll_windows": [6, 12, 24, 48],
    },
}

_DEFAULT_STATS_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "station_stats_lookup.json"


# ── Imputation helpers (matching Adit exactly) ─────────────────────────────


def _fill_weather_series(s: pd.Series) -> pd.Series:
    return s.interpolate(method="linear", limit_area="inside").bfill().ffill()


def _fill_aux_series(s: pd.Series) -> pd.Series:
    s = s.ffill(limit=3)
    s = s.interpolate(method="linear", limit_area="inside")
    s = s.bfill().ffill()
    if s.isna().any():
        med = s.median()
        if pd.notna(med):
            s = s.fillna(med)
    return s


def _fill_pm25_series(s: pd.Series) -> pd.Series:
    s = s.ffill(limit=3)
    s = s.interpolate(method="linear", limit_area="inside")
    s = s.bfill().ffill()
    if s.isna().any():
        med = s.median()
        if pd.notna(med):
            s = s.fillna(med)
    return s


def _fill_engineered_numeric(frame: pd.DataFrame, by: str = "station_slug") -> pd.DataFrame:
    out = frame.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()

    for col in numeric_cols:
        if out[col].isna().any():
            station_med = out.groupby(by)[col].transform("median")
            out[col] = out[col].fillna(station_med)

            if out[col].isna().any():
                global_med = out[col].median()
                if pd.notna(global_med):
                    out[col] = out[col].fillna(global_med)
                else:
                    out[col] = out[col].fillna(0)
    return out


# ── Station stats ──────────────────────────────────────────────────────────


def _load_station_stats(path: Optional[Path] = None) -> dict:
    path = path or _DEFAULT_STATS_PATH
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _get_station_stat(lookup: dict, station_slug: str, hour: int, month: int) -> tuple[float, float]:
    key_hour = f"{station_slug}|{hour}"
    key_month = f"{station_slug}|{month}"
    hour_mean = lookup.get("hour_mean", {}).get(key_hour, np.nan)
    month_mean = lookup.get("month_mean", {}).get(key_month, np.nan)
    return hour_mean, month_mean


# ── Main pipeline ──────────────────────────────────────────────────────────


def build_prediction_features(
    df: pd.DataFrame,
    horizon: int = 6,
    station_stats_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Build all features matching Adit's training data format.

    Takes a merged DataFrame (pollutants + weather) with at least 72h of
    history per station, computes all lag/rolling/temporal features, and
    returns the latest row per station ready for model prediction.

    Args:
        df: Merged DataFrame with columns: datetime, station_slug, station_name,
            station_id, lokasi, pm25, pm10, so2, co, o3, no2, kategori,
            temperature_2m, relative_humidity_2m, precipitation, rain,
            surface_pressure, wind_speed_10m, wind_direction_10m.
        horizon: Prediction horizon (6, 12, or 24).
        station_stats_path: Path to precomputed station stats JSON.

    Returns:
        DataFrame with one row per station and all model feature columns.
    """
    cfg = HORIZON_CONFIG[horizon]
    out = df.copy()

    # Ensure datetime
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out = out.sort_values(["station_slug", "datetime"]).reset_index(drop=True)

    # ── Step 1: Imputation ──────────────────────────────────────────────
    out["pm25_raw"] = out["pm25"].copy()

    # Clip pm25
    out["pm25"] = out["pm25"].clip(upper=500)

    # Missing flags
    for col in [TARGET_COL] + POLLUTANT_COLS:
        if col in out.columns:
            out[f"{col}_missing_flag"] = out[col].isna().astype("int8")

    # Weather imputation
    for col in WEATHER_COLS:
        if col in out.columns:
            out[col] = out.groupby("station_slug")[col].transform(_fill_weather_series)

    # Pollutant imputation → *_work columns
    for col in POLLUTANT_COLS:
        if col in out.columns:
            out[f"{col}_work"] = out.groupby("station_slug")[col].transform(_fill_aux_series)
            if out[f"{col}_work"].isna().any():
                global_med = out[f"{col}_work"].median()
                if pd.notna(global_med):
                    out[f"{col}_work"] = out[f"{col}_work"].fillna(global_med)

    # PM25 full clean
    out["pm25"] = out.groupby("station_slug")["pm25"].transform(_fill_pm25_series)
    if out["pm25"].isna().any():
        global_med = out["pm25"].median()
        if pd.notna(global_med):
            out["pm25"] = out["pm25"].fillna(global_med)

    # ── Step 2: Temporal features ────────────────────────────────────────
    out["hour_num"] = out["datetime"].dt.hour
    out["dayofweek"] = out["datetime"].dt.dayofweek
    out["month"] = out["datetime"].dt.month
    out["day"] = out["datetime"].dt.day
    out["year"] = out["datetime"].dt.year
    out["is_weekend"] = (out["dayofweek"] >= 5).astype("int8")

    out["is_rush_morning"] = out["hour_num"].between(6, 9).astype("int8")
    out["is_rush_evening"] = out["hour_num"].between(16, 19).astype("int8")
    out["is_workhour"] = out["hour_num"].between(8, 17).astype("int8")

    out["hour_sin"] = np.sin(2 * np.pi * out["hour_num"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour_num"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)

    out["season_simple"] = np.where(out["month"].isin([11, 12, 1, 2, 3, 4]), "wet", "dry")
    out["season_dry_flag"] = (out["season_simple"] == "dry").astype("int8")

    # ── Step 3: Wind features ────────────────────────────────────────────
    wind_rad = np.deg2rad(out["wind_direction_10m"])
    out["wind_u"] = out["wind_speed_10m"] * np.cos(wind_rad)
    out["wind_v"] = out["wind_speed_10m"] * np.sin(wind_rad)
    out["has_rain"] = (out["rain"] > 0).astype("int8")

    # ── Step 4: Lag features ─────────────────────────────────────────────
    g = out.groupby("station_slug")

    # PM25 lags
    for lag in cfg["pm25_lags"]:
        out[f"pm25_lag_{lag}"] = g["pm25"].shift(lag)

    # PM25 diffs
    if 1 in cfg["pm25_lags"] and 2 in cfg["pm25_lags"]:
        out["pm25_diff_1"] = out["pm25_lag_1"] - out["pm25_lag_2"]
    if 24 in cfg["pm25_lags"] and 1 in cfg["pm25_lags"]:
        out["pm25_diff_24"] = out["pm25_lag_1"] - out["pm25_lag_24"]

    # PM25 rolling (past-only: shift(1) before rolling)
    for w in cfg["roll_windows"]:
        min_periods = max(1, w // 2)
        out[f"pm25_roll_mean_{w}"] = g["pm25"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=min_periods).mean()
        )
        out[f"pm25_roll_std_{w}"] = g["pm25"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=min_periods).std()
        )
        out[f"pm25_roll_min_{w}"] = g["pm25"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=min_periods).min()
        )
        out[f"pm25_roll_max_{w}"] = g["pm25"].transform(
            lambda s: s.shift(1).rolling(w, min_periods=min_periods).max()
        )

    # Weather lags
    for col in WEATHER_LAG_COLS:
        if col in out.columns:
            for lag in cfg["weather_lags"]:
                out[f"{col}_lag_{lag}"] = g[col].shift(lag)

    # Weather rolling
    for w in cfg["roll_windows"]:
        min_periods = max(1, w // 2)
        if "rain" in out.columns:
            out[f"rain_roll_sum_{w}"] = g["rain"].transform(
                lambda s: s.shift(1).rolling(w, min_periods=min_periods).sum()
            )
        if "temperature_2m" in out.columns:
            out[f"temperature_2m_roll_mean_{w}"] = g["temperature_2m"].transform(
                lambda s: s.shift(1).rolling(w, min_periods=min_periods).mean()
            )
        if "relative_humidity_2m" in out.columns:
            out[f"relative_humidity_2m_roll_mean_{w}"] = g["relative_humidity_2m"].transform(
                lambda s: s.shift(1).rolling(w, min_periods=min_periods).mean()
            )
        if "wind_speed_10m" in out.columns:
            out[f"wind_speed_10m_roll_mean_{w}"] = g["wind_speed_10m"].transform(
                lambda s: s.shift(1).rolling(w, min_periods=min_periods).mean()
            )

    # Pollutant work lags
    pollutant_work_cols = [f"{c}_work" for c in POLLUTANT_COLS if f"{c}_work" in out.columns]
    for col in pollutant_work_cols:
        for lag in cfg["pollutant_lags"]:
            out[f"{col}_lag_{lag}"] = g[col].shift(lag)

    # ── Step 5: Station stats ────────────────────────────────────────────
    stats_lookup = _load_station_stats(station_stats_path)
    if stats_lookup:
        hour_means = []
        month_means = []
        for _, row in out.iterrows():
            h_mean, m_mean = _get_station_stat(
                stats_lookup, row["station_slug"],
                int(row["hour_num"]), int(row["month"]),
            )
            hour_means.append(h_mean)
            month_means.append(m_mean)
        out["station_hour_mean_pm25"] = hour_means
        out["station_month_mean_pm25"] = month_means
    else:
        # Fallback: compute from available data
        out["station_hour_mean_pm25"] = out.groupby(
            ["station_slug", "hour_num"],
        )["pm25"].transform("mean")
        out["station_month_mean_pm25"] = out.groupby(
            ["station_slug", "month"],
        )["pm25"].transform("mean")

    # ── Step 6: Fill remaining NaN ───────────────────────────────────────
    protected = {"pm25_raw"}
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    fill_cols = [c for c in numeric_cols if c not in protected]
    temp = out[["station_slug"] + fill_cols].copy()
    temp = _fill_engineered_numeric(temp, by="station_slug")
    for col in fill_cols:
        out[col] = temp[col]

    # ── Step 7: Return latest row per station ────────────────────────────
    latest = out.groupby("station_slug").last().reset_index()

    # Ensure station_slug is category for LightGBM
    if "station_slug" in latest.columns:
        latest["station_slug"] = latest["station_slug"].astype("category")
    if "kategori" in latest.columns:
        latest["kategori"] = latest["kategori"].astype("category")

    logger.info(
        "Built features for horizon h%d: %d stations, %d columns",
        horizon, len(latest), len(latest.columns),
    )
    return latest


def prepare_model_input(
    df: pd.DataFrame,
    target_col: str,
    model_feature_order: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Select feature columns matching Adit's prepare_xy function.

    Excludes: target_col, datetime, date, station_name, lokasi,
    season_simple (string), pm25_raw, pm25_clean_full.

    Args:
        df: Feature DataFrame from build_prediction_features.
        target_col: Target column name to exclude.
        model_feature_order: If provided, reorder columns to match model's
            feature_name_ list. Critical for correct LightGBM predictions.

    Returns:
        (X, feature_cols) — X has features ready for model.predict().
    """
    exclude = {
        target_col,
        "datetime", "date",
        "station_name", "lokasi",
        "season_simple",
        "pm25_raw", "pm25_clean_full",
        "last_update", "source_url", "source_file",
        "hour", "hc",
    }
    exclude = {c for c in exclude if c in df.columns}

    feature_cols = [c for c in df.columns if c not in exclude]
    X = df[feature_cols].copy()

    # Object columns → category for LightGBM
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = X[col].astype("category")

    # Reorder to match model's expected feature order
    if model_feature_order is not None:
        ordered = [c for c in model_feature_order if c in X.columns]
        X = X[ordered]
        feature_cols = ordered

    return X, feature_cols
