"""
Data transforms — cleaning, merging, and aggregation for the AirSafe pipeline.

Provides conversions from raw API responses to clean DataFrames, daily
weather summaries, and ISPU risk classification for PM2.5 readings.

Example:
    >>> from src.data.transforms import classify_pm25_risk
    >>> classify_pm25_risk(42.5)
    'SEDANG'
"""

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Column rename maps ──────────────────────────────────────────────────────

_HOURLY_COL_MAP: dict[str, str] = {
    "temperature_2m": "temperature",
    "relative_humidity_2m": "humidity",
    "wind_speed_10m": "wind_speed",
    "precipitation": "precipitation",
    "surface_pressure": "pressure",
}

_DAILY_COL_MAP: dict[str, str] = {
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "temperature_2m_mean": "temp_mean",
    "precipitation_sum": "precipitation_sum",
    "wind_speed_10m_max": "wind_speed_max",
    "relative_humidity_2m_mean": "humidity_mean",
    "surface_pressure_mean": "pressure_mean",
}

# ── PM2.5 ISPU risk thresholds ──────────────────────────────────────────────

_RISK_COLORS: dict[str, str] = {
    "BAIK": "#22c55e",
    "SEDANG": "#eab308",
    "TIDAK SEHAT": "#f97316",
    "SANGAT TIDAK SEHAT": "#ef4444",
    "BERBAHAYA": "#7c3aed",
    "TIDAK ADA DATA": "#9ca3af",
}


def spku_to_dataframe(pm25_records: list[dict[str, Any]]) -> pd.DataFrame:
    """Convert PM2.5 records from an SPKU snapshot to a clean DataFrame.

    Filters out records with zero coordinates, missing/invalid PM2.5,
    or implausible values (>500 µg/m³).

    Args:
        pm25_records: List of dicts from :func:`~src.data.spku_client.extract_pm25_readings`.

    Returns:
        Cleaned DataFrame with numeric ``latitude``, ``longitude``, ``pm25``.
    """
    df = pd.DataFrame(pm25_records)
    if df.empty:
        return df

    for col in ("latitude", "longitude", "pm25"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[
        (df["latitude"] != 0)
        & (df["longitude"] != 0)
        & (df["pm25"].notna())
        & (df["pm25"] >= 0)
        & (df["pm25"] <= 500)
    ].copy()

    logger.info("SPKU clean: %d valid PM2.5 readings", len(df))
    return df


def weather_hourly_to_dataframe(hourly_data: dict[str, Any]) -> pd.DataFrame:
    """Convert Open-Meteo hourly data dict to a DataFrame.

    Args:
        hourly_data: ``hourly`` key from Open-Meteo API response.

    Returns:
        DataFrame with human-readable column names.
    """
    df = pd.DataFrame(hourly_data)
    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"])
    df.rename(
        columns={k: v for k, v in _HOURLY_COL_MAP.items() if k in df.columns},
        inplace=True,
    )

    for col in _HOURLY_COL_MAP.values():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def weather_daily_to_dataframe(daily_data: dict[str, Any]) -> pd.DataFrame:
    """Convert Open-Meteo daily aggregates dict to a DataFrame.

    Args:
        daily_data: ``daily`` key from Open-Meteo API response.

    Returns:
        DataFrame with human-readable column names.
    """
    df = pd.DataFrame(daily_data)
    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"])
    df.rename(
        columns={k: v for k, v in _DAILY_COL_MAP.items() if k in df.columns},
        inplace=True,
    )

    return df


def compute_daily_summary(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily aggregates from hourly weather data.

    Args:
        hourly_df: Output of :func:`weather_hourly_to_dataframe`.

    Returns:
        DataFrame with one row per date and aggregate statistics.
    """
    if hourly_df.empty:
        return pd.DataFrame()

    hourly_df = hourly_df.copy()
    hourly_df["date"] = hourly_df["time"].dt.date
    daily = hourly_df.groupby("date").agg(
        temp_mean=("temperature", "mean"),
        temp_max=("temperature", "max"),
        temp_min=("temperature", "min"),
        humidity_mean=("humidity", "mean"),
        wind_speed_mean=("wind_speed", "mean"),
        wind_speed_max=("wind_speed", "max"),
        precipitation_sum=("precipitation", "sum"),
        pressure_mean=("pressure", "mean"),
    ).reset_index()

    return daily


def classify_pm25_risk(pm25: float) -> str:
    """Classify a PM2.5 value into an ISPU risk category.

    Uses ISPU daily thresholds (24-hour averaging period):
    BAIK: 0-35, SEDANG: 36-75, TIDAK SEHAT: 76-115,
    SANGAT TIDAK SEHAT: 116-150, BERBAHAYA: >150.

    For hourly PM2.5 data, prefer :func:`classify_pm25_hourly` which uses
    BMKG hourly thresholds.

    Args:
        pm25: PM2.5 concentration in µg/m³.

    Returns:
        Risk category string in Indonesian.
    """
    if pd.isna(pm25):
        return "TIDAK ADA DATA"
    if pm25 <= 35:
        return "BAIK"
    if pm25 <= 75:
        return "SEDANG"
    if pm25 <= 115:
        return "TIDAK SEHAT"
    if pm25 <= 150:
        return "SANGAT TIDAK SEHAT"
    return "BERBAHAYA"


def classify_pm25_hourly(pm25: float) -> str:
    """Classify hourly PM2.5 using BMKG thresholds.

    BMKG defines hourly PM2.5 categories for Indonesia
    (source: cews.bmkg.go.id/dashboard_pm2p5.html):
    Baik: 0-15.5, Sedang: 15.6-55.4, Tidak Sehat: 55.5-150.4,
    Sangat Tidak Sehat: 150.5-250.4, Berbahaya: >250.4.

    These differ from ISPU daily thresholds because hourly
    concentrations fluctuate more than 24-hour averages.

    Args:
        pm25: Hourly PM2.5 concentration in µg/m³.

    Returns:
        Risk category string in Indonesian.
    """
    if pd.isna(pm25):
        return "TIDAK ADA DATA"
    if pm25 <= 15.5:
        return "BAIK"
    if pm25 <= 55.4:
        return "SEDANG"
    if pm25 <= 150.4:
        return "TIDAK SEHAT"
    if pm25 <= 250.4:
        return "SANGAT TIDAK SEHAT"
    return "BERBAHAYA"


def risk_to_color(risk: str) -> str:
    """Map an ISPU risk category to a hex colour string.

    Args:
        risk: Risk category from :func:`classify_pm25_risk`.

    Returns:
        Hex colour string (e.g. ``#22c55e``).
    """
    return _RISK_COLORS.get(risk, "#9ca3af")


def enrich_spku_with_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add ISPU risk classification and colour columns to SPKU data.

    Args:
        df: DataFrame with a ``pm25`` column.

    Returns:
        Same DataFrame with added ``risk_level`` and ``risk_color`` columns.
    """
    df = df.copy()
    df["risk_level"] = df["pm25"].apply(classify_pm25_risk)
    df["risk_color"] = df["risk_level"].apply(risk_to_color)
    return df
