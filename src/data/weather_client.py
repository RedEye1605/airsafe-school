"""
Open-Meteo Weather Client — Historical and forecast data.

Provides free (no API key) access to ERA5 reanalysis archives and
real-time forecasts for Jakarta and surrounding areas.

Historical archive:
    https://archive-api.open-meteo.com/v1/archive
Forecast:
    https://api.open-meteo.com/v1/forecast

Example:
    >>> data = fetch_historical_weather("2024-01-01", "2024-01-31")
    >>> df = weather_hourly_to_dataframe(data["hourly"])
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Optional

import requests

from src.config import (
    OPEN_METEO_ARCHIVE_URL,
    OPEN_METEO_FORECAST_URL,
    JAKARTA_LAT,
    JAKARTA_LON,
    WEATHER_REQUEST_TIMEOUT,
)

logger = logging.getLogger(__name__)

HOURLY_VARS: list[str] = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "surface_pressure",
]

DAILY_VARS: list[str] = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "precipitation_sum",
    "wind_speed_10m_max",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
]


def fetch_historical_weather(
    start_date: str,
    end_date: str,
    latitude: float = JAKARTA_LAT,
    longitude: float = JAKARTA_LON,
    *,
    hourly: bool = True,
    daily: bool = True,
) -> dict[str, Any]:
    """Fetch historical weather from the Open-Meteo Archive API.

    Args:
        start_date: Start date in ``YYYY-MM-DD`` format.
        end_date: End date in ``YYYY-MM-DD`` format.
        latitude: Latitude (default Jakarta centre).
        longitude: Longitude (default Jakarta centre).
        hourly: Include hourly-resolution data.
        daily: Include daily aggregates.

    Returns:
        Dictionary with optional ``hourly`` and ``daily`` keys
        containing raw API response dicts.

    Raises:
        requests.HTTPError: On non-2xx responses.
    """
    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": "Asia/Jakarta",
    }

    if hourly:
        params["hourly"] = ",".join(HOURLY_VARS)
    if daily:
        params["daily"] = ",".join(DAILY_VARS)

    logger.info("Open-Meteo Archive: %s to %s", start_date, end_date)

    resp = requests.get(
        OPEN_METEO_ARCHIVE_URL, params=params, timeout=WEATHER_REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    data = resp.json()

    result: dict[str, Any] = {}

    if hourly and "hourly" in data:
        result["hourly"] = data["hourly"]
        logger.info("  Hourly: %d records", len(data["hourly"].get("time", [])))

    if daily and "daily" in data:
        result["daily"] = data["daily"]
        logger.info("  Daily: %d records", len(data["daily"].get("time", [])))

    return result


def fetch_forecast_weather(
    latitude: float = JAKARTA_LAT,
    longitude: float = JAKARTA_LON,
    forecast_days: int = 3,
    *,
    hourly: bool = True,
) -> dict[str, Any]:
    """Fetch forecast weather from the Open-Meteo Forecast API.

    Args:
        latitude: Latitude.
        longitude: Longitude.
        forecast_days: Number of forecast days.
        hourly: Include hourly data.

    Returns:
        Dictionary with ``hourly`` key if available.
    """
    params: dict[str, Any] = {
        "latitude": latitude,
        "longitude": longitude,
        "forecast_days": forecast_days,
        "timezone": "Asia/Jakarta",
    }

    if hourly:
        params["hourly"] = ",".join(HOURLY_VARS)

    resp = requests.get(
        OPEN_METEO_FORECAST_URL, params=params, timeout=WEATHER_REQUEST_TIMEOUT
    )
    resp.raise_for_status()
    data = resp.json()

    result: dict[str, Any] = {}
    if "hourly" in data:
        result["hourly"] = data["hourly"]
    return result


def fetch_latest_weather(days_back: int = 7) -> dict[str, Any]:
    """Fetch the most recent weather data (archive + forecast gap-fill).

    Open-Meteo archive has ~5 day delay, so forecast data is used to
    cover the gap.

    Args:
        days_back: How many days before the archive delay to retrieve.

    Returns:
        Dictionary with ``archive`` and ``forecast`` keys.
    """
    today = datetime.now()

    end_archive = (today - timedelta(days=5)).strftime("%Y-%m-%d")
    start_archive = (today - timedelta(days=days_back + 5)).strftime("%Y-%m-%d")

    result: dict[str, Any] = {"archive": None, "forecast": None}

    try:
        result["archive"] = fetch_historical_weather(
            start_date=start_archive, end_date=end_archive,
        )
    except Exception:
        logger.exception("Failed to fetch archive weather")

    try:
        result["forecast"] = fetch_forecast_weather(forecast_days=5)
    except Exception:
        logger.exception("Failed to fetch forecast weather")

    return result
