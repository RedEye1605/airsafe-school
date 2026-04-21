"""Data acquisition and transformation sub-package."""

from src.data.spku_client import fetch_all_stations, extract_pm25_readings
from src.data.weather_client import (
    fetch_historical_weather,
    fetch_forecast_weather,
    fetch_latest_weather,
)
from src.data.transforms import (
    spku_to_dataframe,
    weather_hourly_to_dataframe,
    weather_daily_to_dataframe,
    compute_daily_summary,
    classify_pm25_risk,
    risk_to_color,
    enrich_spku_with_risk,
)

__all__ = [
    "fetch_all_stations",
    "extract_pm25_readings",
    "fetch_historical_weather",
    "fetch_forecast_weather",
    "fetch_latest_weather",
    "spku_to_dataframe",
    "weather_hourly_to_dataframe",
    "weather_daily_to_dataframe",
    "compute_daily_summary",
    "classify_pm25_risk",
    "risk_to_color",
    "enrich_spku_with_risk",
]
