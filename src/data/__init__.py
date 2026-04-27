"""Data acquisition and transformation sub-package."""

from src.data.blob_client import (
    is_blob_configured,
    upload_dataframe,
    upload_json,
    upload_text,
    save_dataframe_dual,
    save_json_dual,
)
from src.data.bmkg_client import bmkg_to_dataframe, fetch_bmkg_forecast
from src.data.spku_client import extract_pm25_readings, fetch_all_stations
from src.data.transforms import (
    classify_pm25_risk,
    compute_daily_summary,
    enrich_spku_with_risk,
    risk_to_color,
    spku_to_dataframe,
    weather_daily_to_dataframe,
    weather_hourly_to_dataframe,
)
from src.data.weather_client import (
    fetch_forecast_weather,
    fetch_historical_weather,
    fetch_latest_weather,
)

__all__ = [
    # Blob
    "is_blob_configured",
    "upload_dataframe",
    "upload_json",
    "upload_text",
    "save_dataframe_dual",
    "save_json_dual",
    # BMKG
    "fetch_bmkg_forecast",
    "bmkg_to_dataframe",
    # SPKU
    "fetch_all_stations",
    "extract_pm25_readings",
    # Weather
    "fetch_historical_weather",
    "fetch_forecast_weather",
    "fetch_latest_weather",
    # Transforms
    "spku_to_dataframe",
    "weather_hourly_to_dataframe",
    "weather_daily_to_dataframe",
    "compute_daily_summary",
    "classify_pm25_risk",
    "risk_to_color",
    "enrich_spku_with_risk",
]
