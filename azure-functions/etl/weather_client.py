"""Re-export weather client from src.data.weather_client."""

from src.data.weather_client import (  # noqa: F401
    JAKARTA_LAT,
    JAKARTA_LON,
    WeatherApiError,
    fetch_forecast_weather,
    fetch_historical_weather,
    fetch_latest_weather,
)
