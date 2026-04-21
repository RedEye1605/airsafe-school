"""
Open-Meteo Weather Client — Historical + Forecast data for Jakarta.

Historical: ERA5 reanalysis (free, no API key needed)
  URL: https://archive-api.open-meteo.com/v1/archive
  
Forecast: Real-time weather
  URL: https://api.open-meteo.com/v1/forecast

Coordinates for Jakarta center: lat=-6.2, lon=106.85
"""
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Jakarta center coordinates
JAKARTA_LAT = -6.2
JAKARTA_LON = 106.85

# Variables to fetch
HOURLY_VARS = [
    'temperature_2m',
    'relative_humidity_2m',
    'wind_speed_10m',
    'precipitation',
    'surface_pressure',
]

DAILY_VARS = [
    'temperature_2m_max',
    'temperature_2m_min',
    'temperature_2m_mean',
    'precipitation_sum',
    'wind_speed_10m_max',
    'relative_humidity_2m_mean',
    'surface_pressure_mean',
]


def fetch_historical_weather(
    start_date: str,
    end_date: str,
    latitude: float = JAKARTA_LAT,
    longitude: float = JAKARTA_LON,
    hourly: bool = True,
    daily: bool = True,
) -> dict:
    """
    Fetch historical weather data from Open-Meteo Archive API.
    
    Args:
        start_date: YYYY-MM-DD format
        end_date: YYYY-MM-DD format
        latitude: Latitude (default Jakarta center)
        longitude: Longitude (default Jakarta center)
        hourly: Include hourly data
        daily: Include daily aggregates
    
    Returns:
        dict with 'hourly' and/or 'daily' DataFrames as dict records
    """
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'timezone': 'Asia/Jakarta',
    }
    
    if hourly:
        params['hourly'] = ','.join(HOURLY_VARS)
    if daily:
        params['daily'] = ','.join(DAILY_VARS)
    
    logger.info(f"Open-Meteo Archive: {start_date} to {end_date}")
    
    resp = requests.get(ARCHIVE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    
    result = {}
    
    if hourly and 'hourly' in data:
        result['hourly'] = data['hourly']
        n_hours = len(data['hourly'].get('time', []))
        logger.info(f"  Hourly: {n_hours} records")
    
    if daily and 'daily' in data:
        result['daily'] = data['daily']
        n_days = len(data['daily'].get('time', []))
        logger.info(f"  Daily: {n_days} records")
    
    return result


def fetch_forecast_weather(
    latitude: float = JAKARTA_LAT,
    longitude: float = JAKARTA_LON,
    forecast_days: int = 3,
    hourly: bool = True,
) -> dict:
    """
    Fetch forecast weather from Open-Meteo Forecast API.
    """
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'forecast_days': forecast_days,
        'timezone': 'Asia/Jakarta',
    }
    
    if hourly:
        params['hourly'] = ','.join(HOURLY_VARS)
    
    resp = requests.get(FORECAST_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    result = {}
    if 'hourly' in data:
        result['hourly'] = data['hourly']
    
    return result


def fetch_latest_weather(days_back: int = 7) -> dict:
    """
    Convenience function: fetch the most recent days of weather data.
    Open-Meteo archive has ~5 day delay, so this also fetches forecast for gap.
    """
    today = datetime.now()
    
    # Archive: goes back from 5 days ago
    end_archive = (today - timedelta(days=5)).strftime('%Y-%m-%d')
    start_archive = (today - timedelta(days=days_back + 5)).strftime('%Y-%m-%d')
    
    result = {'archive': None, 'forecast': None}
    
    try:
        result['archive'] = fetch_historical_weather(
            start_date=start_archive,
            end_date=end_archive,
        )
    except Exception as e:
        logger.error(f"Failed to fetch archive weather: {e}")
    
    try:
        result['forecast'] = fetch_forecast_weather(forecast_days=5)
    except Exception as e:
        logger.error(f"Failed to fetch forecast weather: {e}")
    
    return result


def fetch_training_weather(
    start_date: str = '2021-01-01',
    end_date: Optional[str] = None,
) -> dict:
    """
    Fetch historical weather for model training.
    Default: 2021-01-01 to present (5-day delay).
    """
    if end_date is None:
        end_date = (datetime.now() - timedelta(days=5)).strftime('%Y-%m-%d')
    
    return fetch_historical_weather(
        start_date=start_date,
        end_date=end_date,
    )
