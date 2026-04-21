"""
Data transforms — Cleaning, merging, and aggregation for AirSafe pipeline.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


def spku_to_dataframe(pm25_records: list[dict]) -> pd.DataFrame:
    """Convert PM2.5 records from SPKU snapshot to clean DataFrame."""
    df = pd.DataFrame(pm25_records)
    if df.empty:
        return df
    
    # Ensure numeric types
    for col in ['latitude', 'longitude', 'pm25']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filter out invalid
    df = df[
        (df['latitude'] != 0) &
        (df['longitude'] != 0) &
        (df['pm25'].notna()) &
        (df['pm25'] >= 0) &
        (df['pm25'] <= 500)  # Cap at 500 µg/m³ (extreme but plausible)
    ].copy()
    
    logger.info(f"SPKU clean: {len(df)} valid PM2.5 readings")
    return df


def weather_hourly_to_dataframe(hourly_data: dict) -> pd.DataFrame:
    """Convert Open-Meteo hourly data to DataFrame."""
    df = pd.DataFrame(hourly_data)
    if df.empty:
        return df
    
    df['time'] = pd.to_datetime(df['time'])
    
    # Rename columns for clarity
    col_map = {
        'temperature_2m': 'temperature',
        'relative_humidity_2m': 'humidity',
        'wind_speed_10m': 'wind_speed',
        'precipitation': 'precipitation',
        'surface_pressure': 'pressure',
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    
    # Ensure numeric
    numeric_cols = ['temperature', 'humidity', 'wind_speed', 'precipitation', 'pressure']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def weather_daily_to_dataframe(daily_data: dict) -> pd.DataFrame:
    """Convert Open-Meteo daily aggregates to DataFrame."""
    df = pd.DataFrame(daily_data)
    if df.empty:
        return df
    
    df['time'] = pd.to_datetime(df['time'])
    
    col_map = {
        'temperature_2m_max': 'temp_max',
        'temperature_2m_min': 'temp_min',
        'temperature_2m_mean': 'temp_mean',
        'precipitation_sum': 'precipitation_sum',
        'wind_speed_10m_max': 'wind_speed_max',
        'relative_humidity_2m_mean': 'humidity_mean',
        'surface_pressure_mean': 'pressure_mean',
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)
    
    return df


def compute_daily_summary(hourly_df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily summary from hourly weather data."""
    if hourly_df.empty:
        return pd.DataFrame()
    
    hourly_df['date'] = hourly_df['time'].dt.date
    daily = hourly_df.groupby('date').agg(
        temp_mean=('temperature', 'mean'),
        temp_max=('temperature', 'max'),
        temp_min=('temperature', 'min'),
        humidity_mean=('humidity', 'mean'),
        wind_speed_mean=('wind_speed', 'mean'),
        wind_speed_max=('wind_speed', 'max'),
        precipitation_sum=('precipitation', 'sum'),
        pressure_mean=('pressure', 'mean'),
    ).reset_index()
    
    return daily


def classify_pm25_risk(pm25: float) -> str:
    """Classify PM2.5 into ISPU risk categories."""
    if pd.isna(pm25):
        return 'TIDAK ADA DATA'
    if pm25 <= 35:
        return 'BAIK'
    elif pm25 <= 75:
        return 'SEDANG'
    elif pm25 <= 115:
        return 'TIDAK SEHAT'
    elif pm25 <= 150:
        return 'SANGAT TIDAK SEHAT'
    else:
        return 'BERBAHAYA'


def risk_to_color(risk: str) -> str:
    """Map risk category to hex color."""
    colors = {
        'BAIK': '#22c55e',
        'SEDANG': '#eab308',
        'TIDAK SEHAT': '#f97316',
        'SANGAT TIDAK SEHAT': '#ef4444',
        'BERBAHAYA': '#7c3aed',
        'TIDAK ADA DATA': '#9ca3af',
    }
    return colors.get(risk, '#9ca3af')


def enrich_spku_with_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add risk classification and color to SPKU DataFrame."""
    df['risk_level'] = df['pm25'].apply(classify_pm25_risk)
    df['risk_color'] = df['risk_level'].apply(risk_to_color)
    return df
