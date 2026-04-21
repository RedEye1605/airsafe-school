#!/usr/bin/env python3
"""
Fetch full historical weather data for model training (2021-2025).
Open-Meteo Archive API allows up to ~365 days per request.
"""
import os
import sys
import json
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from etl.weather_client import fetch_historical_weather
from etl.transforms import weather_hourly_to_dataframe, weather_daily_to_dataframe

DATA_ROOT = os.path.expanduser('~/airsafe-school/data')

def fetch_training_data():
    """Fetch weather in yearly chunks to stay within API limits."""
    # Open-Meteo allows large ranges, but let's be safe with yearly chunks
    chunks = [
        ('2021-01-01', '2021-12-31'),
        ('2022-01-01', '2022-12-31'),
        ('2023-01-01', '2023-12-31'),
        ('2024-01-01', '2024-12-31'),
        ('2025-01-01', '2025-12-31'),
        ('2026-01-01', datetime.now().strftime('%Y-%m-%d')),
    ]
    
    all_hourly = []
    all_daily = []
    
    for start, end in chunks:
        print(f"Fetching {start} to {end}...")
        try:
            data = fetch_historical_weather(
                start_date=start,
                end_date=end,
                hourly=True,
                daily=True,
            )
            
            if 'hourly' in data and data['hourly']:
                df_h = weather_hourly_to_dataframe(data['hourly'])
                if not df_h.empty:
                    all_hourly.append(df_h)
                    print(f"  Hourly: {len(df_h)} records")
            
            if 'daily' in data and data['daily']:
                df_d = weather_daily_to_dataframe(data['daily'])
                if not df_d.empty:
                    all_daily.append(df_d)
                    print(f"  Daily: {len(df_d)} records")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Combine and save
    if all_hourly:
        hourly = pd.concat(all_hourly, ignore_index=True)
        hourly.drop_duplicates(subset=['time'], keep='last', inplace=True)
        hourly.sort_values('time', inplace=True)
        
        path = os.path.join(DATA_ROOT, 'raw', 'weather', 'weather_hourly_2021_2026.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        hourly.to_csv(path, index=False)
        print(f"\n✅ Saved hourly: {len(hourly)} records → {path}")
    
    if all_daily:
        daily = pd.concat(all_daily, ignore_index=True)
        daily.drop_duplicates(subset=['time'], keep='last', inplace=True)
        daily.sort_values('time', inplace=True)
        
        path = os.path.join(DATA_ROOT, 'raw', 'weather', 'weather_daily_2021_2026.csv')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        daily.to_csv(path, index=False)
        print(f"✅ Saved daily: {len(daily)} records → {path}")

if __name__ == '__main__':
    fetch_training_data()
