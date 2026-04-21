#!/usr/bin/env python3
"""Fetch full historical weather data for model training (2021–2026).

Fetches from Open-Meteo Archive API in yearly chunks and saves
hourly + daily CSV files.
"""

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR
from src.data.weather_client import fetch_historical_weather
from src.data.transforms import (
    weather_daily_to_dataframe,
    weather_hourly_to_dataframe,
)
from src.utils import setup_logging

logger = logging.getLogger(__name__)

_CHUNKS = [
    ("2021-01-01", "2021-12-31"),
    ("2022-01-01", "2022-12-31"),
    ("2023-01-01", "2023-12-31"),
    ("2024-01-01", "2024-12-31"),
    ("2025-01-01", "2025-12-31"),
    ("2026-01-01", datetime.now().strftime("%Y-%m-%d")),
]


def main() -> None:
    """Fetch weather in yearly chunks and save combined CSVs."""
    setup_logging()

    data_root = Path(DATA_DIR)
    weather_dir = data_root / "raw" / "weather"
    weather_dir.mkdir(parents=True, exist_ok=True)

    all_hourly: list[pd.DataFrame] = []
    all_daily: list[pd.DataFrame] = []

    for start, end in _CHUNKS:
        logger.info("Fetching %s → %s ...", start, end)
        try:
            data = fetch_historical_weather(start, end, hourly=True, daily=True)

            if "hourly" in data and data["hourly"]:
                df_h = weather_hourly_to_dataframe(data["hourly"])
                if not df_h.empty:
                    all_hourly.append(df_h)
                    logger.info("  Hourly: %d records", len(df_h))

            if "daily" in data and data["daily"]:
                df_d = weather_daily_to_dataframe(data["daily"])
                if not df_d.empty:
                    all_daily.append(df_d)
                    logger.info("  Daily: %d records", len(df_d))
        except Exception as exc:
            logger.error("  Error: %s", exc)

    if all_hourly:
        hourly = pd.concat(all_hourly, ignore_index=True)
        hourly.drop_duplicates(subset=["time"], keep="last", inplace=True)
        hourly.sort_values("time", inplace=True)
        path = weather_dir / "weather_hourly_2021_2026.csv"
        hourly.to_csv(path, index=False)
        logger.info("Saved hourly: %d records → %s", len(hourly), path)

    if all_daily:
        daily = pd.concat(all_daily, ignore_index=True)
        daily.drop_duplicates(subset=["time"], keep="last", inplace=True)
        daily.sort_values("time", inplace=True)
        path = weather_dir / "weather_daily_2021_2026.csv"
        daily.to_csv(path, index=False)
        logger.info("Saved daily: %d records → %s", len(daily), path)


if __name__ == "__main__":
    main()
