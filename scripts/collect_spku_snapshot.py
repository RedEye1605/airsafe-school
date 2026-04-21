#!/usr/bin/env python3
"""Collect an SPKU snapshot and append to the time-series CSV.

Designed to be called from cron (hourly). Reads configuration from
environment variables — see ``src/config.py`` for defaults.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, SPKU_STALE_DAYS
from src.data.spku_client import fetch_all_stations
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Entry point for the SPKU snapshot collector."""
    setup_logging()

    output_dir = Path(DATA_DIR) / "raw" / "spku"
    snap_dir = output_dir / "snapshots"
    snap_dir.mkdir(parents=True, exist_ok=True)
    timeseries_path = output_dir / "spku_timeseries.csv"

    now = datetime.now()
    cutoff = now - timedelta(days=SPKU_STALE_DAYS)

    result = fetch_all_stations()
    stations = result["stations"]

    active = [s for s in stations if s["is_active"]]
    stale = [s for s in stations if not s["is_active"]]

    logger.info(
        "[%s] %d stations, %d active, %d stale",
        now.isoformat(), len(stations), len(active), len(stale),
    )

    # Save raw snapshot
    ts_str = now.strftime("%Y%m%d_%H%M%S")
    snap_path = snap_dir / f"spku_{ts_str}.json"
    with open(snap_path, "w") as fh:
        json.dump({
            "collected_at": now.isoformat(),
            "active": len(active),
            "stale": len(stale),
            "data": stations,
        }, fh, indent=2)

    # Build time-series rows
    rows = []
    for s in stations:
        rows.append({
            "collection_time": now.strftime("%Y-%m-%d %H:%M:%S"),
            "station_name": s["station_name"],
            "station_id": s["station_id"],
            "latitude": s["latitude"],
            "longitude": s["longitude"],
            "parameter": s["parameter"],
            "value": s["value"],
            "status": s["status"],
            "station_timestamp": s["timestamp"],
            "is_active": s["is_active"],
        })

    df = pd.DataFrame(rows)
    if timeseries_path.exists():
        existing = pd.read_csv(timeseries_path)
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(timeseries_path, index=False)

    active_pm25 = sum(1 for s in active if s["parameter"] == "PM25")
    logger.info(
        "Active PM25: %d | Stale: %d | Total rows: %d",
        active_pm25, len(stale), len(df),
    )


if __name__ == "__main__":
    main()
