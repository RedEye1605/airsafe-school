"""
AirSafe School — Azure Function App (Python v2 model).

Functions:
    1. etl_daily: Timer trigger (every 2 hours) — pulls SPKU + weather
    2. etl_on_demand: HTTP trigger — run ETL manually for testing

For local testing without Azure:
    python function_app.py --local
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Allow imports from project src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import DATA_DIR, LOCAL_MODE
from src.data.spku_client import SpkuApiError, extract_pm25_readings, fetch_all_stations
from src.data.transforms import (
    enrich_spku_with_risk,
    spku_to_dataframe,
    weather_daily_to_dataframe,
    weather_hourly_to_dataframe,
)
from src.data.weather_client import (
    WeatherApiError,
    fetch_forecast_weather,
    fetch_historical_weather,
)
from src.utils import ensure_dir, save_json

logger = logging.getLogger(__name__)

# ── Azure Functions v2 model registration ────────────────────────────────

if not LOCAL_MODE:
    import azure.functions as func

    app = func.FunctionApp()
else:
    app = None  # type: ignore[assignment]


def run_etl_pipeline(data_root: Path | None = None) -> dict[str, Any]:
    """Core ETL logic — pulls SPKU + Open-Meteo weather, transforms, saves.

    Args:
        data_root: Root data directory (default: from config).

    Returns:
        Summary dict with stats and file paths.
    """
    if data_root is None:
        data_root = Path(DATA_DIR)

    now = datetime.now()
    today_str = now.strftime("%Y-%m-%d")
    ts_str = now.strftime("%Y-%m-%d_%H%M%S")

    summary: dict[str, Any] = {
        "run_at": now.isoformat(),
        "status": "running",
        "files": [],
    }

    # Step 1: SPKU
    logger.info("Step 1: Fetching SPKU data...")
    try:
        spku_data = fetch_all_stations()
        pm25_records = extract_pm25_readings(spku_data)
        spku_df = spku_to_dataframe(pm25_records)
        spku_df = enrich_spku_with_risk(spku_df)

        raw_dir = data_root / "raw" / "spku" / "snapshots"
        ensure_dir(raw_dir)
        raw_path = raw_dir / f"spku_etl_{ts_str}.json"
        save_json({
            "collected_at": spku_data["collected_at"],
            "station_count": spku_data["total"],
            "pm25_records": len(pm25_records),
        }, raw_path)
        summary["files"].append(str(raw_path))

        if not spku_df.empty:
            proc_dir = data_root / "processed" / "daily"
            ensure_dir(proc_dir)
            for name in (f"spku_pm25_{today_str}.csv", "spku_pm25_latest.csv"):
                p = proc_dir / name
                spku_df.to_csv(p, index=False)
                summary["files"].append(str(p))

        summary["spku"] = {
            "total_stations": spku_data["total"],
            "active_stations": spku_data["active"],
            "active_pm25": spku_data["active_pm25"],
            "valid_readings": len(spku_df),
        }
    except SpkuApiError as exc:
        logger.error("SPKU fetch failed: %s", exc)
        summary["spku_error"] = str(exc)

    # Step 2: Weather archive
    logger.info("Step 2: Fetching historical weather...")
    try:
        archive_end = (now - timedelta(days=5)).strftime("%Y-%m-%d")
        archive_start = (now - timedelta(days=12)).strftime("%Y-%m-%d")
        weather_data = fetch_historical_weather(archive_start, archive_end)

        if "hourly" in weather_data:
            hourly_df = weather_hourly_to_dataframe(weather_data["hourly"])
            if not hourly_df.empty:
                wdir = data_root / "raw" / "weather"
                ensure_dir(wdir)
                for name in (f"weather_hourly_{today_str}.csv", "weather_hourly_latest.csv"):
                    hourly_df.to_csv(wdir / name, index=False)
                    summary["files"].append(str(wdir / name))
                summary["weather_hourly_records"] = len(hourly_df)

        if "daily" in weather_data:
            daily_df = weather_daily_to_dataframe(weather_data["daily"])
            if not daily_df.empty:
                wdir = data_root / "raw" / "weather"
                ensure_dir(wdir)
                daily_df.to_csv(wdir / f"weather_daily_{today_str}.csv", index=False)
                summary["files"].append(str(wdir / f"weather_daily_{today_str}.csv"))
                summary["weather_daily_records"] = len(daily_df)
    except WeatherApiError as exc:
        logger.error("Weather fetch failed: %s", exc)
        summary["weather_error"] = str(exc)

    # Step 3: Forecast
    logger.info("Step 3: Fetching forecast...")
    try:
        forecast_data = fetch_forecast_weather(forecast_days=3)
        if "hourly" in forecast_data:
            forecast_df = weather_hourly_to_dataframe(forecast_data["hourly"])
            if not forecast_df.empty:
                wdir = data_root / "raw" / "weather"
                ensure_dir(wdir)
                path = wdir / f"weather_forecast_{today_str}.csv"
                forecast_df.to_csv(path, index=False)
                summary["files"].append(str(path))
                summary["forecast_records"] = len(forecast_df)
    except WeatherApiError as exc:
        logger.error("Forecast fetch failed: %s", exc)
        summary["forecast_error"] = str(exc)

    # Save summary
    summary["status"] = "completed"
    summary_path = data_root / "processed" / f"etl_summary_{today_str}.json"
    save_json(summary, summary_path)
    summary["files"].append(str(summary_path))

    logger.info("ETL completed. %d files saved.", len(summary["files"]))
    return summary


# ── Azure Function endpoints ─────────────────────────────────────────────

if not LOCAL_MODE and app is not None:
    @app.timer_trigger(
        schedule="0 */2 * * *",
        arg_name="myTimer",
        run_on_startup=False,
    )
    def etl_daily(myTimer: func.TimerRequest) -> None:
        """Timer-triggered ETL: every 2 hours."""
        if myTimer.past_due:
            logger.warning("Timer is past due!")
        summary = run_etl_pipeline()
        logger.info("ETL completed: %s", summary["status"])

    @app.route(route="etl", methods=["POST"])
    def etl_on_demand(req: func.HttpRequest) -> func.HttpResponse:
        """HTTP-triggered ETL for manual testing."""
        try:
            summary = run_etl_pipeline()
            return func.HttpResponse(
                json.dumps(summary, default=str),
                mimetype="application/json",
                status_code=200,
            )
        except Exception as exc:
            return func.HttpResponse(
                json.dumps({"error": str(exc)}),
                mimetype="application/json",
                status_code=500,
            )


# ── Local entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    if "--local" in sys.argv or LOCAL_MODE:
        from src.utils import setup_logging

        setup_logging()
        os.environ["AIRSAFE_LOCAL_MODE"] = "1"

        print("=" * 60)
        print("AirSafe ETL Pipeline — LOCAL MODE")
        print("=" * 60)

        summary = run_etl_pipeline()
        print(json.dumps(summary, indent=2, default=str))
        print(f"\n{len(summary['files'])} files saved:")
        for f in summary["files"]:
            print(f"  → {f}")
    else:
        print("Usage: python function_app.py --local")
        print("  Or set AIRSAFE_LOCAL_MODE=1")
