"""
AirSafe School — Azure Function App (Python v2 model).

ETL pipeline that pulls SPKU + BMKG + Open-Meteo data and saves to
Azure Blob Storage (when configured) or local filesystem (fallback).

Functions:
    1. etl_timer: Timer trigger (configurable schedule) — automated ETL
    2. etl_http: HTTP POST trigger — run ETL manually for testing

Local testing:
    python function_app.py --local
"""

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Allow imports from project src/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env before importing config (which reads env vars at module level)
from dotenv import load_dotenv
load_dotenv(_PROJECT_ROOT / ".env")

from src.config import (
    AIRSAFE_LOG_CONTAINER,
    AIRSAFE_PROCESSED_CONTAINER,
    AIRSAFE_RAW_CONTAINER,
    DATA_DIR,
    ETL_SCHEDULE,
    LOCAL_MODE,
)
from src.data.blob_client import (
    is_blob_configured,
    save_dataframe_dual,
    save_json_dual,
    upload_json,
)
from src.data.bmkg_client import bmkg_to_dataframe, fetch_bmkg_forecast
from src.data.spku_client import extract_pm25_readings, fetch_all_stations
from src.data.transforms import (
    enrich_spku_with_risk,
    spku_to_dataframe,
    weather_daily_to_dataframe,
    weather_hourly_to_dataframe,
)
from src.data.weather_client import (
    fetch_forecast_weather,
    fetch_historical_weather,
)
from src.exceptions import DataAcquisitionError

logger = logging.getLogger(__name__)

# ── Azure Functions v2 model registration ────────────────────────────────

if not LOCAL_MODE:
    import azure.functions as func

    app = func.FunctionApp()
else:
    app = None  # type: ignore[assignment]


# ── ETL Pipeline ─────────────────────────────────────────────────────────


def _ts_utc() -> tuple[str, str, str]:
    """Return (iso_now, date_str, hour_str) in UTC."""
    now = datetime.now(timezone.utc)
    return (
        now.isoformat(),
        now.strftime("%Y-%m-%d"),
        now.strftime("%H"),
    )


def run_etl_pipeline(data_root: Path | None = None) -> dict[str, Any]:
    """Core ETL — pulls SPKU + BMKG + weather, transforms, saves to Blob.

    Args:
        data_root: Root data directory (default: from config).

    Returns:
        Manifest dict with stats, file paths, and per-source status.
    """
    if data_root is None:
        data_root = Path(DATA_DIR)

    now_iso, date_str, hour_str = _ts_utc()
    ts_compact = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    manifest: dict[str, Any] = {
        "run_id": f"etl_{ts_compact}",
        "started_at_utc": now_iso,
        "status": "running",
        "blob_enabled": is_blob_configured(),
        "files": [],
        "sources": {},
    }

    # Step 1: SPKU air quality
    logger.info("Step 1: Fetching SPKU data...")
    try:
        spku_data = fetch_all_stations()
        pm25_records = extract_pm25_readings(spku_data)
        spku_df = spku_to_dataframe(pm25_records)
        spku_df = enrich_spku_with_risk(spku_df)

        # Raw snapshot
        raw_blob = f"spku/date={date_str}/hour={hour_str}/spku_{ts_compact}.json"
        raw_local = data_root / "raw" / "spku" / "snapshots" / f"spku_etl_{date_str}_{ts_compact}.json"
        raw_payload = {
            "collected_at": spku_data["collected_at"],
            "total_stations": spku_data["total"],
            "active_stations": spku_data["active"],
            "active_pm25": spku_data["active_pm25"],
            "pm25_record_count": len(pm25_records),
            "valid_readings": len(spku_df),
        }
        path = save_json_dual(raw_payload, raw_blob, raw_local)
        manifest["files"].append(path)

        # Processed CSV
        if not spku_df.empty:
            for name in (f"spku_pm25_{date_str}.csv", "spku_pm25_latest.csv"):
                proc_blob = f"daily/{name}"
                proc_local = data_root / "processed" / "daily" / name
                path = save_dataframe_dual(spku_df, proc_blob, proc_local, container=AIRSAFE_PROCESSED_CONTAINER)
                manifest["files"].append(path)

        manifest["sources"]["spku"] = {
            "ok": True,
            "total_stations": spku_data["total"],
            "active_pm25": spku_data["active_pm25"],
            "valid_readings": len(spku_df),
        }

    except Exception as exc:
        logger.error("SPKU fetch failed: %s", exc)
        manifest["sources"]["spku"] = {"ok": False, "error": str(exc)}

    # Step 2: BMKG weather
    logger.info("Step 2: Fetching BMKG weather...")
    try:
        bmkg_data = fetch_bmkg_forecast()
        bmkg_records = bmkg_to_dataframe(bmkg_data)

        bmkg_blob = f"bmkg/date={date_str}/hour={hour_str}/bmkg_{ts_compact}.json"
        bmkg_local = data_root / "raw" / "bmkg" / f"bmkg_{date_str}_{ts_compact}.json"
        path = save_json_dual(bmkg_data, bmkg_blob, bmkg_local)
        manifest["files"].append(path)

        manifest["sources"]["bmkg"] = {
            "ok": True,
            "areas_fetched": bmkg_data.get("count", 0),
            "areas_ok": bmkg_data.get("ok_count", 0),
            "forecast_records": len(bmkg_records),
        }

    except Exception as exc:
        logger.exception("BMKG fetch failed")
        manifest["sources"]["bmkg"] = {"ok": False, "error": str(exc)}

    # Step 3: Open-Meteo weather archive
    logger.info("Step 3: Fetching historical weather...")
    try:
        archive_end = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        archive_start = (datetime.now() - timedelta(days=12)).strftime("%Y-%m-%d")
        weather_data = fetch_historical_weather(archive_start, archive_end)

        if "hourly" in weather_data:
            hourly_df = weather_hourly_to_dataframe(weather_data["hourly"])
            if not hourly_df.empty:
                for name in (f"weather_hourly_{date_str}.csv", "weather_hourly_latest.csv"):
                    w_blob = f"openmeteo/date={date_str}/{name}"
                    w_local = data_root / "raw" / "weather" / name
                    path = save_dataframe_dual(hourly_df, w_blob, w_local)
                    manifest["files"].append(path)
                manifest["sources"]["weather"] = {
                    "ok": True,
                    "hourly_records": len(hourly_df),
                }

        if "daily" in weather_data:
            daily_df = weather_daily_to_dataframe(weather_data["daily"])
            if not daily_df.empty:
                name = f"weather_daily_{date_str}.csv"
                w_blob = f"openmeteo/date={date_str}/{name}"
                w_local = data_root / "raw" / "weather" / name
                path = save_dataframe_dual(daily_df, w_blob, w_local)
                manifest["files"].append(path)

    except Exception as exc:
        logger.error("Weather fetch failed: %s", exc)
        manifest["sources"]["weather"] = {"ok": False, "error": str(exc)}

    # Step 4: Open-Meteo forecast
    logger.info("Step 4: Fetching forecast...")
    try:
        forecast_data = fetch_forecast_weather(forecast_days=3)
        if "hourly" in forecast_data:
            forecast_df = weather_hourly_to_dataframe(forecast_data["hourly"])
            if not forecast_df.empty:
                name = f"weather_forecast_{date_str}.csv"
                w_blob = f"openmeteo/date={date_str}/{name}"
                w_local = data_root / "raw" / "weather" / name
                path = save_dataframe_dual(forecast_df, w_blob, w_local)
                manifest["files"].append(path)
                manifest["sources"]["forecast"] = {
                    "ok": True,
                    "records": len(forecast_df),
                }
    except Exception as exc:
        logger.error("Forecast fetch failed: %s", exc)
        manifest["sources"]["forecast"] = {"ok": False, "error": str(exc)}

    # Finalise manifest
    manifest["status"] = "completed"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()

    # Save manifest
    manifest_blob = f"etl/date={date_str}/run_{ts_compact}.json"
    manifest_local = data_root / "processed" / f"etl_summary_{date_str}.json"
    path = save_json_dual(
        manifest, manifest_blob, manifest_local, container=AIRSAFE_LOG_CONTAINER
    )
    manifest["files"].append(path)

    logger.info("ETL completed. %d files saved. Blob: %s", len(manifest["files"]), is_blob_configured())
    return manifest


# ── Azure Function endpoints ─────────────────────────────────────────────

if not LOCAL_MODE and app is not None:
    @app.function_name(name="etl_timer")
    @app.timer_trigger(
        schedule="%ETL_SCHEDULE%",
        arg_name="myTimer",
        run_on_startup=False,
    )
    def etl_timer(myTimer: func.TimerRequest) -> None:
        """Timer-triggered ETL on configurable schedule."""
        if myTimer.past_due:
            logger.warning("Timer is past due!")
        manifest = run_etl_pipeline()
        logger.info("ETL finished: %s", manifest["status"])

    @app.function_name(name="etl_http")
    @app.route(route="etl", methods=["POST"])
    def etl_http(req: func.HttpRequest) -> func.HttpResponse:
        """HTTP-triggered ETL for manual testing."""
        try:
            manifest = run_etl_pipeline()
            return func.HttpResponse(
                json.dumps(manifest, default=str, indent=2),
                mimetype="application/json",
                status_code=200 if manifest["status"] == "completed" else 500,
            )
        except Exception as exc:
            logger.exception("ETL HTTP failed")
            return func.HttpResponse(
                json.dumps({"error": str(exc)}),
                mimetype="application/json",
                status_code=500,
            )


# ── Local entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    if "--local" in sys.argv or LOCAL_MODE:
        from dotenv import load_dotenv
        from src.utils import setup_logging

        load_dotenv(_PROJECT_ROOT / ".env")
        setup_logging()
        import os
        os.environ["AIRSAFE_LOCAL_MODE"] = "1"

        print("=" * 60)
        print("AirSafe ETL Pipeline — LOCAL MODE")
        print(f"Blob Storage: {'ENABLED' if is_blob_configured() else 'NOT CONFIGURED'}")
        print("=" * 60)

        manifest = run_etl_pipeline()
        print(json.dumps(manifest, indent=2, default=str))
        print(f"\n{len(manifest['files'])} files saved:")
        for f in manifest["files"]:
            print(f"  -> {f}")
    else:
        print("Usage: python function_app.py --local")
        print("  Or set AIRSAFE_LOCAL_MODE=1")
