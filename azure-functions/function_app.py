"""
AirSafe School — Azure Function App (Python v2 model)

Functions:
  1. etl_daily: Timer trigger (every 2 hours) — pulls SPKU + weather, saves to blob
  2. etl_on_demand: HTTP trigger — run ETL manually for testing

For local testing without Azure:
  python function_app.py --local
"""
import os
import json
import logging
from datetime import datetime, timedelta

# Local mode: skip azure.functions import
LOCAL_MODE = os.environ.get('AIRSAFE_LOCAL_MODE', '0') == '1'

if not LOCAL_MODE:
    import azure.functions as func
    app = func.FunctionApp()
else:
    app = None

from etl.spku_client import fetch_all_stations, extract_pm25_readings
from etl.weather_client import (
    fetch_historical_weather, fetch_forecast_weather, fetch_latest_weather,
    JAKARTA_LAT, JAKARTA_LON
)
from etl.transforms import (
    spku_to_dataframe, weather_hourly_to_dataframe, weather_daily_to_dataframe,
    compute_daily_summary, enrich_spku_with_risk
)

logger = logging.getLogger(__name__)

# Output paths (local filesystem or Azure Blob)
DATA_ROOT = os.environ.get('DATA_DIR', os.path.expanduser('~/airsafe-school/data'))


def run_etl_pipeline() -> dict:
    """
    Core ETL logic — pulls SPKU + Open-Meteo weather data, transforms, saves.
    
    Returns summary dict with stats and file paths.
    """
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    timestamp_str = now.strftime('%Y-%m-%d_%H%M%S')
    
    summary = {
        'run_at': now.isoformat(),
        'status': 'running',
        'files': [],
    }
    
    # ── Step 1: Pull SPKU Data ──────────────────────────────────────────
    logger.info("Step 1: Fetching SPKU data...")
    try:
        spku_data = fetch_all_stations()
        pm25_records = extract_pm25_readings(spku_data)
        spku_df = spku_to_dataframe(pm25_records)
        spku_df = enrich_spku_with_risk(spku_df)
        
        # Save raw snapshot
        raw_spku_dir = os.path.join(DATA_ROOT, 'raw', 'spku', 'snapshots')
        os.makedirs(raw_spku_dir, exist_ok=True)
        raw_path = os.path.join(raw_spku_dir, f'spku_etl_{timestamp_str}.json')
        with open(raw_path, 'w') as f:
            # Remove non-serializable fields
            serializable = {k: v for k, v in spku_data.items() if k != 'stations'}
            serializable['station_count'] = spku_data['total']
            serializable['pm25_records'] = len(pm25_records)
            json.dump(serializable, f, indent=2, default=str)
        summary['files'].append(raw_path)
        
        # Save processed PM2.5 CSV
        if not spku_df.empty:
            proc_dir = os.path.join(DATA_ROOT, 'processed', 'daily')
            os.makedirs(proc_dir, exist_ok=True)
            proc_path = os.path.join(proc_dir, f'spku_pm25_{today_str}.csv')
            spku_df.to_csv(proc_path, index=False)
            summary['files'].append(proc_path)
            
            # Also save as "latest"
            latest_path = os.path.join(proc_dir, 'spku_pm25_latest.csv')
            spku_df.to_csv(latest_path, index=False)
            summary['files'].append(latest_path)
        
        summary['spku'] = {
            'total_stations': spku_data['total'],
            'active_stations': spku_data['active'],
            'active_pm25': spku_data['active_pm25'],
            'valid_readings': len(spku_df),
        }
        logger.info(f"  SPKU: {spku_data['active_pm25']} active PM2.5 stations, {len(spku_df)} valid readings")
    except Exception as e:
        logger.error(f"  SPKU fetch failed: {e}")
        summary['spku_error'] = str(e)
    
    # ── Step 2: Pull Open-Meteo Weather (Archive) ──────────────────────
    logger.info("Step 2: Fetching Open-Meteo historical weather...")
    try:
        # Fetch last 7 days of archive (5-day delay, so ~2 days available)
        archive_end = (now - timedelta(days=5)).strftime('%Y-%m-%d')
        archive_start = (now - timedelta(days=12)).strftime('%Y-%m-%d')
        
        weather_data = fetch_historical_weather(
            start_date=archive_start,
            end_date=archive_end,
        )
        
        # Save hourly weather
        if 'hourly' in weather_data:
            hourly_df = weather_hourly_to_dataframe(weather_data['hourly'])
            if not hourly_df.empty:
                raw_weather_dir = os.path.join(DATA_ROOT, 'raw', 'weather')
                os.makedirs(raw_weather_dir, exist_ok=True)
                hourly_path = os.path.join(raw_weather_dir, f'weather_hourly_{today_str}.csv')
                hourly_df.to_csv(hourly_path, index=False)
                summary['files'].append(hourly_path)
                
                # Save latest
                latest_hourly = os.path.join(raw_weather_dir, 'weather_hourly_latest.csv')
                hourly_df.to_csv(latest_hourly, index=False)
                summary['files'].append(latest_hourly)
                
                summary['weather_hourly_records'] = len(hourly_df)
        
        # Save daily weather
        if 'daily' in weather_data:
            daily_df = weather_daily_to_dataframe(weather_data['daily'])
            if not daily_df.empty:
                raw_weather_dir = os.path.join(DATA_ROOT, 'raw', 'weather')
                os.makedirs(raw_weather_dir, exist_ok=True)
                daily_path = os.path.join(raw_weather_dir, f'weather_daily_{today_str}.csv')
                daily_df.to_csv(daily_path, index=False)
                summary['files'].append(daily_path)
                
                summary['weather_daily_records'] = len(daily_df)
        
        logger.info(f"  Weather: archive {archive_start} to {archive_end}")
    except Exception as e:
        logger.error(f"  Weather fetch failed: {e}")
        summary['weather_error'] = str(e)
    
    # ── Step 3: Pull Open-Meteo Forecast (covers gap) ──────────────────
    logger.info("Step 3: Fetching Open-Meteo forecast...")
    try:
        forecast_data = fetch_forecast_weather(forecast_days=3)
        
        if 'hourly' in forecast_data:
            forecast_df = weather_hourly_to_dataframe(forecast_data['hourly'])
            if not forecast_df.empty:
                raw_weather_dir = os.path.join(DATA_ROOT, 'raw', 'weather')
                os.makedirs(raw_weather_dir, exist_ok=True)
                forecast_path = os.path.join(raw_weather_dir, f'weather_forecast_{today_str}.csv')
                forecast_df.to_csv(forecast_path, index=False)
                summary['files'].append(forecast_path)
                summary['forecast_records'] = len(forecast_df)
        
        logger.info(f"  Forecast: {summary.get('forecast_records', 0)} records")
    except Exception as e:
        logger.error(f"  Forecast fetch failed: {e}")
        summary['forecast_error'] = str(e)
    
    # ── Step 4: Save ETL summary ───────────────────────────────────────
    summary['status'] = 'completed'
    summary_path = os.path.join(DATA_ROOT, 'processed', f'etl_summary_{today_str}.json')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    summary['files'].append(summary_path)
    
    logger.info(f"ETL completed. {len(summary['files'])} files saved.")
    return summary


# ── Azure Function Definitions (v2 model) ──────────────────────────────

if not LOCAL_MODE:
    @app.timer_trigger(
        schedule="0 */2 * * *",  # Every 2 hours
        arg_name="myTimer",
        run_on_startup=False,
    )
    def etl_daily(myTimer: func.TimerRequest) -> None:
        """Timer-triggered ETL: runs every 2 hours, pulls SPKU + weather."""
        if myTimer.past_due:
            logger.warning("Timer is past due!")
        
        summary = run_etl_pipeline()
        logger.info(f"ETL completed: {summary['status']}")

    @app.route(route="etl", methods=["POST"])
    def etl_on_demand(req: func.HttpRequest) -> func.HttpResponse:
        """HTTP-triggered ETL: run manually for testing."""
        import azure.functions as func
        
        try:
            summary = run_etl_pipeline()
            return func.HttpResponse(
                json.dumps(summary, default=str),
                mimetype="application/json",
                status_code=200,
            )
        except Exception as e:
            return func.HttpResponse(
                json.dumps({"error": str(e)}),
                mimetype="application/json",
                status_code=500,
            )


# ── Local Mode Entry Point ─────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    
    if '--local' in sys.argv or LOCAL_MODE:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
        os.environ['AIRSAFE_LOCAL_MODE'] = '1'
        
        print("=" * 60)
        print("AirSafe ETL Pipeline — LOCAL MODE")
        print("=" * 60)
        
        summary = run_etl_pipeline()
        
        print("\n" + "=" * 60)
        print("ETL SUMMARY")
        print("=" * 60)
        print(json.dumps(summary, indent=2, default=str))
        
        print(f"\n✅ {len(summary['files'])} files saved.")
        for f in summary['files']:
            print(f"  → {f}")
    else:
        print("Usage: python function_app.py --local")
        print("  Or set AIRSAFE_LOCAL_MODE=1 environment variable")
