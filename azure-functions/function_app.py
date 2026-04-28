"""
AirSafe School — Azure Function App (Python v2 model).

Three functions:
    1. etl_timer / etl_http: ETL — scrape rendahemisi + Open-Meteo weather,
       produce dataset_master_spku_weather format, append to Blob.
    2. predict_timer / predict_http: Load LightGBM models, predict PM2.5 at
       5 ISPU stations, Kriging + residual correction to 4,215 schools.
    3. recommend_http: API — load predictions, generate Bahasa Indonesia
       recommendations for schools via OpenRouter LLM (template fallback).

Local testing:
    python function_app.py --local [etl|predict|recommend]
"""

import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

# Locate project root
_here = Path(__file__).resolve().parent
_PROJECT_ROOT = _here.parent if (_here.parent / "src").is_dir() else _here
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_env_file = _PROJECT_ROOT / ".env"
if _env_file.exists():
    from dotenv import load_dotenv
    load_dotenv(_env_file)

from src.config import (
    AIRSAFE_LOG_CONTAINER,
    AIRSAFE_PREDICT_CONTAINER,
    AIRSAFE_PROCESSED_CONTAINER,
    AIRSAFE_RAW_CONTAINER,
    AIRSAFE_REFERENCE_CONTAINER,
    DATA_DIR,
    LOCAL_MODE,
)
from src.data.blob_client import (
    is_blob_configured,
    save_dataframe_dual,
    save_json_dual,
)
from src.data.rendahemisi_client import STATIONS, fetch_recent_hours
from src.data.transforms import classify_pm25_hourly
from src.recommendations.engine import (
    bmkg_to_action,
    from_prediction_row,
    generate_recommendation,
    worst_risk,
)


def _download_blob_to_temp(container: str, blob_name: str) -> Path:
    """Download a blob to a temp file and return its path."""
    from src.data.blob_client import _get_service
    import tempfile

    svc = _get_service()
    bc = svc.get_blob_client(container=container, blob=blob_name)
    suffix = Path(blob_name).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp_path = Path(tmp.name)
    try:
        bc.download_blob().readinto(tmp)
        tmp.close()
        logger.info("Downloaded blob %s/%s → %s", container, blob_name, tmp.name)
        _temp_files.append(tmp_path)
        return tmp_path
    except Exception:
        tmp.close()
        tmp_path.unlink(missing_ok=True)
        raise


# Track temp files for cleanup after pipeline completes
_temp_files: list[Path] = []


def _cleanup_temp_files() -> None:
    for p in _temp_files:
        p.unlink(missing_ok=True)
    _temp_files.clear()

logger = logging.getLogger(__name__)

# ── Azure Functions v2 model registration ────────────────────────────────

if not LOCAL_MODE:
    import azure.functions as func

    app = func.FunctionApp()
else:
    app = None  # type: ignore[assignment]


# ── Station coordinates for Open-Meteo per-station weather ────────────────

_STATION_COORDS = {
    "dki1-bundaran-hi":   {"latitude": -6.1931, "longitude": 106.8230},
    "dki2-kelapa-gading": {"latitude": -6.1586, "longitude": 106.9050},
    "dki3-jagakarsa":     {"latitude": -6.3346, "longitude": 106.8228},
    "dki4-lubang-buaya":  {"latitude": -6.2908, "longitude": 106.9019},
    "dki5-kebun-jeruk":   {"latitude": -6.1951, "longitude": 106.7694},
}

_WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]


# ── ETL Pipeline (improved — matches Adit's format) ──────────────────────


def _fetch_openmeteo_hourly(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    import requests
    import time

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(_WEATHER_VARS),
        "timezone": "Asia/Jakarta",
    }
    for attempt in range(2):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            return resp.json().get("hourly", {})
        except requests.exceptions.HTTPError:
            if attempt == 0 and resp.status_code >= 500:
                logger.warning("Open-Meteo %d, retrying in 5s", resp.status_code)
                time.sleep(5)
                continue
            raise
    return {}


def _fetch_station_weather(slug: str, start_date: str, end_date: str) -> Optional[list[dict]]:
    coords = _STATION_COORDS.get(slug)
    if not coords:
        return None
    try:
        return _fetch_openmeteo_hourly(coords["latitude"], coords["longitude"], start_date, end_date)
    except Exception as exc:
        logger.error("Open-Meteo fetch failed for %s: %s", slug, exc)
        return None


def run_etl_pipeline(data_root: Optional[Path] = None) -> dict[str, Any]:
    """Core ETL — scrape rendahemisi + Open-Meteo weather, merge, save.

    Produces data matching Adit's dataset_master_spku_weather.csv format.

    Args:
        data_root: Root data directory (default: from config).

    Returns:
        Manifest dict with stats and file paths.
    """
    import pandas as pd

    if data_root is None:
        data_root = Path(DATA_DIR)

    now_utc = datetime.now(timezone.utc)
    now_wib = now_utc + timedelta(hours=7)
    ts_compact = now_utc.strftime("%Y%m%dT%H%M%SZ")
    date_str = now_utc.strftime("%Y-%m-%d")

    manifest: dict[str, Any] = {
        "run_id": f"etl_{ts_compact}",
        "started_at_utc": now_utc.isoformat(),
        "status": "running",
        "blob_enabled": is_blob_configured(),
        "files": [],
        "sources": {},
    }

    # Step 1: Rendahemisi — scrape latest data
    logger.info("Step 1: Scraping rendahemisi...")
    try:
        spku_df = fetch_recent_hours(hours=72)
        manifest["sources"]["rendahemisi"] = {
            "ok": not spku_df.empty,
            "rows": len(spku_df),
            "stations": spku_df["station_slug"].nunique() if not spku_df.empty else 0,
        }
    except Exception as exc:
        logger.error("Rendahemisi scrape failed: %s", exc)
        spku_df = pd.DataFrame()
        manifest["sources"]["rendahemisi"] = {"ok": False, "error": str(exc)}

    if spku_df.empty:
        manifest["status"] = "failed"
        manifest["error"] = "No rendahemisi data"
        return manifest

    # Save raw snapshot
    raw_blob = f"spku/date={date_str}/spku_{ts_compact}.csv"
    raw_local = data_root / "raw" / "spku" / f"spku_etl_{date_str}_{ts_compact}.csv"
    path = save_dataframe_dual(spku_df, raw_blob, raw_local, container=AIRSAFE_RAW_CONTAINER)
    manifest["files"].append(path)

    # Step 2: Open-Meteo weather per station
    logger.info("Step 2: Fetching Open-Meteo weather per station...")
    end_date = now_wib.strftime("%Y-%m-%d")
    start_date = (now_wib - timedelta(days=3)).strftime("%Y-%m-%d")

    weather_dfs = []
    for st in STATIONS:
        slug = st["slug"]
        try:
            hourly = _fetch_station_weather(slug, start_date, end_date)
            if hourly and "time" in hourly:
                wdf = pd.DataFrame(hourly)
                wdf["datetime"] = pd.to_datetime(wdf["time"])
                wdf["station_slug"] = slug
                wdf["station_name"] = st["station_name"]
                wdf = wdf.drop(columns=["time"], errors="ignore")
                weather_dfs.append(wdf)
        except Exception as exc:
            logger.error("Weather fetch failed for %s: %s", slug, exc)

    if weather_dfs:
        weather_df = pd.concat(weather_dfs, ignore_index=True)
        manifest["sources"]["weather"] = {
            "ok": True,
            "rows": len(weather_df),
            "stations": weather_df["station_slug"].nunique(),
        }
    else:
        weather_df = pd.DataFrame()
        manifest["sources"]["weather"] = {"ok": False, "error": "No weather data"}

    # Step 3: Merge pollutant + weather
    logger.info("Step 3: Merging pollutant + weather...")
    if not weather_df.empty:
        spku_df["datetime"] = pd.to_datetime(spku_df["datetime"]).dt.floor("h")
        weather_df["datetime"] = pd.to_datetime(weather_df["datetime"]).dt.floor("h")

        merged = spku_df.merge(
            weather_df,
            on=["station_slug", "station_name", "datetime"],
            how="left",
        )

        # Add temporal columns
        merged["date"] = merged["datetime"].dt.normalize()
        merged["year"] = merged["datetime"].dt.year
        merged["month"] = merged["datetime"].dt.month
        merged["day"] = merged["datetime"].dt.day
        merged["hour_num"] = merged["datetime"].dt.hour
        merged["dayofweek"] = merged["datetime"].dt.dayofweek
        merged["is_weekend"] = (merged["dayofweek"] >= 5).astype(int)
    else:
        merged = spku_df.copy()

    # Save merged dataset (Adit's format)
    proc_blob = f"daily/dataset_master_spku_weather_{date_str}.csv"
    proc_local = data_root / "processed" / "daily" / f"dataset_master_spku_weather_{date_str}.csv"
    path = save_dataframe_dual(merged, proc_blob, proc_local, container=AIRSAFE_PROCESSED_CONTAINER)
    manifest["files"].append(path)

    # Also save as "latest"
    latest_local = data_root / "processed" / "daily" / "dataset_master_spku_weather_latest.csv"
    path = save_dataframe_dual(merged, "daily/dataset_master_spku_weather_latest.csv", latest_local, container=AIRSAFE_PROCESSED_CONTAINER)
    manifest["files"].append(path)

    # Finalise manifest
    manifest["status"] = "completed"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["merged_rows"] = len(merged)

    manifest_blob = f"etl/date={date_str}/run_{ts_compact}.json"
    manifest_local = data_root / "processed" / f"etl_summary_{date_str}.json"
    path = save_json_dual(manifest, manifest_blob, manifest_local, container=AIRSAFE_LOG_CONTAINER)
    manifest["files"].append(path)

    logger.info("ETL completed. %d rows, %d files. Blob: %s",
                len(merged), len(manifest["files"]), is_blob_configured())
    _cleanup_temp_files()
    return manifest


# ── Predict Pipeline ──────────────────────────────────────────────────────


def _run_predict_pipeline(data_root: Optional[Path] = None) -> dict[str, Any]:
    """Load models, predict at 5 stations, Kriging to 4,215 schools."""
    import numpy as np
    import pandas as pd
    import pickle

    from src.spatial.hourly_kriging import ISPU_STATION_COORDS
    from src.spatial.kriging import KrigingConfig, kriging_interpolate
    from src.spatial.residual_corrector import ResidualCorrector
    from src.features.lag_features import build_prediction_features, prepare_model_input

    if data_root is None:
        data_root = Path(DATA_DIR)

    now_utc = datetime.now(timezone.utc)
    now_wib = now_utc + timedelta(hours=7)
    ts_compact = now_utc.strftime("%Y%m%dT%H%M%SZ")

    manifest: dict[str, Any] = {
        "run_id": f"predict_{ts_compact}",
        "started_at_utc": now_utc.isoformat(),
        "status": "running",
    }

    models_dir = _PROJECT_ROOT / "models"
    schools_path = data_root / "processed" / "schools" / "schools_geocoded.csv"

    # Fall back to Blob if local file missing (Azure deployment)
    if not schools_path.exists() and is_blob_configured():
        schools_path = _download_blob_to_temp(
            AIRSAFE_REFERENCE_CONTAINER, "schools/schools_geocoded.csv"
        )

    # Load models
    models = {}
    for h in [6, 12, 24]:
        model_path = models_dir / f"final_lgbm_h{h}.pkl"
        try:
            with open(model_path, "rb") as f:
                models[h] = pickle.load(f)
            logger.info("Loaded model h%d: %d features", h, models[h].n_features_)
        except Exception as exc:
            manifest["status"] = "failed"
            manifest["error"] = f"Failed to load model h{h}: {exc}"
            logger.error("Model load failed h%d: %s", h, exc)
            return manifest

    # Load residual corrector
    corrector_path = models_dir / "hourly_residual_corrector.pkl"
    if corrector_path.exists():
        corrector = ResidualCorrector.load(str(corrector_path))
    else:
        corrector = None
        logger.warning("No residual corrector found — skipping correction")

    # Load school locations
    try:
        schools = pd.read_csv(schools_path)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"Failed to read schools CSV: {exc}"
        logger.error("Schools CSV read failed: %s", exc)
        return manifest
    valid_schools = schools.dropna(subset=["latitude", "longitude"]).copy()
    logger.info("Loaded %d schools (%d with valid coords)", len(schools), len(valid_schools))

    # Load historical data for feature context
    master_path = data_root / "processed" / "daily" / "dataset_master_spku_weather_latest.csv"
    if not master_path.exists() and is_blob_configured():
        master_path = _download_blob_to_temp(
            AIRSAFE_PROCESSED_CONTAINER, "daily/dataset_master_spku_weather_latest.csv"
        )
    if not master_path.exists():
        # Try the full historical dataset from reference container
        if is_blob_configured():
            master_path = _download_blob_to_temp(
                AIRSAFE_REFERENCE_CONTAINER, "historical/dataset_master_spku_weather.csv"
            )
    if not master_path.exists():
        manifest["status"] = "failed"
        manifest["error"] = f"No master dataset found"
        return manifest

    try:
        master = pd.read_csv(master_path)
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = f"Failed to read master dataset: {exc}"
        logger.error("Master dataset read failed: %s", exc)
        return manifest
    master["datetime"] = pd.to_datetime(master["datetime"])
    cutoff = pd.Timestamp.now() - pd.Timedelta(hours=72)
    historical = master[master["datetime"] >= cutoff].copy()
    logger.info("Historical context: %d rows (last 72h)", len(historical))

    # Build features and predict per horizon
    station_predictions = []
    school_results = valid_schools[["npsn", "latitude", "longitude"]].copy() if "npsn" in valid_schools.columns else valid_schools[["latitude", "longitude"]].copy()

    for h in [6, 12, 24]:
        logger.info("Predicting horizon h%d...", h)
        features_df = build_prediction_features(historical, horizon=h)
        if features_df.empty:
            logger.warning("No features for h%d — skipping", h)
            school_results[f"pm25_h{h}"] = np.nan
            school_results[f"risk_h{h}"] = "TIDAK ADA DATA"
            continue

        X, feature_cols = prepare_model_input(
            features_df, f"target_pm25_t_plus_{h}",
            model_feature_order=models[h].feature_name_,
        )
        preds = models[h].predict(X)

        # Station predictions
        for i, row in features_df.iterrows():
            station_predictions.append({
                "station": row.get("station_name", ""),
                "slug": str(row.get("station_slug", "")),
                "horizon": h,
                "pm25_predicted": float(preds[i]),
            })

        # Kriging interpolation
        slug_to_coord = {
            slug: coords for slug, coords in zip(
                ["DKI1 Bundaran HI", "DKI2 Kelapa Gading", "DKI3 Jagakarsa",
                 "DKI4 Lubang Buaya", "DKI5 Kebun Jeruk"],
                ISPU_STATION_COORDS.values(),
            )
        }

        sensor_data = []
        for i, row in features_df.iterrows():
            name = row.get("station_name", "")
            coords = slug_to_coord.get(name)
            if coords:
                sensor_data.append({
                    "station_name": name,
                    "latitude": coords[0],
                    "longitude": coords[1],
                    "pm25": float(preds[i]),
                })

        if len(sensor_data) >= 3:
            sensor_df = pd.DataFrame(sensor_data)

            try:
                kriging_result = kriging_interpolate(
                    sensor_df, valid_schools,
                    config=KrigingConfig(nlags=4, min_sensors=3),
                )

                if corrector is not None:
                    kriging_result = corrector.correct(kriging_result, sensor_df)

                pm25_col = "pm25_corrected" if "pm25_corrected" in kriging_result.columns else "pm25_kriging"
                school_results[f"pm25_h{h}"] = kriging_result[pm25_col].values
                school_results[f"risk_h{h}"] = kriging_result[pm25_col].apply(classify_pm25_hourly).values
            except Exception as exc:
                logger.error("Kriging failed for h%d: %s", h, exc)
                school_results[f"pm25_h{h}"] = np.nan
                school_results[f"risk_h{h}"] = "TIDAK ADA DATA"
        else:
            logger.warning("Not enough sensors (%d) for Kriging h%d", len(sensor_data), h)
            school_results[f"pm25_h{h}"] = np.nan
            school_results[f"risk_h{h}"] = "TIDAK ADA DATA"

    # Build output JSON
    output = {
        "timestamp_utc": now_utc.isoformat(),
        "timestamp_wib": now_wib.isoformat(),
        "station_predictions": station_predictions,
        "n_schools": len(school_results),
    }

    # School predictions — convert to list of dicts
    school_list = []
    for _, row in school_results.iterrows():
        entry = {k: v for k, v in row.items() if pd.notna(v)}
        school_list.append(entry)
    output["school_predictions"] = school_list

    # Save to Blob (predictions container)
    date_str = now_utc.strftime("%Y-%m-%d")
    predict_blob = f"daily/predict_{date_str}_{ts_compact}.json"
    predict_local = data_root / "processed" / "daily" / f"predict_{date_str}_{ts_compact}.json"
    save_json_dual(output, predict_blob, predict_local, container=AIRSAFE_PREDICT_CONTAINER)

    # Also save latest
    latest_local = data_root / "processed" / "daily" / "predict_latest.json"
    save_json_dual(output, "daily/predict_latest.json", latest_local, container=AIRSAFE_PREDICT_CONTAINER)

    manifest["status"] = "completed"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["n_schools"] = len(school_results)
    manifest["n_stations"] = len(station_predictions)
    _cleanup_temp_files()
    return manifest


# ── Recommend Pipeline ───────────────────────────────────────────────────


def _load_predictions(data_root: Path) -> Optional[dict]:
    """Load latest predictions from local or Blob."""
    predict_path = data_root / "processed" / "daily" / "predict_latest.json"
    if not predict_path.exists() and is_blob_configured():
        predict_path = _download_blob_to_temp(
            AIRSAFE_PREDICT_CONTAINER, "daily/predict_latest.json"
        )
    if not predict_path.exists():
        return None
    with open(predict_path) as f:
        data = json.load(f)
    if not data.get("school_predictions"):
        return None
    return data


def _load_school_meta(data_root: Path) -> dict[str, dict]:
    """Load school metadata as {npsn_str: {nama_sekolah, kecamatan, jenjang}}."""
    import pandas as pd

    schools_path = data_root / "processed" / "schools" / "schools_geocoded.csv"
    if not schools_path.exists() and is_blob_configured():
        schools_path = _download_blob_to_temp(
            AIRSAFE_REFERENCE_CONTAINER, "schools/schools_geocoded.csv"
        )
    if not schools_path.exists():
        return {}
    df = pd.read_csv(schools_path)
    meta = {}
    for _, row in df.iterrows():
        npsn = str(int(row.get("npsn", 0))) if pd.notna(row.get("npsn")) else ""
        if not npsn:
            continue
        meta[npsn] = {
            "nama_sekolah": str(row.get("nama_sekolah", "")),
            "kecamatan": str(row.get("kecamatan", "")),
            "jenjang": str(row.get("jenjang", "")),
        }
    return meta


def _generate_recommendations(
    predictions: dict,
    school_meta: dict[str, dict],
) -> list[dict]:
    """Generate recommendations for all schools in predictions."""
    results = []
    for school in predictions.get("school_predictions", []):
        npsn = str(school.get("npsn", ""))
        meta = school_meta.get(npsn, {})
        adapted = from_prediction_row(
            school,
            school_name=meta.get("nama_sekolah", npsn),
            district=meta.get("kecamatan", ""),
        )
        rec = generate_recommendation(adapted)
        results.append({
            "npsn": npsn,
            "nama_sekolah": meta.get("nama_sekolah", ""),
            "kecamatan": meta.get("kecamatan", ""),
            "jenjang": meta.get("jenjang", ""),
            "latitude": school.get("latitude"),
            "longitude": school.get("longitude"),
            "pm25_h6": school.get("pm25_h6"),
            "pm25_h12": school.get("pm25_h12"),
            "pm25_h24": school.get("pm25_h24"),
            "risk_h6": school.get("risk_h6", ""),
            "risk_h12": school.get("risk_h12", ""),
            "risk_h24": school.get("risk_h24", ""),
            "risk_action": rec["risk_level"],
            "recommendation": rec,
        })
    return results


def _run_recommend_pipeline(data_root: Optional[Path] = None) -> dict[str, Any]:
    """Generate recommendations for all schools and save to Blob."""
    if data_root is None:
        data_root = Path(DATA_DIR)

    now_utc = datetime.now(timezone.utc)
    ts_compact = now_utc.strftime("%Y%m%dT%H%M%SZ")

    manifest: dict[str, Any] = {
        "run_id": f"recommend_{ts_compact}",
        "started_at_utc": now_utc.isoformat(),
        "status": "running",
    }

    predictions = _load_predictions(data_root)
    if not predictions:
        manifest["status"] = "failed"
        manifest["error"] = "No predictions available"
        return manifest

    school_meta = _load_school_meta(data_root)
    recommendations = _generate_recommendations(predictions, school_meta)

    # Risk summary
    risk_counts: dict[str, int] = {}
    for r in recommendations:
        action = r["risk_action"]
        risk_counts[action] = risk_counts.get(action, 0) + 1

    output = {
        "timestamp_utc": predictions.get("timestamp_utc"),
        "timestamp_wib": predictions.get("timestamp_wib"),
        "generated_at_utc": now_utc.isoformat(),
        "station_predictions": predictions.get("station_predictions", []),
        "n_schools": len(recommendations),
        "risk_summary": risk_counts,
        "recommendations": recommendations,
    }

    date_str = now_utc.strftime("%Y-%m-%d")
    rec_blob = f"daily/recommend_{date_str}_{ts_compact}.json"
    rec_local = data_root / "processed" / "daily" / f"recommend_{date_str}_{ts_compact}.json"
    save_json_dual(output, rec_blob, rec_local, container=AIRSAFE_PREDICT_CONTAINER)

    latest_local = data_root / "processed" / "daily" / "recommend_latest.json"
    save_json_dual(output, "daily/recommend_latest.json", latest_local, container=AIRSAFE_PREDICT_CONTAINER)

    manifest["status"] = "completed"
    manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["n_schools"] = len(recommendations)
    manifest["risk_summary"] = risk_counts
    logger.info("Recommend completed. %d schools, risk: %s", len(recommendations), risk_counts)
    _cleanup_temp_files()
    return manifest


def _build_recommend_response(
    data_root: Path,
    npsn: Optional[str] = None,
    district: Optional[str] = None,
) -> tuple[dict, int]:
    """Build recommendation response, optionally filtered. Returns (data, status_code)."""
    predictions = _load_predictions(data_root)
    if not predictions:
        return {"error": "No predictions available. Run predict pipeline first."}, 503

    school_meta = _load_school_meta(data_root)

    if npsn:
        npsn_str = str(npsn)
        schools = predictions.get("school_predictions", [])
        school = next((s for s in schools if str(s.get("npsn", "")) == npsn_str), None)
        if not school:
            return {"error": f"School NPSN {npsn} not found"}, 404
        meta = school_meta.get(npsn_str, {})
        adapted = from_prediction_row(
            school,
            school_name=meta.get("nama_sekolah", npsn_str),
            district=meta.get("kecamatan", ""),
        )
        rec = generate_recommendation(adapted)
        return {
            "school": {
                "npsn": npsn_str,
                "nama_sekolah": meta.get("nama_sekolah", ""),
                "kecamatan": meta.get("kecamatan", ""),
                "jenjang": meta.get("jenjang", ""),
                "latitude": school.get("latitude"),
                "longitude": school.get("longitude"),
                "pm25_h6": school.get("pm25_h6"),
                "pm25_h12": school.get("pm25_h12"),
                "pm25_h24": school.get("pm25_h24"),
                "risk_h6": school.get("risk_h6", ""),
                "risk_h12": school.get("risk_h12", ""),
                "risk_h24": school.get("risk_h24", ""),
                "risk_action": rec["risk_level"],
            },
            "recommendation": rec,
        }, 200

    # District filter — uses template fallback to avoid LLM timeout
    if district:
        from src.recommendations.engine import RecommendationInput, _generate_with_template
        district_lower = district.lower()
        recommendations = []
        for school in predictions.get("school_predictions", []):
            npsn_val = str(school.get("npsn", ""))
            meta = school_meta.get(npsn_val, {})
            if meta.get("kecamatan", "").lower() != district_lower:
                continue
            adapted = from_prediction_row(
                school,
                school_name=meta.get("nama_sekolah", npsn_val),
                district=meta.get("kecamatan", ""),
            )
            inp_obj = RecommendationInput(**adapted)
            rec = _generate_with_template(inp_obj)
            recommendations.append({
                "npsn": npsn_val,
                "nama_sekolah": meta.get("nama_sekolah", ""),
                "kecamatan": meta.get("kecamatan", ""),
                "pm25_h6": school.get("pm25_h6"),
                "pm25_h12": school.get("pm25_h12"),
                "pm25_h24": school.get("pm25_h24"),
                "risk_action": rec["risk_level"],
                "recommendation": rec,
            })
        return {
            "district": district,
            "n_schools": len(recommendations),
            "recommendations": recommendations,
        }, 200

    # Summary — compute risk counts directly from predictions (no LLM)
    schools = predictions.get("school_predictions", [])
    risk_counts: dict[str, int] = {}
    for school in schools:
        action = bmkg_to_action(worst_risk(
            school.get("risk_h6", ""),
            school.get("risk_h12", ""),
            school.get("risk_h24", ""),
        ))
        risk_counts[action] = risk_counts.get(action, 0) + 1

    return {
        "timestamp_utc": predictions.get("timestamp_utc"),
        "timestamp_wib": predictions.get("timestamp_wib"),
        "n_schools": len(schools),
        "risk_summary": risk_counts,
        "station_predictions": predictions.get("station_predictions", []),
    }, 200


def _error_response(data: dict, status: int = 500) -> "func.HttpResponse":
    """Build a JSON error response (Azure Functions only)."""
    if LOCAL_MODE or app is None:
        return data  # type: ignore[return-value]
    return func.HttpResponse(
        json.dumps(data, default=str),
        mimetype="application/json",
        status_code=status,
    )


# ── Azure Function endpoints ─────────────────────────────────────────────

if not LOCAL_MODE and app is not None:
    @app.function_name(name="etl_timer")
    @app.timer_trigger(
        schedule="%ETL_SCHEDULE%",
        arg_name="myTimer",
        run_on_startup=False,
    )
    def etl_timer(myTimer: func.TimerRequest) -> None:
        if myTimer.past_due:
            logger.warning("Timer is past due!")
        manifest = run_etl_pipeline()
        logger.info("ETL finished: %s", manifest["status"])

    @app.function_name(name="etl_http")
    @app.route(route="etl", methods=["POST"])
    def etl_http(req: func.HttpRequest) -> func.HttpResponse:
        try:
            manifest = run_etl_pipeline()
            return func.HttpResponse(
                json.dumps(manifest, default=str, indent=2),
                mimetype="application/json",
                status_code=200 if manifest["status"] == "completed" else 500,
            )
        except Exception as exc:
            logger.exception("ETL HTTP failed")
            return _error_response({"error": str(exc), "type": type(exc).__name__})

    @app.function_name(name="predict_timer")
    @app.timer_trigger(
        schedule="0 0 15 * * *",
        arg_name="myTimer",
        run_on_startup=False,
    )
    def predict_timer(myTimer: func.TimerRequest) -> None:
        if myTimer.past_due:
            logger.warning("Predict timer is past due!")
        manifest = _run_predict_pipeline()
        logger.info("Predict finished: %s", manifest["status"])

    @app.function_name(name="predict_http")
    @app.route(route="predict", methods=["POST"])
    def predict_http(req: func.HttpRequest) -> func.HttpResponse:
        try:
            manifest = _run_predict_pipeline()
            return func.HttpResponse(
                json.dumps(manifest, default=str, indent=2),
                mimetype="application/json",
                status_code=200 if manifest["status"] == "completed" else 500,
            )
        except Exception as exc:
            logger.exception("Predict HTTP failed")
            return _error_response({"error": str(exc), "type": type(exc).__name__})

    @app.function_name(name="recommend_http")
    @app.route(route="recommend", methods=["GET"])
    def recommend_http(req: func.HttpRequest) -> func.HttpResponse:
        try:
            data_root = Path(DATA_DIR)
            npsn = req.params.get("npsn")
            district = req.params.get("district")
            data, status = _build_recommend_response(data_root, npsn=npsn, district=district)
            return func.HttpResponse(
                json.dumps(data, default=str, indent=2),
                mimetype="application/json",
                status_code=status,
            )
        except Exception as exc:
            logger.exception("Recommend HTTP failed")
            return _error_response({"error": str(exc), "type": type(exc).__name__})


# ── Local entry point ───────────────────────────────────────────────────

if __name__ == "__main__":
    if "--local" in sys.argv or LOCAL_MODE:
        from dotenv import load_dotenv
        from src.utils import setup_logging

        load_dotenv(_PROJECT_ROOT / ".env")
        setup_logging()
        import os
        os.environ["AIRSAFE_LOCAL_MODE"] = "1"

        mode = sys.argv[sys.argv.index("--local") + 1] if len(sys.argv) > sys.argv.index("--local") + 1 else "etl"

        print("=" * 60)
        print(f"AirSafe Pipeline — LOCAL MODE ({mode})")
        print(f"Blob Storage: {'ENABLED' if is_blob_configured() else 'NOT CONFIGURED'}")
        print("=" * 60)

        if mode == "predict":
            manifest = _run_predict_pipeline()
        elif mode == "recommend":
            manifest = _run_recommend_pipeline()
        else:
            manifest = run_etl_pipeline()

        print(json.dumps(manifest, indent=2, default=str))
    else:
        print("Usage: python function_app.py --local [etl|predict|recommend]")
        print("  Or set AIRSAFE_LOCAL_MODE=1")
