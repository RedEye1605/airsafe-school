"""
Hourly temporal-spatial Kriging pipeline.

Runs Kriging interpolation per-hour using real hourly PM2.5 from the
5 ISPU SPKU stations (DKI1-DKI5). Produces hourly PM2.5 estimates at
all school locations, optionally with residual correction.

Data source: ``dataset_master_spku_weather.csv`` from
``rendahemisi.jakarta.go.id`` — real hourly SPKU readings with
Open-Meteo weather already merged.

Pipeline steps (per hour):
    1. Extract station subset with valid PM2.5 at that hour.
    2. Build sensor DataFrame with lat, lon, pm25.
    3. Call ``kriging_interpolate()`` for all school locations.
    4. Optionally apply ``ResidualCorrector.correct()``.
    5. Attach datetime + averaged weather metadata.

Output: long-format CSV/Parquet with one row per (datetime, school).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.spatial.kriging import KrigingConfig, kriging_interpolate
from src.spatial.residual_corrector import ResidualCorrector

logger = logging.getLogger(__name__)

ISPU_STATION_COORDS: dict[str, tuple[float, float]] = {
    "DKI1 Bundaran HI": (-6.195459, 106.822731),
    "DKI2 Kelapa Gading": (-6.154085, 106.908249),
    "DKI3 Jagakarsa": (-6.336216, 106.818082),
    "DKI4 Lubang Buaya": (-6.290800, 106.908501),
    "DKI5 Kebun Jeruk": (-6.207255, 106.752560),
}

_WEATHER_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "wind_speed_10m",
    "wind_direction_10m",
]

_HOURLY_KRIGING_CONFIG = KrigingConfig(
    nlags=4,
    min_sensors=3,
)


def load_hourly_data(
    csv_path: str | Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load and prepare the hourly ISPU dataset for Kriging.

    Reads the friend's CSV, maps station names to coordinates from our
    sensor catalog, filters to valid PM2.5 rows, and slices to date range.

    Returns:
        DataFrame with columns: datetime, station_name, latitude,
        longitude, pm25, plus weather columns.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["datetime"])

    required = {"datetime", "station_name", "pm25"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Filter to known ISPU stations only
    known = set(ISPU_STATION_COORDS)
    df = df[df["station_name"].isin(known)].copy()

    if df.empty:
        raise ValueError(f"No rows matched known ISPU stations: {known}")

    # Attach coordinates from our catalog
    coord_map = {
        name: {"latitude": lat, "longitude": lon}
        for name, (lat, lon) in ISPU_STATION_COORDS.items()
    }
    df["latitude"] = df["station_name"].map(lambda s: coord_map[s]["latitude"])
    df["longitude"] = df["station_name"].map(lambda s: coord_map[s]["longitude"])

    # Date filtering
    if start_date:
        df = df[df["datetime"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["datetime"] <= pd.Timestamp(end_date)]

    # Keep only rows with valid PM2.5 and clip sensor malfunctions
    df = df.dropna(subset=["pm25"]).copy()
    n_extreme = int((df["pm25"] > 300).sum())
    df["pm25"] = df["pm25"].clip(upper=300.0)
    if n_extreme:
        logger.info("Clipped %d PM2.5 values > 300 µg/m³ (sensor malfunctions)", n_extreme)

    select_cols = [
        "datetime", "station_name", "latitude", "longitude", "pm25",
        *_WEATHER_COLUMNS,
    ]
    available = [c for c in select_cols if c in df.columns]

    logger.info(
        "Loaded hourly data: %d rows, %d stations, %s to %s",
        len(df), df["station_name"].nunique(),
        df["datetime"].min(), df["datetime"].max(),
    )

    return df[available].reset_index(drop=True)


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, day, hour_num, dayofweek, is_weekend columns."""
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour_num"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    return df


def _interpolate_one_hour(
    dt: pd.Timestamp,
    hour_data: pd.DataFrame,
    school_df: pd.DataFrame,
    config: KrigingConfig,
    corrector: Optional[ResidualCorrector],
) -> Optional[pd.DataFrame]:
    """Run Kriging + correction for a single hour."""
    sensors = hour_data[["station_name", "latitude", "longitude", "pm25"]].rename(
        columns={"station_name": "station_name_orig"},
    )
    if sensors.empty:
        return None

    try:
        result = kriging_interpolate(
            sensors, school_df, value_col="pm25", config=config,
        )
    except Exception as exc:
        logger.warning("Kriging failed for %s: %s", dt, exc)
        return None

    if corrector is not None:
        try:
            result = corrector.correct(result, sensors)
        except Exception as exc:
            logger.warning("Correction failed for %s: %s", dt, exc)

    # Keep only spatial interpolation columns — weather/feature engineering
    # is the ML engineer's responsibility.
    keep = [
        "npsn", "latitude", "longitude",
        "pm25_kriging", "kriging_std", "kriging_variance",
        "variogram_model", "n_sensors",
    ]
    if "pm25_corrected" in result.columns:
        keep.append("pm25_corrected")
    if "residual_pred" in result.columns:
        keep.append("residual_pred")

    result = result[[c for c in keep if c in result.columns]].copy()
    result["datetime"] = dt
    return result


def hourly_kriging_interpolate(
    hourly_df: pd.DataFrame,
    school_df: pd.DataFrame,
    config: Optional[KrigingConfig] = None,
    corrector: Optional[ResidualCorrector] = None,
    progress_every: int = 200,
) -> pd.DataFrame:
    """Run Kriging interpolation for every hour in hourly_df.

    For each unique datetime:
    1. Extract the station subset with valid PM2.5 at that hour.
    2. Build sensor_df with lat, lon, pm25.
    3. Call kriging_interpolate(sensors, schools, config).
    4. Optionally apply ResidualCorrector.correct().
    5. Attach datetime and weather metadata.

    Args:
        hourly_df: From load_hourly_data(). Must have datetime, station_name,
            latitude, longitude, pm25 columns.
        school_df: Schools with npsn, latitude, longitude.
        config: KrigingConfig for the 5-sensor regime.
        corrector: Optional trained ResidualCorrector.
        progress_every: Log progress every N hours.

    Returns:
        Long-format DataFrame with one row per (datetime, school).
    """
    config = config or _HOURLY_KRIGING_CONFIG

    hours = sorted(hourly_df["datetime"].unique())
    n_hours = len(hours)
    logger.info("Running hourly Kriging: %d hours, %d schools", n_hours, len(school_df))

    chunks: list[pd.DataFrame] = []

    for i, dt in enumerate(hours):
        hour_data = hourly_df[hourly_df["datetime"] == dt]
        result = _interpolate_one_hour(dt, hour_data, school_df, config, corrector)
        if result is not None:
            chunks.append(result)

        if (i + 1) % progress_every == 0:
            logger.info("Progress: %d / %d hours", i + 1, n_hours)

    if not chunks:
        raise ValueError("No hours produced valid Kriging results.")

    output = pd.concat(chunks, ignore_index=True)

    logger.info("Hourly Kriging complete: %d rows (%d hours x %d schools)",
                len(output), n_hours, len(school_df))

    return output


def hourly_kriging_to_file(
    hourly_df: pd.DataFrame,
    school_df: pd.DataFrame,
    output_path: str | Path,
    config: Optional[KrigingConfig] = None,
    corrector: Optional[ResidualCorrector] = None,
    progress_every: int = 200,
    chunk_hours: int = 200,
) -> Path:
    """Run hourly Kriging and write results to file in chunks.

    Memory-efficient alternative to hourly_kriging_interpolate() for
    large datasets. Writes intermediate results every chunk_hours to
    avoid accumulating the entire dataset in memory.

    Args:
        hourly_df: From load_hourly_data().
        school_df: Schools with npsn, latitude, longitude.
        output_path: Output file path (CSV or Parquet).
        config: KrigingConfig for the 5-sensor regime.
        corrector: Optional trained ResidualCorrector.
        progress_every: Log progress every N hours.
        chunk_hours: Flush to disk every N hours.

    Returns:
        Path to the output file.
    """
    config = config or _HOURLY_KRIGING_CONFIG
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hours = sorted(hourly_df["datetime"].unique())
    n_hours = len(hours)
    logger.info(
        "Running hourly Kriging (chunked): %d hours, %d schools → %s",
        n_hours, len(school_df), output_path,
    )

    is_parquet = output_path.suffix == ".parquet"
    wrote_header = False
    total_rows = 0
    chunk: list[pd.DataFrame] = []
    chunk_files: list[Path] = []

    for i, dt in enumerate(hours):
        hour_data = hourly_df[hourly_df["datetime"] == dt]
        result = _interpolate_one_hour(dt, hour_data, school_df, config, corrector)
        if result is not None:
            chunk.append(result)

        if len(chunk) >= chunk_hours:
            batch = pd.concat(chunk, ignore_index=True)
            total_rows += len(batch)

            if is_parquet:
                part_path = output_path.with_suffix(
                    f".part{len(chunk_files)}.parquet"
                )
                batch.to_parquet(part_path, index=False)
                chunk_files.append(part_path)
            else:
                batch.to_csv(output_path, mode="a", header=not wrote_header, index=False)
                wrote_header = True

            chunk = []
            logger.info("Flushed chunk: %d total rows (%d / %d hours)",
                        total_rows, i + 1, n_hours)

        if (i + 1) % progress_every == 0:
            logger.info("Progress: %d / %d hours", i + 1, n_hours)

    # Flush remaining
    if chunk:
        batch = pd.concat(chunk, ignore_index=True)
        total_rows += len(batch)
        if is_parquet:
            part_path = output_path.with_suffix(
                f".part{len(chunk_files)}.parquet"
            )
            batch.to_parquet(part_path, index=False)
            chunk_files.append(part_path)
        else:
            batch.to_csv(output_path, mode="a", header=not wrote_header, index=False)

    # Concatenate part files into final Parquet
    if is_parquet and chunk_files:
        if len(chunk_files) == 1:
            chunk_files[0].rename(output_path)
        else:
            parts = [pd.read_parquet(f) for f in chunk_files]
            full = pd.concat(parts, ignore_index=True)
            full.to_parquet(output_path, index=False)
            del parts, full
            for f in chunk_files:
                f.unlink(missing_ok=True)

    logger.info("Output saved: %s (%d rows)", output_path, total_rows)
    return output_path


def run_hourly_pipeline(
    data_path: str | Path = "data/friend_model/Data Lag/dataset_h6.csv",
    school_path: str | Path = "data/processed/schools/schools_geocoded.csv",
    output_path: Optional[str | Path] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Optional[KrigingConfig] = None,
    corrector_path: Optional[str | Path] = None,
    output_format: str = "csv",
) -> Path:
    """End-to-end hourly Kriging pipeline.

    Loads data, runs interpolation, saves results.

    Args:
        data_path: Path to friend's hourly dataset.
        school_path: Path to geocoded schools CSV.
        output_path: Output file path.
        start_date: Start date filter (inclusive).
        end_date: End date filter (inclusive).
        config: KrigingConfig. Defaults to nlags=4, min_sensors=3.
        corrector_path: Path to a trained ResidualCorrector pickle.
        output_format: "csv" or "parquet".

    Returns:
        Path to the output file.
    """
    data_path = Path(data_path)
    school_path = Path(school_path)

    if output_path is None:
        suffix = ".parquet" if output_format == "parquet" else ".csv"
        output_path = Path(f"data/processed/hourly_kriging_predictions{suffix}")
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    hourly_df = load_hourly_data(data_path, start_date=start_date, end_date=end_date)
    school_df = pd.read_csv(school_path)

    corrector = None
    if corrector_path:
        corrector = ResidualCorrector.load(str(corrector_path))

    return hourly_kriging_to_file(
        hourly_df, school_df, output_path,
        config=config, corrector=corrector,
        chunk_hours=200,
    )
