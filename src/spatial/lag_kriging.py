"""
Lag-dataset Kriging spatial interpolation pipeline.

Reads the friend's temporal lag datasets (dataset_h6/h12/h24.csv) which have
hourly PM2.5 from 5 ISPU SPKU stations with 129+ engineered features, then
runs Kriging spatial interpolation per-timestamp to produce school-level PM2.5
estimates at all ~4,000 school locations.

Pipeline (per unique datetime):
    1. Extract 5 station rows for that datetime.
    2. Kriging-interpolate current PM2.5 to all school locations.
    3. Kriging-interpolate target PM2.5 (t+6/12/24) to all school locations.
    4. Average station-level features (weather, lags, rolling stats) across
       the 5 stations and broadcast to every school row.
    5. Attach school identity + temporal metadata.
    6. Append chunk to disk (CSV append mode).

Output: long-format CSV/Parquet with one row per (datetime, school).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.data.transforms import classify_pm25_hourly
from src.spatial.hourly_kriging import ISPU_STATION_COORDS
from src.spatial.kriging import KrigingConfig, kriging_interpolate
from src.spatial.residual_corrector import ResidualCorrector

logger = logging.getLogger(__name__)

_LAG_KRIGING_CONFIG = KrigingConfig(nlags=4, min_sensors=3)

# Columns identifying the station — dropped in school-level output
_STATION_IDENTITY_COLS = [
    "station_id", "station_slug", "station_name", "lokasi",
]

# Columns always the same across stations at a given datetime
_SHARED_TIME_COLS = [
    "date", "year", "month", "day", "hour_num", "dayofweek", "is_weekend",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_rush_morning", "is_rush_evening", "is_workhour",
    "season_simple", "season_dry_flag",
]

# Raw PM2.5 and missing flags — not meaningful at school level after Kriging
_PM25_SOURCE_COLS = [
    "pm25_raw", "pm25_missing_flag", "pm10_missing_flag",
    "so2_missing_flag", "co_missing_flag", "o3_missing_flag",
    "no2_missing_flag", "kategori",
]


def load_lag_dataset(
    csv_path: str | Path,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load and prepare a friend's lag dataset for Kriging.

    Reads the CSV, parses datetime, filters to date range, and validates
    that the expected 5 ISPU stations are present.

    Returns:
        DataFrame sorted by datetime then station_name.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Lag dataset not found: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=["datetime"])

    known = set(ISPU_STATION_COORDS)
    present = set(df["station_name"].unique())
    unknown = present - known
    if unknown:
        logger.warning("Dropping unknown stations: %s", unknown)
        df = df[df["station_name"].isin(known)].copy()

    if start_date:
        df = df[df["datetime"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["datetime"] <= pd.Timestamp(end_date)]

    if df.empty:
        raise ValueError("No data remaining after filtering.")

    df = df.dropna(subset=["pm25"]).copy()
    n_extreme = int((df["pm25"] > 300).sum())
    df["pm25"] = df["pm25"].clip(upper=300.0)
    if n_extreme:
        logger.info("Clipped %d PM2.5 values > 300 µg/m³ (sensor malfunctions)", n_extreme)
    df = df.sort_values(["datetime", "station_name"]).reset_index(drop=True)

    logger.info(
        "Loaded lag dataset: %d rows, %d stations, %s to %s",
        len(df), df["station_name"].nunique(),
        df["datetime"].min(), df["datetime"].max(),
    )
    return df


def _find_target_col(columns: list[str], lag_hours: int) -> str:
    """Find the target column for the given lag hours."""
    target = f"target_pm25_t_plus_{lag_hours}"
    if target in columns:
        return target
    candidates = [c for c in columns if c.startswith("target_pm25_t_plus_")]
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(
        f"Target column '{target}' not found. Available targets: {candidates}"
    )


def _average_station_features(
    station_rows: pd.DataFrame,
    exclude_cols: set[str],
) -> pd.Series:
    """Average numeric station-level features across the 5 stations.

    Non-numeric columns in _SHARED_TIME_COLS are taken from the first station.
    """
    averaged = {}
    for col in station_rows.columns:
        if col in exclude_cols:
            continue
        series = station_rows[col]
        if col in _SHARED_TIME_COLS:
            averaged[col] = series.iloc[0]
        elif pd.api.types.is_numeric_dtype(series):
            averaged[col] = series.mean()
        else:
            averaged[col] = series.iloc[0]
    return pd.Series(averaged)


def _kriging_for_column(
    station_rows: pd.DataFrame,
    school_df: pd.DataFrame,
    value_col: str,
    config: KrigingConfig,
    corrector: Optional[ResidualCorrector],
) -> Optional[pd.DataFrame]:
    """Run Kriging interpolation for a single value column."""
    sensors = station_rows[["station_name", "latitude", "longitude", value_col]].copy()
    sensors = sensors.dropna(subset=[value_col])
    if sensors.empty:
        return None

    try:
        result = kriging_interpolate(
            sensors, school_df, value_col=value_col, config=config,
        )
    except Exception as exc:
        logger.warning("Kriging failed for %s: %s", value_col, exc)
        return None

    if corrector is not None:
        try:
            result = corrector.correct(result, sensors, value_col=value_col)
        except Exception as exc:
            logger.warning("Correction failed for %s: %s", value_col, exc)

    return result


def _interpolate_one_timestamp(
    dt: pd.Timestamp,
    station_rows: pd.DataFrame,
    school_df: pd.DataFrame,
    config: KrigingConfig,
    target_col: str,
    corrector: Optional[ResidualCorrector],
) -> Optional[pd.DataFrame]:
    """Run Kriging for one timestamp: current PM2.5 + target PM2.5 → schools."""
    if len(station_rows) < 2:
        return None

    # Ensure station coordinates attached
    if "latitude" not in station_rows.columns:
        coord_map = {
            name: {"latitude": lat, "longitude": lon}
            for name, (lat, lon) in ISPU_STATION_COORDS.items()
        }
        station_rows = station_rows.copy()
        station_rows["latitude"] = station_rows["station_name"].map(
            lambda s: coord_map[s]["latitude"]
        )
        station_rows["longitude"] = station_rows["station_name"].map(
            lambda s: coord_map[s]["longitude"]
        )

    # Kriging current PM2.5
    pm25_result = _kriging_for_column(
        station_rows, school_df, "pm25", config, corrector,
    )
    if pm25_result is None:
        return None

    # Kriging target PM2.5
    target_result = _kriging_for_column(
        station_rows, school_df, target_col, config, corrector,
    )

    # Build school-level row
    n_schools = len(pm25_result)
    data = {
        "npsn": pm25_result["npsn"].values,
        "latitude": pm25_result["latitude"].values,
        "longitude": pm25_result["longitude"].values,
        "pm25": pm25_result["pm25_kriging"].values,
        "pm25_kriging_std": pm25_result["kriging_std"].values,
        "pm25_variogram_model": pm25_result["variogram_model"].values,
        "pm25_n_sensors": pm25_result["n_sensors"].values,
    }

    if "pm25_corrected" in pm25_result.columns:
        data["pm25_corrected"] = pm25_result["pm25_corrected"].values

    # Target
    if target_result is not None:
        data[target_col] = target_result[f"{target_col}_kriging"].values
        data[f"{target_col}_kriging_std"] = target_result["kriging_std"].values
    else:
        data[target_col] = np.nan
        data[f"{target_col}_kriging_std"] = np.nan

    out = pd.DataFrame(data, index=range(n_schools))

    # Average station features and broadcast to all schools
    exclude = (
        set(_STATION_IDENTITY_COLS)
        | set(_PM25_SOURCE_COLS)
        | {"pm25", target_col, "latitude", "longitude", "datetime"}
    )
    avg_features = _average_station_features(station_rows, exclude)

    # Build extra columns as a single DataFrame to avoid fragmentation
    extra = {}
    for col, val in avg_features.items():
        if col not in out.columns:
            extra[col] = val
    extra["datetime"] = dt

    extra_df = pd.DataFrame(extra, index=range(n_schools))
    out = pd.concat([out, extra_df], axis=1)

    # BMKG hourly risk classification
    out["risk_category"] = out["pm25"].apply(classify_pm25_hourly)

    return out


def lag_kriging_interpolate(
    lag_df: pd.DataFrame,
    school_df: pd.DataFrame,
    target_col: str,
    config: Optional[KrigingConfig] = None,
    corrector: Optional[ResidualCorrector] = None,
    progress_every: int = 500,
) -> pd.DataFrame:
    """Run Kriging interpolation for every timestamp in the lag dataset.

    For each unique datetime:
    1. Extract the 5 station rows.
    2. Kriging current PM2.5 → all school locations.
    3. Kriging target PM2.5 → all school locations.
    4. Average station features (weather, lags, rolling stats).
    5. Broadcast to school rows.

    Args:
        lag_df: From load_lag_dataset(). Must have datetime, station_name,
            pm25, target_col, plus station coordinates.
        school_df: Schools with npsn, latitude, longitude.
        target_col: Name of the target column (e.g. 'target_pm25_t_plus_6').
        config: KrigingConfig for the 5-sensor regime.
        corrector: Optional trained ResidualCorrector.
        progress_every: Log progress every N timestamps.

    Returns:
        Long-format DataFrame with one row per (datetime, school).
    """
    config = config or _LAG_KRIGING_CONFIG

    timestamps = sorted(lag_df["datetime"].unique())
    n_ts = len(timestamps)
    logger.info(
        "Running lag Kriging: %d timestamps, %d schools, target=%s",
        n_ts, len(school_df), target_col,
    )

    chunks: list[pd.DataFrame] = []
    for i, dt in enumerate(timestamps):
        station_rows = lag_df[lag_df["datetime"] == dt]
        result = _interpolate_one_timestamp(
            dt, station_rows, school_df, config, target_col, corrector,
        )
        if result is not None:
            chunks.append(result)

        if (i + 1) % progress_every == 0:
            logger.info("Progress: %d / %d timestamps", i + 1, n_ts)

    if not chunks:
        raise ValueError("No timestamps produced valid Kriging results.")

    output = pd.concat(chunks, ignore_index=True)
    logger.info(
        "Lag Kriging complete: %d rows (%d timestamps x %d schools)",
        len(output), n_ts, len(school_df),
    )
    return output


def lag_kriging_to_file(
    lag_df: pd.DataFrame,
    school_df: pd.DataFrame,
    output_path: str | Path,
    target_col: str,
    config: Optional[KrigingConfig] = None,
    corrector: Optional[ResidualCorrector] = None,
    progress_every: int = 500,
    chunk_timestamps: int = 100,
) -> Path:
    """Run lag Kriging and write results to file in chunks.

    Memory-efficient: flushes to disk every chunk_timestamps instead of
    accumulating the entire dataset in memory.

    Args:
        lag_df: From load_lag_dataset().
        school_df: Schools with npsn, latitude, longitude.
        output_path: Output file path (CSV or Parquet).
        target_col: Target column name.
        config: KrigingConfig.
        corrector: Optional trained ResidualCorrector.
        progress_every: Log progress every N timestamps.
        chunk_timestamps: Flush to disk every N timestamps.

    Returns:
        Path to the output file.
    """
    config = config or _LAG_KRIGING_CONFIG
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    timestamps = sorted(lag_df["datetime"].unique())
    n_ts = len(timestamps)
    logger.info(
        "Running lag Kriging (chunked): %d timestamps, %d schools → %s",
        n_ts, len(school_df), output_path,
    )

    is_parquet = output_path.suffix == ".parquet"
    wrote_header = False
    total_rows = 0
    chunk: list[pd.DataFrame] = []
    chunk_files: list[Path] = []

    for i, dt in enumerate(timestamps):
        station_rows = lag_df[lag_df["datetime"] == dt]
        result = _interpolate_one_timestamp(
            dt, station_rows, school_df, config, target_col, corrector,
        )
        if result is not None:
            chunk.append(result)

        if len(chunk) >= chunk_timestamps:
            batch = pd.concat(chunk, ignore_index=True)
            total_rows += len(batch)

            if is_parquet:
                part_path = output_path.with_suffix(
                    f".part{len(chunk_files)}.parquet"
                )
                part_path.parent.mkdir(parents=True, exist_ok=True)
                batch.to_parquet(part_path, index=False)
                chunk_files.append(part_path)
            else:
                batch.to_csv(
                    output_path, mode="a", header=not wrote_header, index=False,
                )
                wrote_header = True

            chunk = []
            logger.info(
                "Flushed chunk: %d total rows (%d / %d timestamps)",
                total_rows, i + 1, n_ts,
            )

        if (i + 1) % progress_every == 0:
            logger.info("Progress: %d / %d timestamps", i + 1, n_ts)

    # Flush remaining
    if chunk:
        batch = pd.concat(chunk, ignore_index=True)
        total_rows += len(batch)
        if is_parquet:
            part_path = output_path.with_suffix(
                f".part{len(chunk_files)}.parquet"
            )
            part_path.parent.mkdir(parents=True, exist_ok=True)
            batch.to_parquet(part_path, index=False)
            chunk_files.append(part_path)
        else:
            batch.to_csv(
                output_path, mode="a", header=not wrote_header, index=False,
            )

    # Concatenate part files into final Parquet (one-at-a-time to save memory)
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


def run_lag_pipeline(
    lag_path: str | Path,
    school_path: str | Path = "data/processed/schools/schools_geocoded.csv",
    output_path: Optional[str | Path] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lag_hours: int = 6,
    config: Optional[KrigingConfig] = None,
    corrector_path: Optional[str | Path] = None,
    output_format: str = "csv",
    chunk_timestamps: int = 100,
) -> Path:
    """End-to-end lag Kriging pipeline.

    Loads friend's lag dataset, runs Kriging per-timestamp, saves results.

    Args:
        lag_path: Path to friend's lag dataset (dataset_h6/h12/h24.csv).
        school_path: Path to geocoded schools CSV.
        output_path: Output file path.
        start_date: Start date filter (inclusive).
        end_date: End date filter (inclusive).
        lag_hours: Lag horizon (6, 12, or 24).
        config: KrigingConfig.
        corrector_path: Path to trained ResidualCorrector pickle.
        output_format: "csv" or "parquet".
        chunk_timestamps: Flush to disk every N timestamps.

    Returns:
        Path to the output file.
    """
    lag_path = Path(lag_path)
    school_path = Path(school_path)

    if output_path is None:
        suffix = ".parquet" if output_format == "parquet" else ".csv"
        output_path = Path(f"data/processed/lag_kriging_h{lag_hours}{suffix}")
    else:
        output_path = Path(output_path)

    lag_df = load_lag_dataset(lag_path, start_date=start_date, end_date=end_date)
    school_df = pd.read_csv(school_path)
    school_df = school_df.dropna(subset=["latitude", "longitude"]).copy()

    target_col = _find_target_col(list(lag_df.columns), lag_hours)

    corrector = None
    if corrector_path:
        corrector = ResidualCorrector.load(str(corrector_path))

    return lag_kriging_to_file(
        lag_df, school_df, output_path, target_col,
        config=config, corrector=corrector,
        chunk_timestamps=chunk_timestamps,
    )
