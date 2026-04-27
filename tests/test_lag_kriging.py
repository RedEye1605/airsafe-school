"""Tests for lag-dataset Kriging pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.spatial.hourly_kriging import ISPU_STATION_COORDS
from src.spatial.lag_kriging import (
    _average_station_features,
    _find_target_col,
    load_lag_dataset,
    lag_kriging_interpolate,
)


def _make_lag_csv(tmp_path, n_hours: int = 5, lag_hours: int = 6) -> str:
    """Create a small synthetic lag dataset CSV for testing."""
    station_names = list(ISPU_STATION_COORDS.keys())[:3]
    target_col = f"target_pm25_t_plus_{lag_hours}"
    rows = []
    for h in range(n_hours):
        for sn in station_names:
            lat, lon = ISPU_STATION_COORDS[sn]
            rows.append({
                "datetime": f"2025-01-01 {h:02d}:00:00",
                "date": "2025-01-01",
                "station_id": station_names.index(sn) + 4,
                "station_slug": sn.lower().replace(" ", "-"),
                "station_name": sn,
                "lokasi": f"SPKU {sn}",
                "pm25": 30.0 + h * 2 + np.random.normal(0, 3),
                "pm10": 50.0,
                "so2": 10.0,
                "co": 5.0,
                "o3": 15.0,
                "no2": 8.0,
                "kategori": np.nan,
                "year": 2025,
                "month": 1,
                "day": 1,
                "hour_num": h,
                "dayofweek": 2,
                "is_weekend": 0,
                "temperature_2m": 28.0 + h * 0.5,
                "relative_humidity_2m": 75,
                "precipitation": 0.0,
                "rain": 0.0,
                "surface_pressure": 1010.0,
                "wind_speed_10m": 3.5,
                "wind_direction_10m": 180,
                "pm25_raw": np.nan,
                "pm25_missing_flag": 0,
                "hour_sin": np.sin(2 * np.pi * h / 24),
                "hour_cos": np.cos(2 * np.pi * h / 24),
                "dow_sin": 0.0,
                "dow_cos": 0.0,
                "month_sin": 0.0,
                "month_cos": 0.0,
                "is_rush_morning": 0,
                "is_rush_evening": 0,
                "is_workhour": 0,
                "season_simple": "wet",
                "season_dry_flag": 0,
                "has_rain": 0,
                target_col: 35.0 + h * 2,
                "pm25_lag_1": 30.0,
                "pm25_lag_6": 28.0,
                "pm25_roll_mean_3": 31.0,
                "station_hour_mean_pm25": 40.0,
                "station_month_mean_pm25": 45.0,
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "lag_test.csv"
    df.to_csv(path, index=False)
    return str(path)


def _make_schools(n: int = 3) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "npsn": [f"SCH{i:04d}" for i in range(n)],
        "nama_sekolah": [f"School {i}" for i in range(n)],
        "latitude": np.linspace(-6.15, -6.35, n),
        "longitude": np.linspace(106.75, 106.95, n),
    })


class TestFindTargetCol:
    def test_finds_exact_match(self):
        assert _find_target_col(
            ["target_pm25_t_plus_6", "other"], 6,
        ) == "target_pm25_t_plus_6"

    def test_falls_back_to_single_target(self):
        assert _find_target_col(
            ["target_pm25_t_plus_12"], 6,
        ) == "target_pm25_t_plus_12"

    def test_raises_when_no_target(self):
        with pytest.raises(ValueError, match="not found"):
            _find_target_col(["other", "pm25"], 6)


class TestLoadLagDataset:
    def test_loads_valid_csv(self, tmp_path):
        path = _make_lag_csv(tmp_path)
        df = load_lag_dataset(path)
        assert "datetime" in df.columns
        assert "pm25" in df.columns
        assert df["pm25"].notna().all()

    def test_filters_date_range(self, tmp_path):
        path = _make_lag_csv(tmp_path, n_hours=20)
        df = load_lag_dataset(path, start_date="2025-01-01 05:00:00",
                              end_date="2025-01-01 14:00:00")
        assert df["datetime"].min() >= pd.Timestamp("2025-01-01 05:00:00")
        assert df["datetime"].max() <= pd.Timestamp("2025-01-01 14:00:00")

    def test_file_not_found_raises(self):
        with pytest.raises(FileNotFoundError):
            load_lag_dataset("/nonexistent/path.csv")


class TestAverageStationFeatures:
    def test_averages_numeric_cols(self):
        rows = pd.DataFrame({
            "station_name": ["A", "B", "C"],
            "temperature_2m": [28.0, 30.0, 32.0],
            "pm25_lag_1": [10.0, 20.0, 30.0],
        })
        result = _average_station_features(rows, exclude_cols={"station_name"})
        assert abs(result["temperature_2m"] - 30.0) < 1e-6
        assert abs(result["pm25_lag_1"] - 20.0) < 1e-6

    def test_takes_first_for_shared_cols(self):
        rows = pd.DataFrame({
            "station_name": ["A", "B"],
            "year": [2025, 2025],
            "season_simple": ["wet", "wet"],
        })
        result = _average_station_features(
            rows, exclude_cols={"station_name"},
        )
        assert result["year"] == 2025
        assert result["season_simple"] == "wet"

    def test_excludes_specified_cols(self):
        rows = pd.DataFrame({
            "station_name": ["A", "B"],
            "pm25": [30.0, 40.0],
        })
        result = _average_station_features(rows, exclude_cols={"station_name", "pm25"})
        assert "pm25" not in result.index


class TestLagKrigingInterpolate:
    def test_produces_output_for_each_timestamp(self, tmp_path):
        n_hours = 3
        path = _make_lag_csv(tmp_path, n_hours=n_hours)
        lag_df = load_lag_dataset(path)
        schools = _make_schools(3)
        result = lag_kriging_interpolate(lag_df, schools, "target_pm25_t_plus_6")
        assert result["datetime"].nunique() == n_hours

    def test_row_count_matches_timestamps_times_schools(self, tmp_path):
        n_hours = 3
        n_schools = 3
        path = _make_lag_csv(tmp_path, n_hours=n_hours)
        lag_df = load_lag_dataset(path)
        schools = _make_schools(n_schools)
        result = lag_kriging_interpolate(lag_df, schools, "target_pm25_t_plus_6")
        assert len(result) == n_hours * n_schools

    def test_includes_kriging_metadata(self, tmp_path):
        path = _make_lag_csv(tmp_path, n_hours=3)
        lag_df = load_lag_dataset(path)
        schools = _make_schools(3)
        result = lag_kriging_interpolate(lag_df, schools, "target_pm25_t_plus_6")
        assert "pm25_kriging_std" in result.columns
        assert "pm25_variogram_model" in result.columns
        assert "pm25_n_sensors" in result.columns

    def test_includes_target_column(self, tmp_path):
        path = _make_lag_csv(tmp_path, n_hours=3)
        lag_df = load_lag_dataset(path)
        schools = _make_schools(3)
        result = lag_kriging_interpolate(lag_df, schools, "target_pm25_t_plus_6")
        assert "target_pm25_t_plus_6" in result.columns
        assert "target_pm25_t_plus_6_kriging_std" in result.columns

    def test_includes_averaged_features(self, tmp_path):
        path = _make_lag_csv(tmp_path, n_hours=3)
        lag_df = load_lag_dataset(path)
        schools = _make_schools(3)
        result = lag_kriging_interpolate(lag_df, schools, "target_pm25_t_plus_6")
        assert "temperature_2m" in result.columns
        assert "pm25_lag_1" in result.columns
        assert "station_hour_mean_pm25" in result.columns

    def test_includes_school_identity(self, tmp_path):
        path = _make_lag_csv(tmp_path, n_hours=3)
        lag_df = load_lag_dataset(path)
        schools = _make_schools(3)
        result = lag_kriging_interpolate(lag_df, schools, "target_pm25_t_plus_6")
        assert "npsn" in result.columns
        assert result["npsn"].nunique() == 3
