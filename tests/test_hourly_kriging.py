"""Unit tests for hourly Kriging pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.spatial.hourly_kriging import (
    ISPU_STATION_COORDS,
    _HOURLY_KRIGING_CONFIG,
    load_hourly_data,
    hourly_kriging_interpolate,
)


def _make_hourly_csv(tmp_path, n_hours: int = 10) -> str:
    """Create a small synthetic hourly dataset CSV for testing."""
    station_names = list(ISPU_STATION_COORDS.keys())[:3]
    rows = []
    for h in range(n_hours):
        for sn in station_names:
            lat, lon = ISPU_STATION_COORDS[sn]
            rows.append({
                "datetime": f"2025-01-01 {h:02d}:00:00",
                "date": "2025-01-01",
                "hour": f"{h:02d}:00",
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
                "hc": np.nan,
                "kategori": "Sedang",
                "last_update": "test",
                "source_url": "test",
                "source_file": "test",
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
            })
    df = pd.DataFrame(rows)
    path = tmp_path / "hourly_test.csv"
    df.to_csv(path, index=False)
    return str(path)


def _make_schools(n: int = 5) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "npsn": [f"SCH{i:04d}" for i in range(n)],
        "nama_sekolah": [f"School {i}" for i in range(n)],
        "latitude": np.linspace(-6.15, -6.35, n),
        "longitude": np.linspace(106.75, 106.95, n),
    })


class TestLoadHourlyData:
    def test_maps_station_coordinates(self, tmp_path) -> None:
        path = _make_hourly_csv(tmp_path)
        df = load_hourly_data(path)
        for sn in df["station_name"].unique():
            expected_lat, expected_lon = ISPU_STATION_COORDS[sn]
            row = df[df["station_name"] == sn].iloc[0]
            assert abs(row["latitude"] - expected_lat) < 1e-6
            assert abs(row["longitude"] - expected_lon) < 1e-6

    def test_filters_date_range(self, tmp_path) -> None:
        path = _make_hourly_csv(tmp_path, n_hours=20)
        df = load_hourly_data(path, start_date="2025-01-01 05:00:00",
                              end_date="2025-01-01 14:00:00")
        assert df["datetime"].min() >= pd.Timestamp("2025-01-01 05:00:00")
        assert df["datetime"].max() <= pd.Timestamp("2025-01-01 14:00:00")

    def test_drops_rows_without_pm25(self, tmp_path) -> None:
        path = _make_hourly_csv(tmp_path)
        df = load_hourly_data(path)
        assert df["pm25"].notna().all()

    def test_clips_extreme_pm25(self, tmp_path) -> None:
        path = _make_hourly_csv(tmp_path, n_hours=3)
        # Inject a sensor malfunction value
        import csv
        rows = []
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["hour"] == "01:00":
                    row["pm25"] = "999.0"
                rows.append(row)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        df = load_hourly_data(str(path))
        assert df["pm25"].max() <= 300.0

    def test_includes_weather_columns(self, tmp_path) -> None:
        path = _make_hourly_csv(tmp_path)
        df = load_hourly_data(path)
        assert "temperature_2m" in df.columns
        assert "relative_humidity_2m" in df.columns
        assert "wind_speed_10m" in df.columns

    def test_file_not_found_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_hourly_data("/nonexistent/path.csv")

    def test_unknown_stations_filtered(self, tmp_path) -> None:
        rows = [{
            "datetime": "2025-01-01 00:00:00", "station_name": "UNKNOWN",
            "pm25": 50.0, "temperature_2m": 28.0, "relative_humidity_2m": 75,
            "precipitation": 0.0, "surface_pressure": 1010.0,
            "wind_speed_10m": 3.0, "wind_direction_10m": 180,
        }]
        path = tmp_path / "bad.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        with pytest.raises(ValueError, match="No rows matched"):
            load_hourly_data(str(path))


class TestHourlyKrigingInterpolate:
    def test_produces_output_for_each_hour(self, tmp_path) -> None:
        n_hours = 3
        path = _make_hourly_csv(tmp_path, n_hours=n_hours)
        hourly_df = load_hourly_data(path)
        schools = _make_schools(3)
        result = hourly_kriging_interpolate(hourly_df, schools)
        assert result["datetime"].nunique() == n_hours

    def test_includes_weather_columns(self, tmp_path) -> None:
        path = _make_hourly_csv(tmp_path, n_hours=3)
        hourly_df = load_hourly_data(path)
        schools = _make_schools(3)
        result = hourly_kriging_interpolate(hourly_df, schools)
        # Weather columns are NOT in output — that's the ML engineer's job
        assert "pm25_kriging" in result.columns
        assert "datetime" in result.columns

    def test_does_not_include_weather_or_temporal(self, tmp_path) -> None:
        path = _make_hourly_csv(tmp_path, n_hours=3)
        hourly_df = load_hourly_data(path)
        schools = _make_schools(3)
        result = hourly_kriging_interpolate(hourly_df, schools)
        # Output is minimal spatial layer only — no weather or temporal features
        assert "temperature_2m" not in result.columns
        assert "hour_num" not in result.columns
        assert "dayofweek" not in result.columns

    def test_output_has_kriging_columns(self, tmp_path) -> None:
        path = _make_hourly_csv(tmp_path, n_hours=3)
        hourly_df = load_hourly_data(path)
        schools = _make_schools(3)
        result = hourly_kriging_interpolate(hourly_df, schools)
        assert "pm25_kriging" in result.columns
        assert "kriging_std" in result.columns
        assert "variogram_model" in result.columns

    def test_row_count_matches_hours_times_schools(self, tmp_path) -> None:
        n_hours = 3
        n_schools = 3
        path = _make_hourly_csv(tmp_path, n_hours=n_hours)
        hourly_df = load_hourly_data(path)
        schools = _make_schools(n_schools)
        result = hourly_kriging_interpolate(hourly_df, schools)
        assert len(result) == n_hours * n_schools

    def test_with_corrector(self, tmp_path) -> None:
        from src.spatial.residual_corrector import ResidualCorrector
        path = _make_hourly_csv(tmp_path, n_hours=5)
        hourly_df = load_hourly_data(path)
        schools = _make_schools(3)

        corrector = ResidualCorrector()
        # Build synthetic LOSO results for training (need >= 5 rows)
        station_names = list(ISPU_STATION_COORDS.keys())[:3]
        loso_rows = []
        for h in range(4):
            for sn in station_names:
                lat, lon = ISPU_STATION_COORDS[sn]
                loso_rows.append({
                    "sensor_id": sn,
                    "latitude": lat, "longitude": lon,
                    "actual_pm25": 40.0 + h, "predicted_pm25": 38.0 + h,
                    "abs_error": 2.0, "squared_error": 4.0,
                    "variogram_used": "exponential",
                    "kriging_std": 15.0, "n_sensors_fold": 2,
                })
        loso_df = pd.DataFrame(loso_rows)
        sensor_df = pd.DataFrame([
            {"station_name": sn, "latitude": lat, "longitude": lon, "pm25": 40.0}
            for sn, (lat, lon) in ISPU_STATION_COORDS.items()
        ])
        corrector.train(loso_df, sensor_df)

        result = hourly_kriging_interpolate(hourly_df, schools, corrector=corrector)
        assert "pm25_corrected" in result.columns
