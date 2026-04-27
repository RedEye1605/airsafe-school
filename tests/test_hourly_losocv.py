"""Unit tests for hourly LOSOCV."""

import numpy as np
import pandas as pd
import pytest

from src.spatial.hourly_kriging import ISPU_STATION_COORDS
from src.spatial.hourly_losocv import HourlyLosoMetrics, hourly_losocv


def _make_hourly_df(n_hours: int = 5) -> pd.DataFrame:
    """Create synthetic hourly data for testing."""
    station_names = list(ISPU_STATION_COORDS.keys())[:3]
    rows = []
    np.random.seed(42)
    for h in range(n_hours):
        for sn in station_names:
            lat, lon = ISPU_STATION_COORDS[sn]
            rows.append({
                "datetime": pd.Timestamp(f"2025-01-01 {h:02d}:00:00"),
                "station_name": sn,
                "latitude": lat,
                "longitude": lon,
                "pm25": 30.0 + h * 2 + np.random.normal(0, 3),
                "temperature_2m": 28.0,
                "relative_humidity_2m": 75,
                "wind_speed_10m": 3.5,
            })
    return pd.DataFrame(rows)


class TestHourlyLosocv:
    def test_returns_correct_shape(self) -> None:
        df = _make_hourly_df(n_hours=3)
        n_stations = df["station_name"].nunique()
        results, metrics = hourly_losocv(df)
        assert len(results) == 3 * n_stations

    def test_per_prediction_columns(self) -> None:
        df = _make_hourly_df(n_hours=3)
        results, _ = hourly_losocv(df)
        expected = [
            "datetime", "sensor_id", "latitude", "longitude",
            "actual_pm25", "predicted_pm25", "abs_error",
            "squared_error", "variogram_used",
        ]
        for col in expected:
            assert col in results.columns, f"Missing column: {col}"

    def test_metrics_are_reasonable(self) -> None:
        df = _make_hourly_df(n_hours=5)
        _, metrics = hourly_losocv(df)
        assert metrics.mae >= 0
        assert metrics.rmse >= metrics.mae
        assert metrics.n_predictions > 0
        assert metrics.n_hours == 5

    def test_max_hours_subsampling(self) -> None:
        df = _make_hourly_df(n_hours=10)
        results, metrics = hourly_losocv(df, max_hours=3)
        assert metrics.n_hours <= 3

    def test_per_sensor_mae(self) -> None:
        df = _make_hourly_df(n_hours=5)
        _, metrics = hourly_losocv(df)
        assert len(metrics.per_sensor_mae) == df["station_name"].nunique()
        for mae in metrics.per_sensor_mae.values():
            assert mae >= 0

    def test_fallback_count(self) -> None:
        """With min_sensors=3, 3 stations means no fallback."""
        df = _make_hourly_df(n_hours=3)
        results, metrics = hourly_losocv(df)
        fallback = (results["variogram_used"] == "fallback_idw").sum()
        assert metrics.n_fallback == fallback

    def test_too_few_stations_raises(self) -> None:
        rows = [{
            "datetime": pd.Timestamp("2025-01-01 00:00:00"),
            "station_name": "DKI1 Bundaran HI",
            "latitude": -6.195459,
            "longitude": 106.822731,
            "pm25": 50.0,
        }]
        df = pd.DataFrame(rows)
        with pytest.raises(ValueError, match="at least 2"):
            hourly_losocv(df)


class TestHourlyLosoMetrics:
    def test_metrics_fields(self) -> None:
        df = _make_hourly_df(n_hours=3)
        _, metrics = hourly_losocv(df)
        assert hasattr(metrics, "mae")
        assert hasattr(metrics, "rmse")
        assert hasattr(metrics, "r_squared")
        assert hasattr(metrics, "bias")
        assert hasattr(metrics, "n_predictions")
        assert hasattr(metrics, "n_hours")
        assert hasattr(metrics, "n_fallback")
        assert hasattr(metrics, "per_sensor_mae")
