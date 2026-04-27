"""Unit tests for leave-one-sensor-out cross-validation."""

import numpy as np
import pandas as pd
import pytest

from src.spatial.losolocv import LosoMetrics, losocv_validate


def _make_sensors(n: int = 5) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "station_name": [f"ST{i}" for i in range(n)],
        "latitude": np.linspace(-6.15, -6.35, n),
        "longitude": np.linspace(106.75, 106.95, n),
        "pm25": np.random.uniform(30, 80, n),
    })


class TestLosocvValidate:
    def test_returns_all_sensors(self) -> None:
        sensors = _make_sensors(5)
        results, metrics = losocv_validate(sensors)
        assert len(results) == 5
        assert metrics.n_sensors == 5

    def test_per_sensor_columns(self) -> None:
        sensors = _make_sensors(5)
        results, _ = losocv_validate(sensors)
        expected = [
            "sensor_id", "latitude", "longitude", "actual_pm25",
            "predicted_pm25", "abs_error", "squared_error", "variogram_used",
        ]
        for col in expected:
            assert col in results.columns, f"Missing column: {col}"

    def test_preserves_sensor_identity(self) -> None:
        sensors = _make_sensors(5)
        results, _ = losocv_validate(sensors)
        for i in range(5):
            assert results.iloc[i]["sensor_id"] == f"ST{i}"

    def test_metrics_non_negative(self) -> None:
        sensors = _make_sensors(5)
        _, metrics = losocv_validate(sensors)
        assert metrics.mae >= 0
        assert metrics.rmse >= 0
        assert metrics.median_ae >= 0

    def test_rmse_gte_mae(self) -> None:
        sensors = _make_sensors(5)
        _, metrics = losocv_validate(sensors)
        assert metrics.rmse >= metrics.mae

    def test_with_three_sensors(self) -> None:
        sensors = _make_sensors(3)
        results, metrics = losocv_validate(sensors)
        assert len(results) == 3
        assert metrics.n_sensors == 3
        assert not results["predicted_pm25"].isna().all()

    def test_fallback_count(self) -> None:
        """With min_sensors=3, holding out from 3 leaves 2 → IDW fallback."""
        sensors = _make_sensors(3)
        results, metrics = losocv_validate(sensors)
        assert metrics.n_fallback >= 0
        fallback_mask = results["variogram_used"] == "fallback_idw"
        assert int(fallback_mask.sum()) == metrics.n_fallback

    def test_custom_value_col(self) -> None:
        sensors = _make_sensors(5).rename(columns={"pm25": "pm25_pred"})
        results, _ = losocv_validate(sensors, value_col="pm25_pred")
        assert "predicted_pm25" in results.columns
        assert results["actual_pm25"].notna().all()


class TestLosoMetrics:
    def test_perfect_predictions(self) -> None:
        sensors = pd.DataFrame({
            "station_name": ["A", "B", "C", "D"],
            "latitude": [-6.15, -6.20, -6.25, -6.30],
            "longitude": [106.75, 106.80, 106.85, 106.90],
            "pm25": [50.0, 50.0, 50.0, 50.0],
        })
        _, metrics = losocv_validate(sensors)
        # All sensors have same value → IDW fallback predicts exactly 50
        assert metrics.mae == pytest.approx(0.0, abs=0.01)


class TestBuildErrorMap:
    def test_creates_html_file(self, tmp_path: object) -> None:
        from pathlib import Path
        from src.spatial.error_map import build_error_map

        sensors = _make_sensors(5)
        results, metrics = losocv_validate(sensors)
        out = build_error_map(results, metrics, output_path=Path(str(tmp_path)) / "test_map.html")
        assert out.exists()
        content = out.read_text()
        assert "LOSOCV" in content
        assert "AirSafe" in content
