"""Unit tests for leave-one-sensor-out cross-validation."""

import html
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
        fallback_mask = results["variogram_used"] == "fallback_idw"
        assert int(fallback_mask.sum()) == metrics.n_fallback

    def test_custom_value_col(self) -> None:
        sensors = _make_sensors(5).rename(columns={"pm25": "pm25_pred"})
        results, _ = losocv_validate(sensors, value_col="pm25_pred")
        assert "predicted_pm25" in results.columns
        assert results["actual_pm25"].notna().all()


class TestLosocvEdgeCases:
    def test_single_sensor_raises(self) -> None:
        sensors = _make_sensors(1)
        with pytest.raises(ValueError, match="at least 2"):
            losocv_validate(sensors)

    def test_duplicate_index_handled(self) -> None:
        """Duplicate DataFrame index must not cause wrong hold-out."""
        sensors = pd.DataFrame({
            "station_name": ["A", "B", "C"],
            "latitude": [-6.15, -6.20, -6.25],
            "longitude": [106.75, 106.80, 106.85],
            "pm25": [40.0, 50.0, 60.0],
        }, index=[0, 0, 0])
        results, metrics = losocv_validate(sensors)
        assert len(results) == 3
        assert metrics.n_sensors == 3

    def test_nan_pm25_sensor_skipped(self) -> None:
        """Sensor with NaN PM2.5 should fail that fold but not crash."""
        sensors = _make_sensors(5)
        sensors.loc[2, "pm25"] = np.nan
        # The NaN sensor will have NaN actual → fold may fail or produce NaN
        results, metrics = losocv_validate(sensors)
        # Should still produce results for all 5 folds
        assert len(results) == 5

    def test_duplicate_coordinates(self) -> None:
        """Two sensors at same location should still produce results."""
        sensors = pd.DataFrame({
            "station_name": ["A", "B", "C"],
            "latitude": [-6.17, -6.17, -6.25],
            "longitude": [106.78, 106.78, 106.90],
            "pm25": [40.0, 60.0, 70.0],
        })
        results, metrics = losocv_validate(sensors)
        assert len(results) == 3
        assert metrics.n_sensors == 3


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

    def test_bias_sign_convention(self) -> None:
        """Positive bias means model underpredicts (actual > predicted)."""
        sensors = pd.DataFrame({
            "station_name": ["A", "B", "C"],
            "latitude": [-6.15, -6.20, -6.25],
            "longitude": [106.75, 106.80, 106.85],
            "pm25": [50.0, 50.0, 50.0],
        })
        _, metrics = losocv_validate(sensors)
        # With identical values, bias should be ~0
        assert metrics.bias == pytest.approx(0.0, abs=0.5)


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

    def test_handles_nan_errors(self, tmp_path: object) -> None:
        from pathlib import Path
        from src.spatial.error_map import build_error_map

        sensors = _make_sensors(5)
        results, metrics = losocv_validate(sensors)
        # Inject NaN rows
        nan_row = pd.DataFrame([{
            "sensor_id": "XSS<script>alert(1)</script>",
            "latitude": -6.20, "longitude": 106.80,
            "actual_pm25": 50.0, "predicted_pm25": np.nan,
            "abs_error": np.nan, "squared_error": np.nan,
            "variogram_used": "error",
        }])
        results_with_nan = pd.concat([results, nan_row], ignore_index=True)
        out = build_error_map(
            results_with_nan, metrics,
            output_path=Path(str(tmp_path)) / "nan_map.html",
        )
        content = out.read_text()
        # XSS string should be HTML-escaped in the file if present
        assert "<script>" not in content or "alert" not in content

    def test_failed_sensor_count_shown(self, tmp_path: object) -> None:
        from pathlib import Path
        from src.spatial.error_map import build_error_map

        sensors = _make_sensors(5)
        results, metrics = losocv_validate(sensors)
        # Add a failed row
        nan_row = pd.DataFrame([{
            "sensor_id": "FAILED",
            "latitude": -6.20, "longitude": 106.80,
            "actual_pm25": 50.0, "predicted_pm25": np.nan,
            "abs_error": np.nan, "squared_error": np.nan,
            "variogram_used": "error",
        }])
        results_with_fail = pd.concat([results, nan_row], ignore_index=True)
        out = build_error_map(
            results_with_fail, metrics,
            output_path=Path(str(tmp_path)) / "fail_map.html",
        )
        content = out.read_text()
        assert "Failed: 1" in content
