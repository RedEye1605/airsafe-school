"""Unit tests for Kriging spatial interpolation."""

import pandas as pd
import numpy as np
import pytest

from src.spatial.kriging import KrigingConfig, kriging_interpolate


def _make_sensors(n: int = 5) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "sensor_id": [f"S{i}" for i in range(n)],
        "latitude": np.linspace(-6.15, -6.35, n),
        "longitude": np.linspace(106.75, 106.95, n),
        "pm25": np.random.uniform(30, 80, n),
    })


def _make_schools(n: int = 3) -> pd.DataFrame:
    lats = np.linspace(-6.19, -6.28, n)
    lons = np.linspace(106.81, 106.84, n)
    return pd.DataFrame({
        "npsn": [f"SCH{i:03d}" for i in range(n)],
        "nama_sekolah": [f"School {i}" for i in range(n)],
        "latitude": lats,
        "longitude": lons,
    })


class TestKrigingInterpolate:
    def test_returns_school_predictions(self) -> None:
        out = kriging_interpolate(_make_sensors(), _make_schools())
        assert len(out) == 3
        assert "pm25_kriging" in out.columns
        assert "kriging_variance" in out.columns
        assert "kriging_std" in out.columns
        assert "variogram_model" in out.columns
        assert out["pm25_kriging"].notna().all()
        assert (out["pm25_kriging"] >= 0).all()

    def test_fallback_idw_when_too_few_sensors(self) -> None:
        sensors = pd.DataFrame({
            "sensor_id": ["S1"],
            "latitude": [-6.17],
            "longitude": [106.78],
            "pm25": [42.0],
        })
        schools = _make_schools(1)
        out = kriging_interpolate(sensors, schools)
        assert len(out) == 1
        assert out.iloc[0]["variogram_model"] == "fallback_idw"
        assert out.iloc[0]["pm25_kriging"] >= 0

    def test_handles_zero_pm25(self) -> None:
        sensors = _make_sensors()
        sensors.loc[0, "pm25"] = 0.0
        out = kriging_interpolate(sensors, _make_schools())
        assert (out["pm25_kriging"] >= 0).all()

    def test_dedupes_same_coordinates(self) -> None:
        sensors = pd.DataFrame({
            "sensor_id": ["S1", "S2", "S3"],
            "latitude": [-6.17, -6.17, -6.25],
            "longitude": [106.78, 106.78, 106.90],
            "pm25": [40.0, 60.0, 70.0],
        })
        out = kriging_interpolate(sensors, _make_schools(1))
        assert out.iloc[0]["pm25_kriging"] > 0

    def test_empty_sensors_raises(self) -> None:
        sensors = pd.DataFrame({"sensor_id": [], "latitude": [], "longitude": [], "pm25": []})
        with pytest.raises(ValueError):
            kriging_interpolate(sensors, _make_schools())

    def test_custom_value_col(self) -> None:
        sensors = _make_sensors().rename(columns={"pm25": "pm25_pred_6h"})
        out = kriging_interpolate(sensors, _make_schools(), value_col="pm25_pred_6h")
        assert "pm25_pred_6h_kriging" in out.columns

    def test_real_data_columns(self) -> None:
        """Verify the default col names match our actual data schema."""
        sensors = pd.DataFrame({
            "station_name": ["A"],
            "latitude": [-6.17],
            "longitude": [106.78],
            "pm25": [42.0],
        })
        schools = pd.DataFrame({
            "npsn": ["123"],
            "latitude": [-6.19],
            "longitude": [106.81],
        })
        out = kriging_interpolate(sensors, schools)
        assert len(out) == 1
        assert out.iloc[0]["pm25_kriging"] > 0
