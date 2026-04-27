"""Unit tests for Kriging spatial interpolation."""

import numpy as np
import pandas as pd
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
        # IDW fallback should set variance/std to NaN (not comparable)
        assert pd.isna(out.iloc[0]["kriging_variance"])
        assert pd.isna(out.iloc[0]["kriging_std"])

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
        sensors = pd.DataFrame({
            "sensor_id": [], "latitude": [], "longitude": [], "pm25": [],
        })
        with pytest.raises(ValueError):
            kriging_interpolate(sensors, _make_schools())

    def test_custom_value_col(self) -> None:
        sensors = _make_sensors().rename(columns={"pm25": "pm25_pred_6h"})
        out = kriging_interpolate(sensors, _make_schools(), value_col="pm25_pred_6h")
        assert "pm25_pred_6h_kriging" in out.columns

    def test_real_data_columns(self) -> None:
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


class TestBestVariogramSelection:
    """The best-fit variogram should be selected, not just the first."""

    def test_selects_best_variogram(self) -> None:
        sensors = _make_sensors(10)
        schools = _make_schools(2)
        out = kriging_interpolate(sensors, schools)
        model = out.iloc[0]["variogram_model"]
        assert model in ("spherical", "exponential", "gaussian", "linear")
        # With 10 sensors the model should be a real variogram, not IDW
        assert model != "fallback_idw"

    def test_logs_variogram_selection(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging
        with caplog.at_level(logging.INFO, logger="src.spatial.kriging"):
            kriging_interpolate(_make_sensors(5), _make_schools(1))
        # Should log variogram fit cR for each candidate
        assert any("Variogram" in r.message and "cR=" in r.message for r in caplog.records)
        # Should log the selected model
        assert any("Selected variogram" in r.message for r in caplog.records)


class TestIdwFallback:
    """IDW fallback behaviours."""

    def test_idw_with_two_sensors(self) -> None:
        sensors = pd.DataFrame({
            "sensor_id": ["S1", "S2"],
            "latitude": [-6.17, -6.25],
            "longitude": [106.78, 106.90],
            "pm25": [40.0, 70.0],
        })
        out = kriging_interpolate(sensors, _make_schools(1))
        assert out.iloc[0]["variogram_model"] == "fallback_idw"
        assert out.iloc[0]["pm25_kriging"] > 0
        assert pd.isna(out.iloc[0]["kriging_variance"])

    def test_idw_target_at_sensor_location(self) -> None:
        sensors = pd.DataFrame({
            "sensor_id": ["S1"],
            "latitude": [-6.17],
            "longitude": [106.78],
            "pm25": [42.0],
        })
        schools = pd.DataFrame({
            "npsn": ["SCH001"],
            "latitude": [-6.17],
            "longitude": [106.78],
        })
        out = kriging_interpolate(sensors, schools)
        assert out.iloc[0]["pm25_kriging"] == pytest.approx(42.0)


class TestDataCleaning:
    """Edge cases in coordinate/value cleaning."""

    def test_nan_targets_dropped(self) -> None:
        sensors = _make_sensors(5)
        schools = pd.DataFrame({
            "npsn": ["SCH001", "SCH002", "SCH003"],
            "latitude": [-6.19, np.nan, -6.25],
            "longitude": [106.81, 106.82, np.nan],
        })
        out = kriging_interpolate(sensors, schools)
        # SCH002 has NaN lat, SCH003 has NaN lon — both dropped
        # Only SCH001 (valid lat AND lon) survives
        assert len(out) == 1
        assert out.iloc[0]["npsn"] == "SCH001"

    def test_negative_longitude_converted(self) -> None:
        sensors = pd.DataFrame({
            "sensor_id": [f"S{i}" for i in range(3)],
            "latitude": [-6.17, -6.20, -6.25],
            "longitude": [-53.22, -53.25, -53.30],  # negative lon
            "pm25": [40.0, 50.0, 60.0],
        })
        schools = pd.DataFrame({
            "npsn": ["SCH001"],
            "latitude": [-6.22],
            "longitude": [-53.27],
        })
        out = kriging_interpolate(sensors, schools)
        assert len(out) == 1
        assert out.iloc[0]["pm25_kriging"] > 0

    def test_negative_pm25_filtered(self) -> None:
        sensors = _make_sensors(5)
        sensors.loc[0, "pm25"] = -5.0
        out = kriging_interpolate(sensors, _make_schools(1))
        assert out.iloc[0]["pm25_kriging"] > 0

    def test_output_never_exceeds_input_rows(self) -> None:
        sensors = _make_sensors(5)
        schools = _make_schools(10)
        out = kriging_interpolate(sensors, schools)
        assert len(out) <= len(schools)

    def test_duplicate_index_in_target(self) -> None:
        """Non-unique index must not cause row multiplication."""
        sensors = _make_sensors(5)
        schools = pd.DataFrame({
            "npsn": ["SCH001", "SCH002", "SCH003"],
            "latitude": [-6.19, -6.22, -6.25],
            "longitude": [106.81, 106.82, 106.83],
        }, index=[0, 0, 0])
        out = kriging_interpolate(sensors, schools)
        assert len(out) == 3


class TestEmptyTargets:
    def test_empty_targets_raises(self) -> None:
        sensors = _make_sensors(5)
        schools = pd.DataFrame({
            "npsn": [], "latitude": [], "longitude": [],
        })
        with pytest.raises(ValueError, match="No valid target"):
            kriging_interpolate(sensors, schools)


class TestMissingColumns:
    def test_sensor_missing_value_col(self) -> None:
        sensors = pd.DataFrame({
            "sensor_id": ["S1"],
            "latitude": [-6.17],
            "longitude": [106.78],
        })
        with pytest.raises(ValueError, match="sensor_df.*missing"):
            kriging_interpolate(sensors, _make_schools(1))

    def test_target_missing_id_col(self) -> None:
        sensors = _make_sensors(5)
        schools = pd.DataFrame({
            "latitude": [-6.19],
            "longitude": [106.81],
        })
        with pytest.raises(ValueError, match="target_df.*missing"):
            kriging_interpolate(sensors, schools)


class TestKrigingConfigValidation:
    def test_empty_variogram_models_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            KrigingConfig(variogram_models=())

    def test_invalid_variogram_model_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid variogram"):
            KrigingConfig(variogram_models=("cubic",))

    def test_invalid_coordinates_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="coordinates_type"):
            KrigingConfig(coordinates_type="mars")

    def test_nlags_too_small(self) -> None:
        with pytest.raises(ValueError, match="nlags"):
            KrigingConfig(nlags=1)

    def test_min_sensors_zero(self) -> None:
        with pytest.raises(ValueError, match="min_sensors"):
            KrigingConfig(min_sensors=0)

    def test_idw_power_negative(self) -> None:
        with pytest.raises(ValueError, match="idw_power"):
            KrigingConfig(idw_power=-1.0)

    def test_max_output_negative(self) -> None:
        with pytest.raises(ValueError, match="max_output_pm25"):
            KrigingConfig(max_output_pm25=-1.0)

    def test_valid_config_passes(self) -> None:
        cfg = KrigingConfig(nlags=10, min_sensors=3, idw_power=2.5)
        assert cfg.nlags == 10
