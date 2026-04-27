"""Unit tests for residual correction model."""

import numpy as np
import pandas as pd
import pytest

from src.spatial.residual_corrector import (
    ResidualCorrector,
    compute_features,
    _FEATURE_COLUMNS,
)


def _make_sensors(n: int = 20) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "station_name": [f"ST{i}" for i in range(n)],
        "latitude": np.linspace(-6.15, -6.35, n),
        "longitude": np.linspace(106.75, 106.95, n),
        "pm25": np.random.uniform(20, 100, n),
    })


def _make_loso_results(sensors: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic LOSOCV results with kriging metadata."""
    np.random.seed(42)
    n = len(sensors)
    actual = sensors["pm25"].to_numpy()
    predicted = actual + np.random.normal(0, 10, n)
    return pd.DataFrame({
        "sensor_id": sensors["station_name"],
        "latitude": sensors["latitude"],
        "longitude": sensors["longitude"],
        "actual_pm25": actual,
        "predicted_pm25": predicted,
        "abs_error": np.abs(actual - predicted),
        "squared_error": (actual - predicted) ** 2,
        "variogram_used": ["exponential"] * n,
        "kriging_variance": np.random.uniform(500, 900, n),
        "kriging_std": np.random.uniform(20, 30, n),
        "n_sensors_fold": np.full(n, n - 1),
    })


class TestComputeFeatures:
    def test_returns_correct_columns(self) -> None:
        sensors = _make_sensors(10)
        points = sensors[["latitude", "longitude"]].head(3)
        kriging_pred = np.array([50.0, 60.0, 70.0])
        kriging_std = np.array([10.0, 12.0, 8.0])
        n_sensors = np.array([9, 9, 9])
        variogram = np.array(["exponential", "spherical", "gaussian"])

        feats = compute_features(
            points, kriging_pred, kriging_std, n_sensors, variogram,
            sensors,
        )
        assert list(feats.columns) == _FEATURE_COLUMNS
        assert len(feats) == 3

    def test_features_no_nan(self) -> None:
        sensors = _make_sensors(10)
        points = sensors[["latitude", "longitude"]].head(3)
        feats = compute_features(
            points,
            np.array([50.0, 60.0, 70.0]),
            np.array([10.0, 12.0, 8.0]),
            np.array([9, 9, 9]),
            np.array(["exponential"] * 3),
            sensors,
        )
        assert not feats.isna().any().any()

    def test_distance_features_reasonable(self) -> None:
        sensors = _make_sensors(10)
        points = sensors[["latitude", "longitude"]].head(1)
        feats = compute_features(
            points,
            np.array([50.0]),
            np.array([10.0]),
            np.array([9]),
            np.array(["exponential"]),
            sensors,
        )
        assert feats.iloc[0]["dist_nearest"] >= 0
        assert feats.iloc[0]["dist_2nd_nearest"] >= feats.iloc[0]["dist_nearest"]
        assert feats.iloc[0]["sensor_density_5km"] >= 0

    def test_nan_kriging_std_becomes_sentinel(self) -> None:
        sensors = _make_sensors(10)
        points = sensors[["latitude", "longitude"]].head(1)
        feats = compute_features(
            points,
            np.array([50.0]),
            np.array([np.nan]),
            np.array([9]),
            np.array(["exponential"]),
            sensors,
        )
        assert feats.iloc[0]["kriging_std"] == -1.0

    def test_onehot_variogram_encoding(self) -> None:
        sensors = _make_sensors(10)
        points = sensors[["latitude", "longitude"]].head(1)
        feats = compute_features(
            points,
            np.array([50.0]),
            np.array([10.0]),
            np.array([9]),
            np.array(["gaussian"]),
            sensors,
        )
        assert feats.iloc[0]["variogram_gaussian"] == 1.0
        assert feats.iloc[0]["variogram_spherical"] == 0.0
        assert feats.iloc[0]["variogram_exponential"] == 0.0


class TestResidualCorrectorTrain:
    def test_train_returns_metrics(self) -> None:
        sensors = _make_sensors(20)
        loso = _make_loso_results(sensors)
        corrector = ResidualCorrector()
        metrics = corrector.train(loso, sensors)
        assert "cv_mae_mean" in metrics
        assert "in_sample_mae_before" in metrics
        assert "in_sample_mae_after" in metrics
        assert "n_estimators" in metrics
        assert "feature_importances" in metrics
        assert corrector.is_trained

    def test_corrected_mae_improves_in_sample(self) -> None:
        sensors = _make_sensors(20)
        loso = _make_loso_results(sensors)
        corrector = ResidualCorrector()
        metrics = corrector.train(loso, sensors)
        assert metrics["in_sample_mae_after"] <= metrics["in_sample_mae_before"]

    def test_feature_importances_match_features(self) -> None:
        sensors = _make_sensors(20)
        loso = _make_loso_results(sensors)
        corrector = ResidualCorrector()
        metrics = corrector.train(loso, sensors)
        assert set(metrics["feature_importances"].keys()) == set(_FEATURE_COLUMNS)

    def test_too_few_samples_raises(self) -> None:
        sensors = _make_sensors(3)
        loso = _make_loso_results(sensors)
        corrector = ResidualCorrector()
        with pytest.raises(ValueError, match="at least 5"):
            corrector.train(loso, sensors)


class TestResidualCorrectorCorrect:
    def test_correct_adds_columns(self) -> None:
        sensors = _make_sensors(20)
        loso = _make_loso_results(sensors)
        corrector = ResidualCorrector()
        corrector.train(loso, sensors)

        kriging_result = pd.DataFrame({
            "npsn": ["SCH001", "SCH002"],
            "latitude": [-6.20, -6.25],
            "longitude": [106.80, 106.85],
            "pm25_kriging": [45.0, 55.0],
            "kriging_variance": [100.0, 120.0],
            "kriging_std": [10.0, 11.0],
            "variogram_model": ["exponential", "spherical"],
            "n_sensors": [20, 20],
        })
        result = corrector.correct(kriging_result, sensors)
        assert "pm25_corrected" in result.columns
        assert "residual_pred" in result.columns
        assert len(result) == 2

    def test_corrected_values_non_negative(self) -> None:
        sensors = _make_sensors(20)
        loso = _make_loso_results(sensors)
        corrector = ResidualCorrector()
        corrector.train(loso, sensors)

        kriging_result = pd.DataFrame({
            "npsn": ["SCH001"],
            "latitude": [-6.20],
            "longitude": [106.80],
            "pm25_kriging": [50.0],
            "kriging_variance": [100.0],
            "kriging_std": [10.0],
            "variogram_model": ["exponential"],
            "n_sensors": [20],
        })
        result = corrector.correct(kriging_result, sensors)
        assert result.iloc[0]["pm25_corrected"] >= 0

    def test_correct_before_train_raises(self) -> None:
        corrector = ResidualCorrector()
        sensors = _make_sensors(5)
        kriging_result = pd.DataFrame({
            "npsn": ["X"], "latitude": [-6.2], "longitude": [106.8],
            "pm25_kriging": [50.0], "kriging_variance": [100.0],
            "kriging_std": [10.0], "variogram_model": ["exponential"],
            "n_sensors": [5],
        })
        with pytest.raises(RuntimeError, match="not trained"):
            corrector.correct(kriging_result, sensors)


class TestSaveLoad:
    def test_roundtrip(self, tmp_path: object) -> None:
        from pathlib import Path
        sensors = _make_sensors(20)
        loso = _make_loso_results(sensors)

        corrector = ResidualCorrector()
        corrector.train(loso, sensors)
        metrics_before = corrector.train_metrics.copy()

        model_path = corrector.save(str(Path(str(tmp_path)) / "test_model.pkl"))
        assert model_path.exists()

        loaded = ResidualCorrector.load(str(model_path))
        assert loaded.is_trained
        assert loaded.train_metrics["in_sample_mae_after"] == pytest.approx(
            metrics_before["in_sample_mae_after"], abs=0.001,
        )

    def test_load_wrong_type_raises(self, tmp_path: object) -> None:
        from pathlib import Path
        import joblib
        bad_path = Path(str(tmp_path)) / "bad.pkl"
        joblib.dump({"not": "a model"}, bad_path)
        with pytest.raises(TypeError, match="Expected ResidualCorrector"):
            ResidualCorrector.load(str(bad_path))
