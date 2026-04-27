"""
Residual correction model for Kriging interpolation.

Trains a LightGBM regressor on LOSOCV residuals to learn systematic
spatial patterns missed by the variogram. The corrected prediction is:

    pm25_corrected = pm25_kriging + predicted_residual

This is a two-stage interpolation approach (Kriging + ML residual
correction) commonly used in geostatistics when the variogram alone
cannot capture all spatial structure.

LightGBM is used with conservative hyperparameters for small-sample
regimes (~98 training points): shallow trees (max_depth=3),
high min_data_in_leaf, strong L1/L2 regularization, and bagging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from src.spatial.kriging import _haversine_dist

logger = logging.getLogger(__name__)

_VARIOGRAM_ONEHOT = ["spherical", "exponential", "gaussian", "linear"]

_FEATURE_COLUMNS = [
    "kriging_prediction",
    "kriging_std",
    "n_sensors",
    "dist_nearest",
    "dist_2nd_nearest",
    "sensor_density_5km",
    "sensor_density_10km",
    "latitude",
    "longitude",
    "lat_x_lon",
    "variogram_spherical",
    "variogram_exponential",
    "variogram_gaussian",
    "variogram_linear",
]


def compute_features(
    points_df: pd.DataFrame,
    kriging_prediction: np.ndarray,
    kriging_std: np.ndarray,
    n_sensors: np.ndarray,
    variogram_model: np.ndarray,
    sensor_df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    feature_columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Compute feature matrix for residual correction.

    Args:
        points_df: DataFrame with target point coordinates.
        kriging_prediction: Kriging PM2.5 predictions.
        kriging_std: Kriging standard deviation (sqrt variance).
        n_sensors: Number of sensors used per prediction.
        variogram_model: Variogram model name per prediction.
        sensor_df: Sensor DataFrame (for distance/density features).
        lat_col: Latitude column name.
        lon_col: Longitude column name.

    Returns:
        DataFrame with engineered features.
    """
    n = len(points_df)
    feats = pd.DataFrame(index=range(n))

    feats["kriging_prediction"] = kriging_prediction
    # NaN kriging_std occurs when IDW fallback is used (no variance estimate).
    # Encode as -1.0 so LightGBM can distinguish fallback from real Kriging
    # uncertainty without a separate missing-flag column.
    feats["kriging_std"] = np.where(np.isnan(kriging_std), -1.0, kriging_std)
    feats["n_sensors"] = n_sensors

    # One-hot encode variogram model
    for vm in _VARIOGRAM_ONEHOT:
        feats[f"variogram_{vm}"] = np.array([
            1.0 if str(v) == vm else 0.0 for v in variogram_model
        ])

    # Point coordinates
    lats = points_df[lat_col].to_numpy(dtype=float)
    lons = points_df[lon_col].to_numpy(dtype=float)
    feats["latitude"] = lats
    feats["longitude"] = lons
    feats["lat_x_lon"] = lats * lons

    # Sensor coordinates for distance/density
    s_lat = sensor_df[lat_col].to_numpy(dtype=float)
    s_lon = sensor_df[lon_col].to_numpy(dtype=float)
    s_lon_norm = np.where(s_lon > 180, s_lon - 360, s_lon)

    dist_nearest = np.empty(n)
    dist_2nd = np.empty(n)
    density_5 = np.empty(n, dtype=int)
    density_10 = np.empty(n, dtype=int)

    for i in range(n):
        t_lon = lons[i]
        t_lon_norm = t_lon if t_lon <= 180 else t_lon - 360
        dists = _haversine_dist(
            s_lon_norm, s_lat,
            np.full_like(s_lon_norm, t_lon_norm),
            np.full_like(s_lat, lats[i]),
        )
        sorted_d = np.sort(dists)
        dist_nearest[i] = sorted_d[0]
        dist_2nd[i] = sorted_d[1] if len(sorted_d) > 1 else sorted_d[0]
        density_5[i] = int(np.sum(dists <= 5.0))
        density_10[i] = int(np.sum(dists <= 10.0))

    feats["dist_nearest"] = dist_nearest
    feats["dist_2nd_nearest"] = dist_2nd
    feats["sensor_density_5km"] = density_5
    feats["sensor_density_10km"] = density_10

    return feats[feature_columns or _FEATURE_COLUMNS]


class ResidualCorrector:
    """Corrects Kriging predictions using a trained LightGBM residual model.

    Uses gradient boosting with conservative hyperparameters to predict
    Kriging residuals from spatial and interpolation features.

    The correction is additive:
        pm25_corrected = pm25_kriging + model.predict(features)
    """

    # Conservative hyperparameters for ~98 training samples
    _LGBM_PARAMS = {
        "objective": "regression_l2",
        "metric": "mae",
        "n_estimators": 100,
        "max_depth": 2,
        "learning_rate": 0.05,
        "num_leaves": 4,
        "min_child_samples": 10,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "random_state": 42,
        "verbose": -1,
    }

    def __init__(
        self,
        random_state: int = 42,
        lgbm_params: Optional[dict] = None,
        feature_columns: Optional[list[str]] = None,
    ) -> None:
        self.scaler = StandardScaler()
        self.feature_columns = feature_columns or list(_FEATURE_COLUMNS)
        params = {**(lgbm_params or self._LGBM_PARAMS), "random_state": random_state}
        self.model = lgb.LGBMRegressor(**params)
        self.is_trained = False
        self.train_metrics: dict = {}

    def train(
        self,
        loso_results: pd.DataFrame,
        sensor_df: pd.DataFrame,
        lat_col: str = "latitude",
        lon_col: str = "longitude",
    ) -> dict:
        """Train the residual correction model on LOSOCV results.

        Args:
            loso_results: Per-sensor LOSOCV DataFrame from losocv_validate().
                Must include kriging_variance, kriging_std, n_sensors_fold
                columns (stored by losocv_validate since v2).
            sensor_df: Full sensor DataFrame for distance/density features.
            lat_col: Latitude column name.
            lon_col: Longitude column name.

        Returns:
            Dict with training metrics. Note: mae_after, rmse_after, r2_after
            are IN-SAMPLE (training set) metrics. cv_mae_mean is the
            out-of-sample cross-validation metric.
        """
        valid = loso_results.dropna(subset=["predicted_pm25", "actual_pm25"]).copy()
        if len(valid) < 5:
            raise ValueError(
                f"Need at least 5 valid LOSOCV results, got {len(valid)}"
            )

        residuals = (valid["actual_pm25"] - valid["predicted_pm25"]).to_numpy(
            dtype=float,
        )

        # Build features per LOSOCV row, using the HELD-OUT sensor subset
        # for distance/density (each row's held-out sensor excluded)
        all_features = []
        for _, row in valid.iterrows():
            held_id = row["sensor_id"]
            held_sensors = sensor_df[
                sensor_df.iloc[:, 0].astype(str) != held_id
            ] if len(sensor_df.columns) > 0 else sensor_df

            target_df = pd.DataFrame({
                lat_col: [row[lat_col]],
                lon_col: [row[lon_col]],
            })
            kriging_std_val = row.get("kriging_std", np.nan)
            n_sensors_val = row.get("n_sensors_fold", len(held_sensors))
            variogram_val = row["variogram_used"]

            feats = compute_features(
                target_df,
                np.array([row["predicted_pm25"]]),
                np.array([kriging_std_val]),
                np.array([n_sensors_val]),
                np.array([variogram_val]),
                held_sensors,
                lat_col=lat_col, lon_col=lon_col,
                feature_columns=self.feature_columns,
            )
            all_features.append(feats.iloc[0])

        X = pd.DataFrame(all_features).reset_index(drop=True)
        y = residuals

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Cross-validation (out-of-sample) — clone model for fair eval
        from sklearn.base import clone
        cv_model = clone(self.model)
        cv_scores = cross_val_score(
            cv_model, X_scaled, y, cv=min(5, len(valid) // 5),
            scoring="neg_mean_absolute_error",
        )
        cv_mae = -cv_scores

        # Fit on all data
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # In-sample metrics (for comparison only)
        corrected = valid["predicted_pm25"].to_numpy() + self.model.predict(X_scaled)
        corrected = np.clip(corrected, 0, 1000.0)

        actual = valid["actual_pm25"].to_numpy()
        residuals_after = actual - corrected

        ss_tot = float(np.sum((actual - np.mean(actual)) ** 2))

        # Feature importances from LightGBM
        importances = dict(zip(
            self.feature_columns,
            self.model.feature_importances_.tolist(),
        ))

        self.train_metrics = {
            "n_samples": len(valid),
            "n_estimators": self.model.n_estimators,
            "max_depth": self.model.max_depth,
            "learning_rate": self.model.learning_rate,
            "cv_mae_mean": round(float(cv_mae.mean()), 3),
            "cv_mae_std": round(float(cv_mae.std()), 3),
            "in_sample_mae_before": round(float(np.mean(np.abs(residuals))), 3),
            "in_sample_mae_after": round(float(np.mean(np.abs(residuals_after))), 3),
            "in_sample_rmse_before": round(float(np.sqrt(np.mean(residuals ** 2))), 3),
            "in_sample_rmse_after": round(
                float(np.sqrt(np.mean(residuals_after ** 2))), 3,
            ),
            "in_sample_r2_before": round(
                float(1 - np.sum(residuals ** 2) / ss_tot) if ss_tot > 0 else 0.0, 4,
            ),
            "in_sample_r2_after": round(
                float(1 - np.sum(residuals_after ** 2) / ss_tot) if ss_tot > 0 else 0.0, 4,
            ),
            "feature_importances": importances,
        }

        top3 = sorted(importances.items(), key=lambda x: -x[1])[:3]
        logger.info(
            "Residual corrector trained (LightGBM n=%d, depth=%d, lr=%.3f): "
            "CV MAE=%.3f ± %.3f, "
            "in-sample MAE %.3f → %.3f, R² %.4f → %.4f",
            self.model.n_estimators, self.model.max_depth,
            self.model.learning_rate,
            cv_mae.mean(), cv_mae.std(),
            self.train_metrics["in_sample_mae_before"],
            self.train_metrics["in_sample_mae_after"],
            self.train_metrics["in_sample_r2_before"],
            self.train_metrics["in_sample_r2_after"],
        )
        logger.info("Top features: %s", top3)

        return self.train_metrics

    def correct(
        self,
        kriging_result: pd.DataFrame,
        sensor_df: pd.DataFrame,
        value_col: str = "pm25",
        lat_col: str = "latitude",
        lon_col: str = "longitude",
    ) -> pd.DataFrame:
        """Apply residual correction to Kriging predictions.

        Args:
            kriging_result: Output DataFrame from kriging_interpolate().
            sensor_df: Sensor DataFrame for distance/density features.
            value_col: Value column prefix.
            lat_col: Latitude column name.
            lon_col: Longitude column name.

        Returns:
            kriging_result with added columns:
                pm25_corrected, residual_pred.
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() first.")

        result = kriging_result.copy()

        kriging_pred = result[f"{value_col}_kriging"].to_numpy(dtype=float)
        kriging_std = result["kriging_std"].to_numpy(dtype=float)
        n_sensors = result["n_sensors"].to_numpy(dtype=float)
        variogram = result["variogram_model"].to_numpy()

        X = compute_features(
            result, kriging_pred, kriging_std, n_sensors, variogram,
            sensor_df, lat_col=lat_col, lon_col=lon_col,
            feature_columns=self.feature_columns,
        )

        X_scaled = self.scaler.transform(X)
        residual_pred = self.model.predict(X_scaled)
        corrected = np.clip(kriging_pred + residual_pred, 0, 1000.0)

        result["residual_pred"] = residual_pred
        result["pm25_corrected"] = corrected

        return result

    def save(self, path: str) -> Path:
        """Save trained model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Model saved: %s", path)
        return path

    @classmethod
    def load(cls, path: str) -> ResidualCorrector:
        """Load a trained model from disk."""
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(obj).__name__}")
        logger.info("Model loaded: %s", path)
        return obj
