"""Spatial interpolation sub-package."""

from src.spatial.hourly_kriging import (
    ISPU_STATION_COORDS,
    hourly_kriging_interpolate,
    hourly_kriging_to_file,
    load_hourly_data,
    run_hourly_pipeline,
)
from src.spatial.hourly_losocv import HourlyLosoMetrics, hourly_losocv
from src.spatial.kriging import KrigingConfig, kriging_interpolate
from src.spatial.losolocv import LosoMetrics, losocv_validate
from src.spatial.lag_kriging import (
    lag_kriging_interpolate,
    lag_kriging_to_file,
    load_lag_dataset,
    run_lag_pipeline,
)
from src.spatial.residual_corrector import ResidualCorrector


def build_error_map(*args, **kwargs):
    """Lazy import — requires folium."""
    from src.spatial.error_map import build_error_map as _build
    return _build(*args, **kwargs)


__all__ = [
    "HourlyLosoMetrics",
    "ISPU_STATION_COORDS",
    "KrigingConfig",
    "LosoMetrics",
    "ResidualCorrector",
    "build_error_map",
    "hourly_kriging_interpolate",
    "hourly_kriging_to_file",
    "hourly_losocv",
    "kriging_interpolate",
    "lag_kriging_interpolate",
    "lag_kriging_to_file",
    "load_hourly_data",
    "load_lag_dataset",
    "losocv_validate",
    "run_hourly_pipeline",
    "run_lag_pipeline",
]
