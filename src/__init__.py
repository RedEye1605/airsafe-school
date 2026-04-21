"""
AirSafe School — Air Quality Monitoring System for Schools.

Integrates SPKU sensor data, weather data, and school registries to
monitor and predict air quality risks at school locations in
DKI Jakarta.

Modules:
    data: API clients, school registry, data transforms.
    features: Feature engineering for ML models.
    visualization: Coverage maps, charts, dashboards.
    utils: Configuration, shared helpers.

Example:
    >>> from src.data.spku_client import fetch_all_stations
    >>> stations = fetch_all_stations()
"""

__version__ = "0.2.0"
__author__ = "AirSafe Team"
