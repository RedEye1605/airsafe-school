"""Re-export SPKU client from src.data.spku_client."""

from src.data.spku_client import (  # noqa: F401
    SpkuApiError,
    extract_pm25_readings,
    fetch_all_stations,
)
