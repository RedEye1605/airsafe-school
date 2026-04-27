"""
AirSafe custom exceptions.
"""


class AirSafeError(Exception):
    """Base exception for AirSafe errors."""


class DataAcquisitionError(AirSafeError):
    """Raised when data download or API fetch fails."""


class SpkuApiError(DataAcquisitionError):
    """Raised when SPKU API request fails."""


class WeatherApiError(DataAcquisitionError):
    """Raised when weather API request fails."""


class BmkgApiError(DataAcquisitionError):
    """Raised when BMKG API request fails."""


class ConfigError(AirSafeError):
    """Raised on missing or invalid configuration."""
