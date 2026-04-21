"""
AirSafe custom exceptions.
"""


class AirSafeError(Exception):
    """Base exception for AirSafe errors."""


class DataAcquisitionError(AirSafeError):
    """Raised when data download or API fetch fails."""


class ConfigError(AirSafeError):
    """Raised on missing or invalid configuration."""
