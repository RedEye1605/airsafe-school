"""Unit tests for SPKU client helpers."""

from datetime import datetime
from typing import Optional

from src.data.spku_client import _parse_timestamp, _safe_float


class TestParseTimestamp:
    """Tests for _parse_timestamp()."""

    def test_valid(self) -> None:
        result = _parse_timestamp("04/20/2026 12:00:00")
        assert result == datetime(2026, 4, 20, 12, 0, 0)

    def test_empty(self) -> None:
        assert _parse_timestamp("") is None

    def test_invalid(self) -> None:
        assert _parse_timestamp("not-a-date") is None


class TestSafeFloat:
    """Tests for _safe_float()."""

    def test_numeric(self) -> None:
        assert _safe_float("42.5") == 42.5
        assert _safe_float(10) == 10.0

    def test_none(self) -> None:
        assert _safe_float(None) is None

    def test_empty(self) -> None:
        assert _safe_float("") is None

    def test_dash(self) -> None:
        assert _safe_float("-") is None

    def test_invalid(self) -> None:
        assert _safe_float("abc") is None
