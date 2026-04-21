"""Unit tests for SPKU client helpers."""

from src.data.spku_client import _parse_float


class TestParseFloat:
    """Tests for _parse_float()."""

    def test_numeric_string(self) -> None:
        assert _parse_float("42.5") == 42.5

    def test_int(self) -> None:
        assert _parse_float(10) == 10.0

    def test_none(self) -> None:
        assert _parse_float(None) is None

    def test_empty(self) -> None:
        assert _parse_float("") is None

    def test_dash(self) -> None:
        assert _parse_float("-") is None

    def test_invalid(self) -> None:
        assert _parse_float("abc") is None
