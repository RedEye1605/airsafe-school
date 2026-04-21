"""Tests for utility helpers."""

import json
import tempfile
from pathlib import Path

from src.utils.helpers import ensure_dir, load_json, save_json


def test_ensure_dir_creates(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert target.is_dir()
    assert result == target


def test_ensure_dir_idempotent(tmp_path: Path) -> None:
    ensure_dir(tmp_path / "x")
    ensure_dir(tmp_path / "x")  # should not raise


def test_save_and_load_json(tmp_path: Path) -> None:
    data = {"key": "value", "count": 42}
    path = tmp_path / "sub" / "test.json"
    save_json(data, path)
    assert path.exists()
    assert load_json(path) == data
