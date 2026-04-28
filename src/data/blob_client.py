"""
Azure Blob Storage client for the AirSafe ETL pipeline.

Uploads raw snapshots, processed CSVs, and ETL manifests to Blob Storage.
Falls back to local filesystem when ``AIRSAFE_BLOB_CONNECTION_STRING`` is
not set, so local development and tests keep working unchanged.

Blob path convention (partitioned for queryability):
    raw/spku/date=2026-04-23/hour=14/spku_20260423T140000Z.json
    raw/bmkg/date=2026-04-23/hour=14/bmkg_20260423T140000Z.json
    processed/daily/spku_pm25_2026-04-23.csv
    logs/etl/date=2026-04-23/run_20260423T140000Z.json
"""

import json
import logging
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from src.config import (
    AIRSAFE_BLOB_CONNECTION_STRING,
    AIRSAFE_FEATURES_CONTAINER,
    AIRSAFE_LOG_CONTAINER,
    AIRSAFE_MODELS_CONTAINER,
    AIRSAFE_PREDICT_CONTAINER,
    AIRSAFE_PROCESSED_CONTAINER,
    AIRSAFE_RAW_CONTAINER,
    AIRSAFE_REFERENCE_CONTAINER,
    AIRSAFE_SCRATCH_CONTAINER,
    DATA_DIR,
)

logger = logging.getLogger(__name__)


def is_blob_configured() -> bool:
    """Return True if Blob Storage connection is configured."""
    return bool(AIRSAFE_BLOB_CONNECTION_STRING)


def _get_service():
    """Lazy-load BlobServiceClient (avoids import error when not needed)."""
    from azure.storage.blob import BlobServiceClient

    return BlobServiceClient.from_connection_string(AIRSAFE_BLOB_CONNECTION_STRING)


def _ensure_container(service, container: str) -> None:
    """Create container if it doesn't exist."""
    client = service.get_container_client(container)
    try:
        client.get_container_properties()
    except Exception:
        client.create_container()
        logger.info("Created container: %s", container)


def upload_text(
    container: str,
    blob_name: str,
    text: str,
    *,
    content_type: str = "text/plain; charset=utf-8",
) -> str:
    """Upload text content to Blob Storage.

    Args:
        container: Target container name.
        blob_name: Path inside the container.
        text: Content to upload.
        content_type: MIME type for the blob.

    Returns:
        The blob name that was written.
    """
    from azure.storage.blob import ContentSettings

    service = _get_service()
    _ensure_container(service, container)
    client = service.get_blob_client(container=container, blob=blob_name)
    client.upload_blob(
        text.encode("utf-8"),
        overwrite=True,
        content_settings=ContentSettings(content_type=content_type),
    )
    logger.info("Uploaded blob: %s/%s", container, blob_name)
    return blob_name


def upload_json(
    container: str,
    blob_name: str,
    payload: dict[str, Any],
) -> str:
    """Upload a dict as JSON to Blob Storage.

    Args:
        container: Target container name.
        blob_name: Path inside the container.
        payload: JSON-serialisable dict.

    Returns:
        The blob name that was written.
    """
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return upload_text(
        container, blob_name, text, content_type="application/json; charset=utf-8"
    )


def upload_dataframe(
    container: str,
    blob_name: str,
    df: pd.DataFrame,
) -> str:
    """Upload a DataFrame as CSV to Blob Storage.

    Args:
        container: Target container name.
        blob_name: Path inside the container.
        df: DataFrame to upload.

    Returns:
        The blob name that was written.
    """
    csv_text = df.to_csv(index=False)
    return upload_text(
        container, blob_name, csv_text, content_type="text/csv; charset=utf-8"
    )


# ── Convenience: dual-write (Blob + local) ────────────────────────────────


def save_json_dual(
    data: dict[str, Any],
    blob_path: str,
    local_path: Path,
    *,
    container: Optional[str] = None,
) -> str:
    """Save JSON to Blob if configured, always save locally.

    Args:
        data: JSON-serialisable dict.
        blob_path: Blob name (e.g. ``raw/spku/date=.../file.json``).
        local_path: Local filesystem path.
        container: Blob container (default from config).

    Returns:
        Human-readable path where data was saved.
    """
    container = container or AIRSAFE_RAW_CONTAINER
    local_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    local_path.write_text(text, encoding="utf-8")

    if is_blob_configured():
        try:
            upload_text(
                container, blob_path, text, content_type="application/json; charset=utf-8"
            )
            return f"blob://{container}/{blob_path}"
        except Exception:
            logger.exception("Blob upload failed for %s/%s", container, blob_path)

    return str(local_path)


def save_dataframe_dual(
    df: pd.DataFrame,
    blob_path: str,
    local_path: Path,
    *,
    container: Optional[str] = None,
) -> str:
    """Save DataFrame CSV to Blob if configured, always save locally.

    Args:
        df: DataFrame to save.
        blob_path: Blob name.
        local_path: Local filesystem path.
        container: Blob container (default from config).

    Returns:
        Human-readable path where data was saved.
    """
    container = container or AIRSAFE_PROCESSED_CONTAINER
    local_path.parent.mkdir(parents=True, exist_ok=True)
    csv_text = df.to_csv(index=False)
    local_path.write_text(csv_text, encoding="utf-8")

    if is_blob_configured():
        upload_text(
            container, blob_path, csv_text, content_type="text/csv; charset=utf-8"
        )
        return f"blob://{container}/{blob_path}"

    return str(local_path)
