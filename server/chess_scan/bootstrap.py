"""Shared runtime and script bootstrap."""

from __future__ import annotations

from chess_scan.classifier import read_model_metadata
from chess_scan.config import Settings
from chess_scan.database import Database

BASE_MODEL_VERSION = "chess-steps-v2"
_REPLACED_BASE_MODEL_VERSIONS = {"argus-v2r5", "chess-steps-v1", "chess-steps-v1r1"}


def initialize_database(settings: Settings) -> Database:
    settings.ensure_directories()
    model_path = settings.model_dir / f"{BASE_MODEL_VERSION}.onnx"
    metadata_path = settings.model_dir / f"{BASE_MODEL_VERSION}.json"
    if not model_path.exists() or not metadata_path.exists():
        raise RuntimeError(
            f"Base model files are missing under {settings.model_dir}: "
            f"{model_path.name}, {metadata_path.name}"
        )

    database = Database(settings.data_dir / "chess-scan.sqlite3")
    database.initialize(
        base_model_version=BASE_MODEL_VERSION,
        base_model_path=model_path,
        base_model_metadata=read_model_metadata(metadata_path),
    )
    active = database.get_active_model()
    if active["version"] in _REPLACED_BASE_MODEL_VERSIONS:
        database.promote_model(BASE_MODEL_VERSION)
    return database
