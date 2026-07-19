from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from chess_scan.classifier import ModelManager
from chess_scan.config import Settings
from chess_scan.database import Database
from chess_scan.errors import ScanExpiredError
from chess_scan.service import ScannerService


def test_scan_rejects_an_image_without_a_complete_aligned_grid(tmp_path: Path) -> None:
    service, _, settings = _service(tmp_path)
    blank = np.full((640, 640, 3), 230, dtype=np.uint8)
    encoded, payload = cv2.imencode(".png", blank)
    assert encoded

    with pytest.raises(ValueError, match="No complete, aligned 8x8 chess board"):
        service.scan(payload.tobytes())

    assert not list((settings.data_dir / "source-temp").iterdir())
    assert not list((settings.data_dir / "rectified").iterdir())


def test_cleanup_reconciles_confirmed_and_orphaned_files(tmp_path: Path) -> None:
    service, database, settings = _service(tmp_path)
    retained_source, retained_rectified = _create_scan_files(
        database,
        settings,
        scan_id="retained",
    )
    deleted_source, deleted_rectified = _create_scan_files(
        database,
        settings,
        scan_id="deleted",
    )
    database.confirm_scan(
        feedback_id="retained-feedback",
        scan_id="retained",
        labels=[0] * 64,
        orientation="white",
        side_to_move="w",
        castling="-",
        en_passant="-",
        full_fen="8/8/8/8/8/8/8/8 w - - 0 1",
        consent_training=True,
        client_session_id=None,
    )
    database.confirm_scan(
        feedback_id="deleted-feedback",
        scan_id="deleted",
        labels=[0] * 64,
        orientation="white",
        side_to_move="w",
        castling="-",
        en_passant="-",
        full_fen="8/8/8/8/8/8/8/8 w - - 0 1",
        consent_training=False,
        client_session_id=None,
    )
    orphan = settings.data_dir / "source-temp/orphan.jpg"
    orphan.write_bytes(b"orphan")
    old = time.time() - 2 * 24 * 60 * 60
    os.utime(orphan, (old, old))

    result = service.remove_stale_sources()

    assert result.backlog is False
    assert not retained_source.exists()
    assert retained_rectified.exists()
    assert not deleted_source.exists()
    assert not deleted_rectified.exists()
    assert not orphan.exists()


def test_cleanup_drains_more_than_one_batch_and_uses_state_indexes(tmp_path: Path) -> None:
    service, database, settings = _service(tmp_path)
    paths = [_create_scan_files(database, settings, scan_id=f"scan-{index}") for index in range(5)]

    result = service.remove_stale_sources(
        older_than_seconds=-1,
        batch_size=2,
    )

    assert result.removed == 5
    assert result.backlog is False
    assert all(not source.exists() and not rectified.exists() for source, rectified in paths)
    with pytest.raises(ScanExpiredError):
        database.confirm_scan(
            feedback_id="expired-feedback",
            scan_id="scan-0",
            labels=[0] * 64,
            orientation="white",
            side_to_move="w",
            castling="-",
            en_passant="-",
            full_fen="8/8/8/8/8/8/8/8 w - - 0 1",
            consent_training=False,
            client_session_id=None,
        )
    with sqlite3.connect(database.path) as connection:
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index'"
            ).fetchall()
        }
    assert "idx_scans_open_cleanup" in indexes
    assert "idx_scans_pending_cleanup" in indexes


def _service(tmp_path: Path) -> tuple[ScannerService, Database, Settings]:
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=tmp_path / "models",
        web_dist=tmp_path / "web",
        max_upload_bytes=1024,
        max_image_dimension=1200,
        cors_origins=(),
    )
    settings.ensure_directories()
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    database = Database(settings.data_dir / "chess-scan.sqlite3")
    database.initialize(
        base_model_version="base",
        base_model_path=model_path,
        base_model_metadata={},
    )
    return (
        ScannerService(
            settings=settings,
            database=database,
            models=ModelManager(database),
        ),
        database,
        settings,
    )


def _create_scan_files(
    database: Database,
    settings: Settings,
    *,
    scan_id: str,
) -> tuple[Path, Path]:
    source = settings.data_dir / "source-temp" / f"{scan_id}.jpg"
    rectified = settings.data_dir / "rectified" / f"{scan_id}.jpg"
    source.write_bytes(b"source")
    rectified.write_bytes(b"rectified")
    database.create_scan(
        scan_id=scan_id,
        image_sha256=f"hash-{scan_id}",
        source_width=100,
        source_height=100,
        source_image_path=source,
        rectified_image_path=rectified,
        corners=[[0, 0], [99, 0], [99, 99], [0, 99]],
        detection_method="test",
        model_version="base",
        labels=[0] * 64,
        probabilities=[[1.0] + [0.0] * 12 for _ in range(64)],
        board_fen="8/8/8/8/8/8/8/8",
    )
    return source, rectified
