"""SQLite persistence for scans, immutable human feedback, and model versions."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def initialize(
        self,
        *,
        base_model_version: str,
        base_model_path: Path,
        base_model_metadata: dict[str, Any],
    ) -> None:
        with closing(self._connect()) as connection:
            connection.executescript(_SCHEMA)
            existing = connection.execute(
                "SELECT version FROM model_versions WHERE version = ?",
                (base_model_version,),
            ).fetchone()
            if existing is None:
                now = _now()
                connection.execute(
                    """
                    INSERT INTO model_versions (
                        version, artifact_path, metadata_json, created_at, activated_at, is_active
                    ) VALUES (?, ?, ?, ?, ?, 0)
                    """,
                    (
                        base_model_version,
                        str(base_model_path.resolve()),
                        json.dumps(base_model_metadata),
                        now,
                        now,
                    ),
                )
            if (
                connection.execute("SELECT 1 FROM model_versions WHERE is_active = 1").fetchone()
                is None
            ):
                connection.execute(
                    "UPDATE model_versions SET is_active = 1, activated_at = ? WHERE version = ?",
                    (_now(), base_model_version),
                )
            connection.commit()

    def create_scan(
        self,
        *,
        scan_id: str,
        image_sha256: str,
        source_width: int,
        source_height: int,
        source_image_path: Path,
        rectified_image_path: Path,
        corners: list[list[float]],
        detection_method: str,
        model_version: str,
        labels: list[int],
        probabilities: list[list[float]],
        board_fen: str,
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO scans (
                    id, created_at, image_sha256, source_width, source_height,
                    source_image_path, rectified_image_path, corners_json, detection_method,
                    model_version, predicted_labels_json, predicted_probabilities_json,
                    predicted_board_fen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scan_id,
                    _now(),
                    image_sha256,
                    source_width,
                    source_height,
                    str(source_image_path),
                    str(rectified_image_path),
                    json.dumps(corners),
                    detection_method,
                    model_version,
                    json.dumps(labels),
                    json.dumps(probabilities),
                    board_fen,
                ),
            )
            connection.commit()

    def update_scan_prediction(
        self,
        *,
        scan_id: str,
        rectified_image_path: Path,
        corners: list[list[float]],
        detection_method: str,
        model_version: str,
        labels: list[int],
        probabilities: list[list[float]],
        board_fen: str,
    ) -> None:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE scans
                SET rectified_image_path = ?, corners_json = ?, detection_method = ?,
                    model_version = ?, predicted_labels_json = ?,
                    predicted_probabilities_json = ?, predicted_board_fen = ?
                WHERE id = ?
                """,
                (
                    str(rectified_image_path),
                    json.dumps(corners),
                    detection_method,
                    model_version,
                    json.dumps(labels),
                    json.dumps(probabilities),
                    board_fen,
                    scan_id,
                ),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown scan: {scan_id}")
            connection.commit()

    def get_scan(self, scan_id: str) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM scans WHERE id = ?", (scan_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown scan: {scan_id}")
        return dict(row)

    def confirm_scan(
        self,
        *,
        feedback_id: str,
        scan_id: str,
        labels: list[int],
        orientation: str,
        side_to_move: str,
        castling: str,
        en_passant: str,
        full_fen: str,
        changed_squares: int,
        consent_training: bool,
        client_session_id: str | None,
    ) -> None:
        with closing(self._connect()) as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO feedback_events (
                        id, scan_id, created_at, event_type, final_labels_json, orientation,
                        side_to_move, castling, en_passant, final_fen, changed_squares,
                        consent_training, client_session_id
                    ) VALUES (?, ?, ?, 'confirmed', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback_id,
                        scan_id,
                        _now(),
                        json.dumps(labels),
                        orientation,
                        side_to_move,
                        castling,
                        en_passant,
                        full_fen,
                        changed_squares,
                        int(consent_training),
                        client_session_id,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ValueError("This scan has already been confirmed") from exc
            connection.commit()

    def get_active_model(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM model_versions
                WHERE is_active = 1
                ORDER BY activated_at DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            raise RuntimeError("No active model is registered")
        return dict(row)

    def start_training_run(
        self,
        *,
        run_id: str,
        base_model_version: str,
        training_example_count: int,
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO training_runs (
                    id, created_at, status, base_model_version, training_example_count
                ) VALUES (?, ?, 'running', ?, ?)
                """,
                (run_id, _now(), base_model_version, training_example_count),
            )
            connection.commit()

    def complete_training_run(
        self,
        *,
        run_id: str,
        candidate_model_version: str,
        metrics: dict[str, Any],
    ) -> None:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE training_runs
                SET completed_at = ?, status = 'completed', candidate_model_version = ?,
                    metrics_json = ?
                WHERE id = ?
                """,
                (_now(), candidate_model_version, json.dumps(metrics), run_id),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"Unknown training run: {run_id}")
            connection.commit()

    def get_model(self, version: str) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM model_versions WHERE version = ?",
                (version,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown model version: {version}")
        return dict(row)

    def register_candidate(
        self,
        *,
        version: str,
        artifact_path: Path,
        metadata: dict[str, Any],
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO model_versions (
                    version, artifact_path, metadata_json, created_at, is_active
                ) VALUES (?, ?, ?, ?, 0)
                """,
                (version, str(artifact_path.resolve()), json.dumps(metadata), _now()),
            )
            connection.commit()

    def promote_model(self, version: str) -> None:
        with closing(self._connect()) as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM model_versions WHERE version = ?", (version,)
                ).fetchone()
                is None
            ):
                raise KeyError(f"Unknown model version: {version}")
            connection.execute("UPDATE model_versions SET is_active = 0 WHERE is_active = 1")
            connection.execute(
                "UPDATE model_versions SET is_active = 1, activated_at = ? WHERE version = ?",
                (_now(), version),
            )
            connection.execute(
                "UPDATE training_runs SET promoted = 1 WHERE candidate_model_version = ?",
                (version,),
            )
            connection.commit()

    def learning_cycle_progress(self) -> dict[str, int]:
        with closing(self._connect()) as connection:
            confirmed = connection.execute(
                "SELECT COUNT(*) AS count FROM feedback_events WHERE consent_training = 1"
            ).fetchone()
            previous = connection.execute(
                """
                SELECT COALESCE(MAX(training_example_count), 0) AS count
                FROM training_runs
                WHERE status = 'completed'
                """
            ).fetchone()
        total = int(confirmed["count"])
        previous_total = int(previous["count"])
        return {
            "total_training_boards": total,
            "boards_in_last_completed_run": previous_total,
            "new_training_boards": max(0, total - previous_total),
        }

    def latest_candidate(self) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM model_versions
                WHERE is_active = 0
                ORDER BY created_at DESC
                LIMIT 1
                """
            ).fetchone()
        return None if row is None else dict(row)

    def learning_status(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            counts = connection.execute(
                """
                SELECT
                    COUNT(*) AS confirmed_boards,
                    COALESCE(SUM(CASE WHEN changed_squares > 0 THEN 1 ELSE 0 END), 0)
                        AS corrected_boards,
                    COALESCE(SUM(consent_training), 0) AS training_boards
                FROM feedback_events
                """
            ).fetchone()
            active = connection.execute(
                "SELECT version FROM model_versions WHERE is_active = 1 LIMIT 1"
            ).fetchone()
        return {
            "confirmed_boards": int(counts["confirmed_boards"]),
            "corrected_boards": int(counts["corrected_boards"]),
            "training_boards": int(counts["training_boards"]),
            "active_model": str(active["version"]),
        }

    def training_examples(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    f.id AS feedback_id, f.scan_id, f.created_at, f.final_labels_json,
                    f.orientation, f.side_to_move, f.final_fen, f.changed_squares,
                    f.client_session_id, s.rectified_image_path, s.model_version,
                    s.predicted_labels_json, s.predicted_probabilities_json,
                    s.image_sha256
                FROM feedback_events f
                JOIN scans s ON s.id = f.scan_id
                WHERE f.consent_training = 1
                ORDER BY f.created_at
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection


def _now() -> str:
    return datetime.now(UTC).isoformat()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_versions (
    version TEXT PRIMARY KEY,
    artifact_path TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    activated_at TEXT,
    is_active INTEGER NOT NULL DEFAULT 0 CHECK (is_active IN (0, 1))
);

CREATE TABLE IF NOT EXISTS scans (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    image_sha256 TEXT NOT NULL,
    source_width INTEGER NOT NULL,
    source_height INTEGER NOT NULL,
    source_image_path TEXT NOT NULL,
    rectified_image_path TEXT NOT NULL,
    corners_json TEXT NOT NULL,
    detection_method TEXT NOT NULL,
    model_version TEXT NOT NULL REFERENCES model_versions(version),
    predicted_labels_json TEXT NOT NULL,
    predicted_probabilities_json TEXT NOT NULL,
    predicted_board_fen TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback_events (
    id TEXT PRIMARY KEY,
    scan_id TEXT NOT NULL UNIQUE REFERENCES scans(id),
    created_at TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type = 'confirmed'),
    final_labels_json TEXT NOT NULL,
    orientation TEXT NOT NULL CHECK (orientation IN ('white', 'black')),
    side_to_move TEXT NOT NULL CHECK (side_to_move IN ('w', 'b')),
    castling TEXT NOT NULL,
    en_passant TEXT NOT NULL,
    final_fen TEXT NOT NULL,
    changed_squares INTEGER NOT NULL,
    consent_training INTEGER NOT NULL CHECK (consent_training IN (0, 1)),
    client_session_id TEXT
);

CREATE TABLE IF NOT EXISTS training_runs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    status TEXT NOT NULL,
    base_model_version TEXT NOT NULL,
    candidate_model_version TEXT,
    training_example_count INTEGER NOT NULL,
    metrics_json TEXT,
    promoted INTEGER NOT NULL DEFAULT 0 CHECK (promoted IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_feedback_training
ON feedback_events(consent_training, created_at);
"""
