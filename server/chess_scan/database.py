"""SQLite persistence for scans, immutable human feedback, and model versions."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from chess_scan.errors import ScanAlreadyConfirmedError, ScanExpiredError
from chess_scan.model_artifact import verify_model_artifact


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
            _migrate_scans_table(connection)
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
            else:
                connection.execute(
                    """
                    UPDATE model_versions
                    SET artifact_path = ?, metadata_json = ?
                    WHERE version = ?
                    """,
                    (
                        str(base_model_path.resolve()),
                        json.dumps(base_model_metadata),
                        base_model_version,
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
    ) -> Path:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            scan = connection.execute(
                "SELECT state, rectified_image_path FROM scans WHERE id = ?",
                (scan_id,),
            ).fetchone()
            if scan is None:
                raise KeyError(f"Unknown scan: {scan_id}")
            if scan["state"] == "confirmed":
                raise ScanAlreadyConfirmedError("This scan has already been confirmed")
            if scan["state"] == "expired":
                raise ScanExpiredError("This scan has expired")

            cursor = connection.execute(
                """
                UPDATE scans
                SET rectified_image_path = ?, corners_json = ?, detection_method = ?,
                    model_version = ?, predicted_labels_json = ?,
                    predicted_probabilities_json = ?, predicted_board_fen = ?
                WHERE id = ? AND state = 'open'
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
                raise RuntimeError(f"Scan state changed while reprocessing: {scan_id}")
            connection.commit()
        return Path(scan["rectified_image_path"])

    def scan_for_display(self, scan_id: str) -> dict[str, Any]:
        scan = self._scan_projection(
            scan_id,
            """
            state, source_width, source_height, corners_json, detection_method,
            model_version, predicted_labels_json, predicted_probabilities_json,
            predicted_board_fen
            """,
        )
        if scan["state"] == "expired":
            raise ScanExpiredError("This scan has expired")
        return scan

    def scan_for_reprocessing(self, scan_id: str) -> dict[str, Any]:
        return self._scan_projection(
            scan_id,
            "source_image_path, rectified_image_path, source_width, source_height",
        )

    def source_image_path(self, scan_id: str) -> Path:
        row = self._scan_projection(scan_id, "source_image_path")
        return Path(row["source_image_path"])

    def rectified_image_path(self, scan_id: str) -> Path:
        row = self._scan_projection(scan_id, "rectified_image_path")
        return Path(row["rectified_image_path"])

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
        consent_training: bool,
        client_session_id: str | None,
    ) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            scan = connection.execute(
                """
                SELECT predicted_labels_json, source_image_path, rectified_image_path, state
                FROM scans
                WHERE id = ?
                """,
                (scan_id,),
            ).fetchone()
            if scan is None:
                raise KeyError(f"Unknown scan: {scan_id}")
            if scan["state"] == "confirmed":
                raise ScanAlreadyConfirmedError("This scan has already been confirmed")
            if scan["state"] == "expired":
                raise ScanExpiredError("This scan has expired")
            if consent_training and not Path(scan["rectified_image_path"]).is_file():
                raise ValueError("The rectified board is no longer available for training")

            predicted_labels = [int(value) for value in json.loads(scan["predicted_labels_json"])]
            changed_squares = sum(
                predicted != final
                for predicted, final in zip(predicted_labels, labels, strict=True)
            )
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
                duplicate = connection.execute(
                    "SELECT 1 FROM feedback_events WHERE scan_id = ?", (scan_id,)
                ).fetchone()
                if duplicate is not None:
                    raise ScanAlreadyConfirmedError("This scan has already been confirmed") from exc
                raise
            connection.execute(
                "UPDATE scans SET state = 'confirmed', cleanup_completed_at = NULL WHERE id = ?",
                (scan_id,),
            )
            connection.commit()
        return {
            "scan_id": scan_id,
            "changed_squares": changed_squares,
            "source_image_path": str(scan["source_image_path"]),
            "rectified_image_path": str(scan["rectified_image_path"]),
        }

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
        artifact_path = artifact_path.resolve()
        verify_model_artifact(artifact_path, metadata)
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO model_versions (
                    version, artifact_path, metadata_json, created_at, is_active
                ) VALUES (?, ?, ?, ?, 0)
                """,
                (version, str(artifact_path), json.dumps(metadata), _now()),
            )
            connection.commit()

    def promote_model(self, version: str) -> None:
        with closing(self._connect()) as connection:
            model = connection.execute(
                "SELECT artifact_path, metadata_json FROM model_versions WHERE version = ?",
                (version,),
            ).fetchone()
            if model is None:
                raise KeyError(f"Unknown model version: {version}")
            verify_model_artifact(
                Path(model["artifact_path"]),
                str(model["metadata_json"]),
            )
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
        return list(self.iter_training_examples())

    def iter_training_examples(self) -> Iterator[dict[str, Any]]:
        yield from self._iter_rows(
            """
            SELECT
                f.id AS feedback_id, f.scan_id, f.created_at, f.final_labels_json,
                f.orientation, f.side_to_move, f.final_fen, f.changed_squares,
                f.client_session_id, s.rectified_image_path, s.model_version,
                s.predicted_labels_json, s.image_sha256
            FROM feedback_events f
            JOIN scans s ON s.id = f.scan_id
            WHERE f.consent_training = 1
            ORDER BY f.created_at
            """
        )

    def iter_preference_examples(self) -> Iterator[dict[str, Any]]:
        yield from self._iter_rows(
            """
            SELECT
                f.id AS feedback_id, f.changed_squares, f.final_labels_json,
                s.rectified_image_path, s.model_version, s.predicted_labels_json,
                s.predicted_probabilities_json
            FROM feedback_events f
            JOIN scans s ON s.id = f.scan_id
            WHERE f.consent_training = 1 AND f.changed_squares > 0
            ORDER BY f.created_at
            """
        )

    def feedback_split_assignments(self) -> dict[str, str]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                "SELECT feedback_id, split FROM feedback_split_assignments"
            ).fetchall()
        return {str(row["feedback_id"]): str(row["split"]) for row in rows}

    def save_feedback_split_assignments(self, assignments: dict[str, str]) -> None:
        if not assignments:
            return
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for feedback_id, split in assignments.items():
                existing = connection.execute(
                    "SELECT split FROM feedback_split_assignments WHERE feedback_id = ?",
                    (feedback_id,),
                ).fetchone()
                if existing is not None and existing["split"] != split:
                    raise ValueError(f"Feedback split assignment is immutable: {feedback_id}")
                connection.execute(
                    """
                    INSERT OR IGNORE INTO feedback_split_assignments (
                        feedback_id, split, assigned_at
                    ) VALUES (?, ?, ?)
                    """,
                    (feedback_id, split, _now()),
                )
            connection.commit()

    def scan_files_for_cleanup(
        self,
        *,
        created_before: str,
        retry_before: str,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE scans
                SET state = 'expired', expired_at = ?, cleanup_completed_at = NULL
                WHERE id IN (
                    SELECT id
                    FROM scans
                    WHERE state = 'open' AND created_at < ?
                    ORDER BY created_at
                    LIMIT ?
                )
                """,
                (_now(), created_before, limit),
            )
            rows = connection.execute(
                """
                SELECT
                    s.id, s.source_image_path, s.rectified_image_path,
                    CASE
                        WHEN s.state = 'expired' OR COALESCE(f.consent_training, 0) = 0 THEN 1
                        ELSE 0
                    END AS delete_rectified
                FROM scans s
                LEFT JOIN feedback_events f ON f.scan_id = s.id
                WHERE s.state != 'open'
                  AND s.cleanup_completed_at IS NULL
                  AND (s.cleanup_attempted_at IS NULL OR s.cleanup_attempted_at < ?)
                ORDER BY s.cleanup_attempted_at IS NOT NULL, s.cleanup_attempted_at, s.created_at
                LIMIT ?
                """,
                (retry_before, limit),
            ).fetchall()
            if rows:
                placeholders = ", ".join("?" for _ in rows)
                connection.execute(
                    f"UPDATE scans SET cleanup_attempted_at = ? WHERE id IN ({placeholders})",
                    (_now(), *(str(row["id"]) for row in rows)),
                )
            connection.commit()
        return [dict(row) for row in rows]

    def has_cleanup_work(self, *, created_before: str, retry_before: str) -> bool:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM scans
                WHERE (state = 'open' AND created_at < ?)
                   OR (
                       state != 'open' AND cleanup_completed_at IS NULL
                       AND (cleanup_attempted_at IS NULL OR cleanup_attempted_at < ?)
                   )
                LIMIT 1
                """,
                (created_before, retry_before),
            ).fetchone()
        return row is not None

    def referenced_file_paths(self) -> set[Path]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT source_image_path AS path FROM scans WHERE state = 'open'
                UNION ALL
                SELECT rectified_image_path AS path FROM scans WHERE state = 'open'
                UNION ALL
                SELECT s.rectified_image_path AS path
                FROM scans s
                JOIN feedback_events f ON f.scan_id = s.id
                WHERE s.state = 'confirmed' AND f.consent_training = 1
                """
            ).fetchall()
        return {Path(row["path"]).resolve() for row in rows}

    def complete_scan_cleanup(self, scan_ids: list[str]) -> None:
        if not scan_ids:
            return
        placeholders = ", ".join("?" for _ in scan_ids)
        with closing(self._connect()) as connection:
            connection.execute(
                f"UPDATE scans SET cleanup_completed_at = ? WHERE id IN ({placeholders})",
                (_now(), *scan_ids),
            )
            connection.commit()

    def _scan_projection(self, scan_id: str, columns: str) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                f"SELECT {columns} FROM scans WHERE id = ?",
                (scan_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown scan: {scan_id}")
        return dict(row)

    def _iter_rows(self, query: str) -> Iterator[dict[str, Any]]:
        with closing(self._connect()) as connection:
            cursor = connection.execute(query)
            while rows := cursor.fetchmany(100):
                yield from (dict(row) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _migrate_scans_table(connection: sqlite3.Connection) -> None:
    columns = {
        str(row["name"]) for row in connection.execute("PRAGMA table_info(scans)").fetchall()
    }
    state_added = "state" not in columns
    if state_added:
        connection.execute(
            """
            ALTER TABLE scans
            ADD COLUMN state TEXT NOT NULL DEFAULT 'open'
                CHECK (state IN ('open', 'confirmed', 'expired'))
            """
        )
    if "expired_at" not in columns:
        connection.execute("ALTER TABLE scans ADD COLUMN expired_at TEXT")
    if "cleanup_attempted_at" not in columns:
        connection.execute("ALTER TABLE scans ADD COLUMN cleanup_attempted_at TEXT")
    if "cleanup_completed_at" not in columns:
        connection.execute("ALTER TABLE scans ADD COLUMN cleanup_completed_at TEXT")
    if state_added:
        connection.execute(
            """
            UPDATE scans
            SET state = CASE
                WHEN expired_at IS NOT NULL THEN 'expired'
                WHEN EXISTS(SELECT 1 FROM feedback_events f WHERE f.scan_id = scans.id)
                    THEN 'confirmed'
                ELSE 'open'
            END
            """
        )
    connection.executescript(
        """
        CREATE INDEX IF NOT EXISTS idx_scans_open_cleanup
        ON scans(created_at) WHERE state = 'open';

        CREATE INDEX IF NOT EXISTS idx_scans_pending_cleanup
        ON scans(cleanup_attempted_at, created_at)
        WHERE state != 'open' AND cleanup_completed_at IS NULL;
        """
    )


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
    predicted_board_fen TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'open' CHECK (state IN ('open', 'confirmed', 'expired')),
    expired_at TEXT,
    cleanup_attempted_at TEXT,
    cleanup_completed_at TEXT
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

CREATE TABLE IF NOT EXISTS feedback_split_assignments (
    feedback_id TEXT PRIMARY KEY REFERENCES feedback_events(id),
    split TEXT NOT NULL CHECK (split IN ('train', 'selection', 'gate', 'quarantine')),
    assigned_at TEXT NOT NULL
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
