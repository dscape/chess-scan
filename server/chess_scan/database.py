"""SQLite persistence for scans, immutable human feedback, and model versions."""

from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from chess_scan.commentary_contract import (
    CommentaryRunRecord,
    CommentarySelectionError,
    commentary_claim_candidates,
    validate_commentary_response,
    verified_commentary_selection,
)
from chess_scan.commentary_limits import COMMENTARY_MAX_LESSONS
from chess_scan.commentary_narrative import build_coaching_sections
from chess_scan.errors import (
    PositionReviewFeedbackConflictError,
    PositionReviewNotFoundError,
    ScanAlreadyConfirmedError,
    ScanExpiredError,
)
from chess_scan.learning import (
    INITIAL_TRAINING_BOARDS,
    MAX_BOARDS_PER_CLIENT,
    NEW_TRAINING_BOARDS,
    diverse_shadow_rows,
)
from chess_scan.model_artifact import verify_model_artifact
from chess_scan.schemas import (
    COMMENTARY_FALLBACK_MESSAGE,
    COMMENTARY_NO_CLAIM_MESSAGE,
    COMMENTARY_REVIEW_HEADLINE,
    PositionCoachingResponse,
    PositionReviewResponse,
    review_attempt_headline,
)

_DATABASE_SCHEMA_VERSION = 1
_V1_FOCUS_HEADLINES = {
    "cause": "Follow the cause and effect.",
    "concept": "Focus on the verified concept.",
    "comparison": "Compare the checked ideas.",
}
_V1_REVIEW_HEADLINE = "Verified review"
_V1_FALLBACK_MESSAGE = (
    "Deeper coaching is unavailable right now. "
    "The verified evidence-backed lesson is shown instead."
)
_V1_NO_CLAIM_MESSAGE = "No deeper evidence-backed lesson is available for this position."

CommentaryReservationStatus = Literal[
    "stored",
    "busy",
    "global_busy",
    "budget_exhausted",
    "global_budget_exhausted",
    "reserved",
]


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
            _enable_wal(connection)
            connection.executescript(_SCHEMA)
            _migrate_scans_table(connection)
            _migrate_commentary_tables(connection)
            connection.execute("BEGIN IMMEDIATE")
            try:
                now = _now()
                connection.execute(
                    """
                    INSERT INTO model_versions (
                        version, artifact_path, metadata_json, created_at, activated_at, is_active
                    ) VALUES (?, ?, ?, ?, ?, 0)
                    ON CONFLICT(version) DO UPDATE SET
                        artifact_path = excluded.artifact_path,
                        metadata_json = excluded.metadata_json
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
                    connection.execute(
                        "SELECT 1 FROM model_versions WHERE is_active = 1"
                    ).fetchone()
                    is None
                ):
                    connection.execute(
                        """
                        UPDATE model_versions
                        SET is_active = 1, activated_at = ?
                        WHERE version = ?
                        """,
                        (now, base_model_version),
                    )
                connection.commit()
            except Exception:
                connection.rollback()
                raise

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
        if scan["state"] == "confirmed":
            raise ScanAlreadyConfirmedError("This scan has already been confirmed")
        if scan["state"] == "expired":
            raise ScanExpiredError("This scan has expired")
        return scan

    def scan_for_reprocessing(self, scan_id: str) -> dict[str, Any]:
        return self._scan_projection(
            scan_id,
            "source_image_path, rectified_image_path, source_width, source_height",
        )

    def source_image_path(self, scan_id: str) -> Path:
        scan = self._scan_projection(scan_id, "state, source_image_path")
        if scan["state"] == "confirmed":
            raise ScanAlreadyConfirmedError("This scan has already been confirmed")
        if scan["state"] == "expired":
            raise ScanExpiredError("This scan has expired")
        return Path(scan["source_image_path"])

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

    def review_position(self, feedback_id: str) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT id AS feedback_id, final_fen, orientation, changed_squares
                FROM feedback_events
                WHERE id = ? AND event_type = 'confirmed'
                """,
                (feedback_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown review: {feedback_id}")
        return dict(row)

    def save_position_review(
        self,
        *,
        review_id: str,
        feedback_id: str,
        fen: str,
        schema_version: str,
        engine: str,
        request: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        with closing(self._connect()) as connection:
            inserted = connection.execute(
                """
                INSERT INTO position_review_runs (
                    id, feedback_id, created_at, schema_version, engine,
                    request_json, response_json
                )
                SELECT ?, feedback.id, ?, ?, ?, ?, ?
                FROM feedback_events feedback
                WHERE feedback.id = ?
                  AND feedback.event_type = 'confirmed'
                  AND feedback.final_fen = ?
                """,
                (
                    review_id,
                    _now(),
                    schema_version,
                    engine,
                    json.dumps(request, separators=(",", ":")),
                    json.dumps(response, separators=(",", ":")),
                    feedback_id,
                    fen,
                ),
            )
            if inserted.rowcount == 0:
                feedback = connection.execute(
                    "SELECT 1 FROM feedback_events WHERE id = ? AND event_type = 'confirmed'",
                    (feedback_id,),
                ).fetchone()
                if feedback is None:
                    raise KeyError(f"Unknown review feedback: {feedback_id}")
                raise ValueError("Review FEN does not match the confirmed position")
            connection.commit()

    def position_review_run(self, review_id: str) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT response_json FROM position_review_runs WHERE id = ?",
                (review_id,),
            ).fetchone()
        if row is None:
            raise PositionReviewNotFoundError(f"Unknown position review: {review_id}")
        payload = _json_object(str(row["response_json"]), label="position review")
        return _served_position_review(payload)

    def reserve_commentary_planner_run(
        self,
        *,
        review_id: str,
        reservation_id: str,
        lease_seconds: float,
        max_runs_per_feedback: int,
        max_runs_per_hour: int,
        max_concurrent: int,
    ) -> CommentaryReservationStatus:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            now = datetime.now(UTC)
            if connection.execute(
                "SELECT 1 FROM commentary_planner_runs WHERE review_id = ?",
                (review_id,),
            ).fetchone():
                return "stored"
            review = connection.execute(
                "SELECT feedback_id FROM position_review_runs WHERE id = ?",
                (review_id,),
            ).fetchone()
            if review is None:
                raise PositionReviewNotFoundError(f"Unknown position review: {review_id}")
            connection.execute(
                "DELETE FROM commentary_planner_reservations WHERE lease_expires_at <= ?",
                (now.isoformat(),),
            )
            if connection.execute(
                "SELECT 1 FROM commentary_planner_reservations WHERE review_id = ?",
                (review_id,),
            ).fetchone():
                connection.commit()
                return "busy"
            feedback_id = str(review["feedback_id"])
            hourly_cutoff = (now - timedelta(hours=1)).isoformat()
            recent_attempts = connection.execute(
                """
                SELECT COUNT(*) AS count FROM commentary_planner_attempts
                WHERE admitted_at >= ?
                """,
                (hourly_cutoff,),
            ).fetchone()["count"]
            if int(recent_attempts) >= max_runs_per_hour:
                connection.commit()
                return "global_budget_exhausted"
            feedback_attempts = connection.execute(
                """
                SELECT COUNT(*) AS count
                FROM commentary_planner_attempts attempt
                JOIN position_review_runs review ON review.id = attempt.review_id
                WHERE review.feedback_id = ?
                """,
                (feedback_id,),
            ).fetchone()["count"]
            if int(feedback_attempts) >= max_runs_per_feedback:
                connection.commit()
                return "budget_exhausted"
            active = connection.execute(
                "SELECT COUNT(*) AS count FROM commentary_planner_reservations"
            ).fetchone()["count"]
            if int(active) >= max_concurrent:
                connection.commit()
                return "global_busy"
            connection.execute(
                """
                INSERT INTO commentary_planner_attempts (
                    id, review_id, admitted_at
                ) VALUES (?, ?, ?)
                """,
                (reservation_id, review_id, now.isoformat()),
            )
            connection.execute(
                """
                INSERT INTO commentary_planner_reservations (
                    review_id, reservation_id, reserved_at, lease_expires_at
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    review_id,
                    reservation_id,
                    now.isoformat(),
                    (now + timedelta(seconds=lease_seconds)).isoformat(),
                ),
            )
            connection.commit()
        return "reserved"

    def save_commentary_planner_run(
        self,
        record: CommentaryRunRecord,
        *,
        reservation_id: str | None = None,
        release_reservation: bool = True,
    ) -> bool:
        _validate_commentary_run_record(record)
        response = record.response
        if reservation_id is not None and reservation_id != response.run_id:
            raise ValueError("Commentary reservation must match its planner run ID")
        if record.provider_called and reservation_id is None:
            raise ValueError("Provider calls require a matching spend reservation")
        if response.run_id is None:
            raise ValueError("Persisted coaching requires a run ID")
        run_id = response.run_id
        review_id = response.review_id
        review = PositionReviewResponse.model_validate(self.position_review_run(review_id))
        _validate_commentary_candidates(record, review=review)
        request_json = json.dumps(record.request, separators=(",", ":"))
        accepted_claims_json = json.dumps(record.accepted_claim_ids, separators=(",", ":"))
        claim_candidates_json = json.dumps(
            [candidate.model_dump(mode="json") for candidate in record.claim_candidates],
            separators=(",", ":"),
        )
        response_json = response.model_dump_json()
        created_at = _now()

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            if reservation_id is not None:
                reservation = connection.execute(
                    """
                    SELECT 1 FROM commentary_planner_reservations
                    WHERE review_id = ? AND reservation_id = ?
                    """,
                    (review_id, reservation_id),
                ).fetchone()
                if reservation is None:
                    return False
            try:
                connection.execute(
                    """
                    INSERT INTO commentary_planner_runs (
                        id, review_id, created_at, status, provider, model,
                        prompt_version, planner_version, request_json, raw_output,
                        accepted_claims_json, claim_candidates_json, response_json,
                        latency_ms, input_tokens, output_tokens, error_code, provider_called
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        review_id,
                        created_at,
                        response.status,
                        record.provider,
                        record.model,
                        record.prompt_version,
                        response.planner_version,
                        request_json,
                        record.raw_output,
                        accepted_claims_json,
                        claim_candidates_json,
                        response_json,
                        record.latency_ms,
                        record.input_tokens,
                        record.output_tokens,
                        record.error_code,
                        int(record.provider_called),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                existing = connection.execute(
                    "SELECT 1 FROM commentary_planner_runs WHERE review_id = ?",
                    (review_id,),
                ).fetchone()
                if existing is not None:
                    return False
                if _is_foreign_key_violation(exc):
                    raise PositionReviewNotFoundError(
                        f"Unknown position review: {review_id}"
                    ) from exc
                raise
            if reservation_id is not None and release_reservation:
                connection.execute(
                    """
                    DELETE FROM commentary_planner_reservations
                    WHERE review_id = ? AND reservation_id = ?
                    """,
                    (review_id, reservation_id),
                )
            connection.commit()
        return True

    def release_commentary_planner_reservation(
        self,
        *,
        review_id: str,
        reservation_id: str,
    ) -> bool:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                DELETE FROM commentary_planner_reservations
                WHERE review_id = ? AND reservation_id = ?
                """,
                (review_id, reservation_id),
            )
            connection.commit()
        return cursor.rowcount == 1

    def position_coaching(
        self,
        review_id: str,
        *,
        review: PositionReviewResponse | None = None,
    ) -> PositionCoachingResponse | None:
        validated = self._validated_commentary_run(review_id, review=review)
        return validated[0].response if validated is not None else None

    def commentary_planner_snapshot(self, review_id: str) -> dict[str, Any]:
        validated = self._validated_commentary_run(review_id)
        if validated is None:
            raise KeyError(f"Unknown commentary planner run: {review_id}")
        record, created_at, stored_response = validated
        return _commentary_snapshot(
            record,
            created_at=created_at,
            stored_response=stored_response,
        )

    def commentary_snapshot_from_feedback_row(
        self,
        row: Mapping[str, Any],
    ) -> dict[str, Any]:
        review_id = str(row["review_id"])
        review_response_json = str(row["response_json"])
        review = PositionReviewResponse.model_validate(
            _served_position_review(_json_object(review_response_json, label="position review"))
        )
        record = _commentary_run_record(
            row,
            review_id=review_id,
            review=review,
            prefix="commentary_",
        )
        if record.provider_called and row["commentary_attempt_id"] != record.response.run_id:
            raise ValueError(f"Stored commentary has no matching admission: {review_id}")
        return _commentary_snapshot(
            record,
            created_at=str(row["commentary_created_at"]),
            stored_response=_json_object(
                str(row["commentary_response_json"]),
                label="planner response",
            ),
        )

    def _validated_commentary_run(
        self,
        review_id: str,
        *,
        review: PositionReviewResponse | None = None,
    ) -> tuple[CommentaryRunRecord, str, dict[str, Any]] | None:
        if review is not None and review.review_id != review_id:
            raise ValueError("Stored coaching validation received a different review")
        review_join = (
            "JOIN position_review_runs review ON review.id = planner.review_id"
            if review is None
            else ""
        )
        review_column = ", review.response_json AS review_response_json" if review is None else ""
        with closing(self._connect()) as connection:
            row = connection.execute(
                f"""
                SELECT planner.*, attempt.id AS admitted_id {review_column}
                FROM commentary_planner_runs planner
                {review_join}
                LEFT JOIN commentary_planner_attempts attempt
                  ON attempt.id = planner.id AND attempt.review_id = planner.review_id
                WHERE planner.review_id = ?
                """,
                (review_id,),
            ).fetchone()
            if row is None:
                return None
        if review is None:
            review = PositionReviewResponse.model_validate(
                _served_position_review(
                    _json_object(str(row["review_response_json"]), label="position review")
                )
            )
        record = _commentary_run_record(
            row,
            review_id=review_id,
            review=review,
        )
        if record.provider_called and row["admitted_id"] is None:
            raise ValueError(f"Stored commentary has no matching admission: {review_id}")
        stored_response = _json_object(
            str(row["response_json"]),
            label="planner response",
        )
        return record, str(row["created_at"]), stored_response

    def append_position_review_feedback(
        self,
        *,
        feedback_id: str,
        review_id: str,
        rating: str,
        reason: str,
        detail: str | None,
        coaching_status: str,
        commentary_run_id: str | None,
    ) -> None:
        run_required = coaching_status in {"accepted", "fallback"}
        if run_required != (commentary_run_id is not None):
            raise PositionReviewFeedbackConflictError(
                "Presented coaching requires its immutable run ID"
            )
        with closing(self._connect()) as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO position_review_feedback (
                        id, review_id, created_at, rating, reason, detail,
                        coaching_status, commentary_run_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        feedback_id,
                        review_id,
                        _now(),
                        rating,
                        reason,
                        detail,
                        coaching_status,
                        commentary_run_id,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                if _is_foreign_key_violation(exc):
                    raise PositionReviewNotFoundError(
                        f"Unknown position review: {review_id}"
                    ) from exc
                if getattr(exc, "sqlite_errorname", None) == "SQLITE_CONSTRAINT_TRIGGER":
                    raise PositionReviewFeedbackConflictError(
                        "Feedback contradicts its stored coaching run"
                    ) from exc
                raise
            connection.commit()

    def append_position_review_adjudication(
        self,
        *,
        adjudication_id: str,
        review_feedback_id: str,
        reviewer: str,
        disposition: str,
        notes: str,
        regression_fixture: str | None,
    ) -> None:
        allowed = {"confirmed_issue", "rejected", "duplicate", "approved_fix"}
        if disposition not in allowed:
            raise ValueError(f"Unknown review adjudication disposition: {disposition}")
        if not reviewer.strip() or not notes.strip():
            raise ValueError("Review adjudication requires a reviewer and notes")
        if disposition == "approved_fix" and not (regression_fixture or "").strip():
            raise ValueError("An approved review fix requires a regression fixture")
        with closing(self._connect()) as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO position_review_feedback_adjudications (
                        id, review_feedback_id, created_at, reviewer,
                        disposition, notes, regression_fixture
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        adjudication_id,
                        review_feedback_id,
                        _now(),
                        reviewer,
                        disposition,
                        notes,
                        regression_fixture,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                if _is_foreign_key_violation(exc):
                    raise KeyError(
                        f"Unknown position review feedback: {review_feedback_id}"
                    ) from exc
                raise
            connection.commit()

    def iter_position_review_feedback(
        self,
        *,
        rating: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        predicate = "" if rating is None else "WHERE feedback.rating = ?"
        parameters = () if rating is None else (rating,)
        yield from self._iter_rows(
            f"""
            SELECT
                feedback.id AS review_feedback_id,
                feedback.created_at,
                feedback.rating,
                feedback.reason,
                feedback.detail,
                feedback.coaching_status,
                feedback.commentary_run_id AS presented_commentary_run_id,
                run.id AS review_id,
                run.feedback_id AS position_feedback_id,
                run.schema_version,
                run.engine,
                run.request_json,
                run.response_json,
                planner.id AS commentary_id,
                planner.review_id AS commentary_review_id,
                planner.created_at AS commentary_created_at,
                planner.status AS commentary_status,
                planner.provider AS commentary_provider,
                planner.model AS commentary_model,
                planner.prompt_version AS commentary_prompt_version,
                planner.planner_version AS commentary_planner_version,
                planner.request_json AS commentary_request_json,
                planner.raw_output AS commentary_raw_output,
                planner.accepted_claims_json AS commentary_accepted_claims_json,
                planner.claim_candidates_json AS commentary_claim_candidates_json,
                planner.response_json AS commentary_response_json,
                planner.latency_ms AS commentary_latency_ms,
                planner.input_tokens AS commentary_input_tokens,
                planner.output_tokens AS commentary_output_tokens,
                planner.error_code AS commentary_error_code,
                planner.provider_called AS commentary_provider_called,
                attempt.id AS commentary_attempt_id
            FROM position_review_feedback feedback
            JOIN position_review_runs run ON run.id = feedback.review_id
            LEFT JOIN commentary_planner_runs planner
              ON planner.id = feedback.commentary_run_id
            LEFT JOIN commentary_planner_attempts attempt
              ON attempt.id = planner.id AND attempt.review_id = planner.review_id
            {predicate}
            ORDER BY feedback.created_at, feedback.id
            """,
            parameters,
        )

    def iter_position_review_adjudications_for_export(
        self,
        *,
        rating: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        predicate = "" if rating is None else "AND feedback.rating = ?"
        parameters = () if rating is None else (rating,)
        yield from self._iter_rows(
            f"""
            SELECT
                adjudication.id AS adjudication_id,
                adjudication.review_feedback_id,
                adjudication.created_at,
                adjudication.reviewer,
                adjudication.disposition,
                adjudication.notes,
                adjudication.regression_fixture
            FROM position_review_feedback feedback
            JOIN position_review_feedback_adjudications adjudication
              ON adjudication.review_feedback_id = feedback.id
            WHERE 1 = 1 {predicate}
            ORDER BY feedback.created_at, feedback.id,
                     adjudication.created_at, adjudication.id
            """,
            parameters,
        )

    def iter_position_review_adjudications(self) -> Iterator[dict[str, Any]]:
        yield from self._iter_rows(
            """
            SELECT
                adjudication.id AS adjudication_id,
                adjudication.review_feedback_id,
                adjudication.created_at,
                adjudication.reviewer,
                adjudication.disposition,
                adjudication.notes,
                adjudication.regression_fixture,
                feedback.review_id
            FROM position_review_feedback_adjudications adjudication
            JOIN position_review_feedback feedback
              ON feedback.id = adjudication.review_feedback_id
            ORDER BY adjudication.created_at, adjudication.id
            """
        )

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

    def learning_feedback_snapshot(
        self,
        *,
        min_total_boards: int,
        min_new_boards: int,
        max_boards_per_client: int = MAX_BOARDS_PER_CLIENT,
    ) -> dict[str, list[str]] | None:
        with closing(self._connect()) as connection:
            active_cycle = connection.execute(
                """
                SELECT 1 FROM learning_cycles
                WHERE state IN ('training', 'benchmarking', 'shadowing')
                LIMIT 1
                """
            ).fetchone()
            if active_cycle is not None:
                return None
            accepted = connection.execute(
                """
                SELECT f.id
                FROM feedback_events f
                JOIN feedback_learning_pool p ON p.feedback_id = f.id
                WHERE f.consent_training = 1 AND p.state = 'accepted'
                ORDER BY f.created_at
                """
            ).fetchall()
            pending = connection.execute(
                """
                SELECT f.id, f.client_session_id, s.image_sha256
                FROM feedback_events f
                JOIN scans s ON s.id = f.scan_id
                LEFT JOIN feedback_learning_pool p ON p.feedback_id = f.id
                WHERE f.consent_training = 1 AND p.feedback_id IS NULL
                ORDER BY f.created_at
                """
            ).fetchall()
        accepted_ids = [str(row["id"]) for row in accepted]
        pending_ids = _limited_feedback_ids(
            pending,
            max_boards_per_client=max_boards_per_client,
        )
        required = min_new_boards if accepted_ids else min_total_boards
        if len(pending_ids) < required:
            return None
        return {"accepted": accepted_ids, "batch": pending_ids}

    def create_learning_cycle(
        self,
        *,
        cycle_id: str,
        base_model_version: str,
        accepted_feedback_ids: list[str],
        batch_feedback_ids: list[str],
        shadow_target_boards: int,
    ) -> dict[str, Any]:
        if not batch_feedback_ids:
            raise ValueError("A learning cycle requires a new feedback batch")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            active = connection.execute(
                """
                SELECT 1 FROM learning_cycles
                WHERE state IN ('training', 'benchmarking', 'shadowing')
                LIMIT 1
                """
            ).fetchone()
            if active is not None:
                raise RuntimeError("A learning cycle is already active")
            active_model = connection.execute(
                "SELECT version FROM model_versions WHERE is_active = 1 LIMIT 1"
            ).fetchone()
            if active_model is None or active_model["version"] != base_model_version:
                raise RuntimeError("The requested learning-cycle base is not active")
            now = _now()
            connection.execute(
                """
                INSERT INTO learning_cycles (
                    id, created_at, updated_at, state, base_model_version,
                    shadow_target_boards, metrics_json
                ) VALUES (?, ?, ?, 'training', ?, ?, '{}')
                """,
                (cycle_id, now, now, base_model_version, shadow_target_boards),
            )
            for role, feedback_ids in (
                ("replay", accepted_feedback_ids),
                ("batch", batch_feedback_ids),
            ):
                for feedback_id in feedback_ids:
                    feedback = connection.execute(
                        "SELECT consent_training FROM feedback_events WHERE id = ?",
                        (feedback_id,),
                    ).fetchone()
                    if feedback is None or not bool(feedback["consent_training"]):
                        raise ValueError(f"Feedback is not eligible for training: {feedback_id}")
                    connection.execute(
                        """
                        INSERT INTO learning_cycle_feedback (cycle_id, feedback_id, role)
                        VALUES (?, ?, ?)
                        """,
                        (cycle_id, feedback_id, role),
                    )
            connection.commit()
        return self.get_learning_cycle(cycle_id)

    def get_learning_cycle(self, cycle_id: str) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM learning_cycles WHERE id = ?",
                (cycle_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown learning cycle: {cycle_id}")
        return dict(row)

    def active_learning_cycle(self) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT * FROM learning_cycles
                WHERE state IN ('training', 'benchmarking', 'shadowing')
                ORDER BY created_at
                LIMIT 1
                """
            ).fetchone()
        return None if row is None else dict(row)

    def learning_cycle_feedback_ids(self, cycle_id: str) -> list[str]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT feedback_id FROM learning_cycle_feedback
                WHERE cycle_id = ?
                ORDER BY role, feedback_id
                """,
                (cycle_id,),
            ).fetchall()
        return [str(row["feedback_id"]) for row in rows]

    def set_learning_candidate(self, cycle_id: str, candidate_version: str) -> None:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE learning_cycles
                SET state = 'benchmarking', candidate_model_version = ?, updated_at = ?
                WHERE id = ? AND state = 'training'
                """,
                (candidate_version, _now(), cycle_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Learning cycle is not training: {cycle_id}")
            connection.commit()

    def start_shadowing(self, cycle_id: str, metrics: dict[str, Any]) -> None:
        now = _now()
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                """
                UPDATE learning_cycles
                SET state = 'shadowing', shadow_started_at = ?, updated_at = ?, metrics_json = ?
                WHERE id = ? AND state = 'benchmarking'
                """,
                (now, now, json.dumps(metrics), cycle_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Learning cycle is not benchmarking: {cycle_id}")
            connection.commit()

    def reject_learning_cycle(
        self,
        cycle_id: str,
        *,
        reason: str,
        metrics: dict[str, Any],
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE learning_cycles
                SET state = 'rejected', updated_at = ?, decision_reason = ?, metrics_json = ?
                WHERE id = ? AND state IN ('training', 'benchmarking', 'shadowing')
                """,
                (_now(), reason, json.dumps(metrics), cycle_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Learning cycle cannot be rejected: {cycle_id}")
            self._set_cycle_batch_state(connection, cycle_id, "quarantined")
            connection.commit()

    def promote_learning_cycle(
        self,
        cycle_id: str,
        *,
        reason: str,
        metrics: dict[str, Any],
    ) -> str:
        cycle = self.get_learning_cycle(cycle_id)
        candidate_version = str(cycle["candidate_model_version"] or "")
        if not candidate_version:
            raise RuntimeError(f"Learning cycle has no candidate: {cycle_id}")
        model = self.get_model(candidate_version)
        verify_model_artifact(Path(model["artifact_path"]), str(model["metadata_json"]))

        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT version FROM model_versions WHERE is_active = 1 LIMIT 1"
            ).fetchone()
            if current is None or current["version"] != cycle["base_model_version"]:
                raise RuntimeError("The active model changed during shadow evaluation")
            now = _now()
            connection.execute("UPDATE model_versions SET is_active = 0 WHERE is_active = 1")
            connection.execute(
                "UPDATE model_versions SET is_active = 1, activated_at = ? WHERE version = ?",
                (now, candidate_version),
            )
            connection.execute(
                "UPDATE training_runs SET promoted = 1 WHERE candidate_model_version = ?",
                (candidate_version,),
            )
            cursor = connection.execute(
                """
                UPDATE learning_cycles
                SET state = 'promoted', updated_at = ?, decision_reason = ?, metrics_json = ?
                WHERE id = ? AND state = 'shadowing'
                """,
                (now, reason, json.dumps(metrics), cycle_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"Learning cycle is not shadowing: {cycle_id}")
            self._set_cycle_batch_state(connection, cycle_id, "accepted")
            connection.commit()
        return candidate_version

    def shadow_examples(self, cycle_id: str) -> list[dict[str, Any]]:
        cycle = self.get_learning_cycle(cycle_id)
        if cycle["state"] != "shadowing":
            return []
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    f.id AS feedback_id, f.created_at,
                    COALESCE(a.corrected_labels_json, f.final_labels_json) AS final_labels_json,
                    f.client_session_id, s.image_sha256, s.rectified_image_path,
                    s.model_version, s.predicted_labels_json
                FROM feedback_events f
                JOIN scans s ON s.id = f.scan_id
                LEFT JOIN feedback_adjudications a ON a.id = (
                    SELECT latest.id
                    FROM feedback_adjudications latest
                    WHERE latest.feedback_id = f.id
                    ORDER BY latest.created_at DESC, latest.id DESC
                    LIMIT 1
                )
                LEFT JOIN shadow_evaluations e
                    ON e.cycle_id = ? AND e.feedback_id = f.id
                WHERE f.consent_training = 1
                  AND f.created_at > ?
                  AND s.model_version = ?
                  AND e.feedback_id IS NULL
                ORDER BY f.created_at
                """,
                (cycle_id, cycle["shadow_started_at"], cycle["base_model_version"]),
            ).fetchall()
        return [dict(row) for row in rows]

    def record_shadow_evaluation(
        self,
        *,
        cycle_id: str,
        feedback_id: str,
        perceptual_hash: str,
        candidate_labels: list[int],
        active_square_errors: int,
        candidate_square_errors: int,
        active_non_empty_errors: int,
        candidate_non_empty_errors: int,
        active_board_exact: bool,
        candidate_board_exact: bool,
    ) -> None:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO shadow_evaluations (
                    cycle_id, feedback_id, evaluated_at, perceptual_hash,
                    candidate_labels_json, active_square_errors, candidate_square_errors,
                    active_non_empty_errors, candidate_non_empty_errors,
                    active_board_exact, candidate_board_exact
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    cycle_id,
                    feedback_id,
                    _now(),
                    perceptual_hash,
                    json.dumps(candidate_labels),
                    active_square_errors,
                    candidate_square_errors,
                    active_non_empty_errors,
                    candidate_non_empty_errors,
                    int(active_board_exact),
                    int(candidate_board_exact),
                ),
            )
            connection.commit()

    def shadow_evaluations(self, cycle_id: str) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT
                    e.*, f.client_session_id, s.image_sha256
                FROM shadow_evaluations e
                JOIN feedback_events f ON f.id = e.feedback_id
                JOIN scans s ON s.id = f.scan_id
                WHERE e.cycle_id = ?
                ORDER BY e.evaluated_at, e.feedback_id
                """,
                (cycle_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    @staticmethod
    def _set_cycle_batch_state(
        connection: sqlite3.Connection,
        cycle_id: str,
        state: str,
    ) -> None:
        now = _now()
        connection.execute(
            """
            INSERT INTO feedback_learning_pool (feedback_id, state, cycle_id, updated_at)
            SELECT feedback_id, ?, ?, ?
            FROM learning_cycle_feedback
            WHERE cycle_id = ? AND role = 'batch'
            ON CONFLICT(feedback_id) DO UPDATE SET
                state = excluded.state,
                cycle_id = excluded.cycle_id,
                updated_at = excluded.updated_at
            """,
            (state, cycle_id, now, cycle_id),
        )

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
            cycle = connection.execute(
                """
                SELECT * FROM learning_cycles
                WHERE state IN ('training', 'benchmarking', 'shadowing')
                ORDER BY created_at
                LIMIT 1
                """
            ).fetchone()
            accepted = connection.execute(
                "SELECT COUNT(*) AS count FROM feedback_learning_pool WHERE state = 'accepted'"
            ).fetchone()
            pending = connection.execute(
                """
                SELECT f.id, f.client_session_id, s.image_sha256
                FROM feedback_events f
                JOIN scans s ON s.id = f.scan_id
                LEFT JOIN feedback_learning_pool p ON p.feedback_id = f.id
                WHERE f.consent_training = 1 AND p.feedback_id IS NULL
                ORDER BY f.created_at
                """
            ).fetchall()
            shadow_count = 0
            if cycle is not None and cycle["state"] == "shadowing":
                shadow_rows = connection.execute(
                    """
                    SELECT e.*, f.client_session_id, s.image_sha256
                    FROM shadow_evaluations e
                    JOIN feedback_events f ON f.id = e.feedback_id
                    JOIN scans s ON s.id = f.scan_id
                    WHERE e.cycle_id = ?
                    ORDER BY e.evaluated_at, e.feedback_id
                    """,
                    (cycle["id"],),
                ).fetchall()
                shadow_count = len(diverse_shadow_rows([dict(row) for row in shadow_rows]))
        accepted_count = int(accepted["count"])
        learning_state = "collecting" if cycle is None else str(cycle["state"])
        learning_progress = len(
            _limited_feedback_ids(pending, max_boards_per_client=MAX_BOARDS_PER_CLIENT)
        )
        learning_target = NEW_TRAINING_BOARDS if accepted_count else INITIAL_TRAINING_BOARDS
        candidate = None
        if cycle is not None:
            candidate = cycle["candidate_model_version"]
            if cycle["state"] == "shadowing":
                learning_progress = shadow_count
                learning_target = int(cycle["shadow_target_boards"])
        return {
            "confirmed_boards": int(counts["confirmed_boards"]),
            "corrected_boards": int(counts["corrected_boards"]),
            "training_boards": int(counts["training_boards"]),
            "active_model": str(active["version"]),
            "learning_state": learning_state,
            "learning_progress": learning_progress,
            "learning_target": learning_target,
            "candidate_model": None if candidate is None else str(candidate),
        }

    def feedback_for_adjudication(self, feedback_id: str) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT
                    f.id AS feedback_id, f.orientation, f.side_to_move, f.castling,
                    f.en_passant,
                    COALESCE(a.corrected_labels_json, f.final_labels_json) AS final_labels_json,
                    COALESCE(a.corrected_fen, f.final_fen) AS final_fen,
                    s.predicted_labels_json
                FROM feedback_events f
                JOIN scans s ON s.id = f.scan_id
                LEFT JOIN feedback_adjudications a ON a.id = (
                    SELECT latest.id
                    FROM feedback_adjudications latest
                    WHERE latest.feedback_id = f.id
                    ORDER BY latest.created_at DESC, latest.id DESC
                    LIMIT 1
                )
                WHERE f.id = ?
                """,
                (feedback_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown feedback: {feedback_id}")
        return dict(row)

    def append_feedback_adjudication(
        self,
        *,
        adjudication_id: str,
        feedback_id: str,
        labels: list[int],
        full_fen: str,
        reason: str,
    ) -> int:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            feedback = connection.execute(
                """
                SELECT s.predicted_labels_json
                FROM feedback_events f
                JOIN scans s ON s.id = f.scan_id
                WHERE f.id = ?
                """,
                (feedback_id,),
            ).fetchone()
            if feedback is None:
                raise KeyError(f"Unknown feedback: {feedback_id}")
            predicted = [int(value) for value in json.loads(feedback["predicted_labels_json"])]
            changed_squares = sum(
                predicted_label != final_label
                for predicted_label, final_label in zip(predicted, labels, strict=True)
            )
            connection.execute(
                """
                INSERT INTO feedback_adjudications (
                    id, feedback_id, created_at, corrected_labels_json,
                    corrected_fen, changed_squares, reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    adjudication_id,
                    feedback_id,
                    _now(),
                    json.dumps(labels),
                    full_fen,
                    changed_squares,
                    reason,
                ),
            )
            connection.commit()
        return changed_squares

    def training_examples(self, feedback_ids: set[str] | None = None) -> list[dict[str, Any]]:
        rows = list(self.iter_training_examples())
        if feedback_ids is None:
            return rows
        return [row for row in rows if str(row["feedback_id"]) in feedback_ids]

    def iter_training_examples(self) -> Iterator[dict[str, Any]]:
        yield from self._iter_rows(
            """
            SELECT
                f.id AS feedback_id, f.scan_id, f.created_at,
                COALESCE(a.corrected_labels_json, f.final_labels_json) AS final_labels_json,
                f.orientation, f.side_to_move,
                COALESCE(a.corrected_fen, f.final_fen) AS final_fen,
                COALESCE(a.changed_squares, f.changed_squares) AS changed_squares,
                f.client_session_id, s.rectified_image_path, s.model_version,
                s.predicted_labels_json, s.image_sha256
            FROM feedback_events f
            JOIN scans s ON s.id = f.scan_id
            LEFT JOIN feedback_adjudications a ON a.id = (
                SELECT latest.id
                FROM feedback_adjudications latest
                WHERE latest.feedback_id = f.id
                ORDER BY latest.created_at DESC, latest.id DESC
                LIMIT 1
            )
            WHERE f.consent_training = 1
            ORDER BY f.created_at
            """
        )

    def iter_preference_examples(self) -> Iterator[dict[str, Any]]:
        yield from self._iter_rows(
            """
            SELECT
                f.id AS feedback_id,
                COALESCE(a.changed_squares, f.changed_squares) AS changed_squares,
                COALESCE(a.corrected_labels_json, f.final_labels_json) AS final_labels_json,
                s.rectified_image_path, s.model_version, s.predicted_labels_json,
                s.predicted_probabilities_json
            FROM feedback_events f
            JOIN scans s ON s.id = f.scan_id
            LEFT JOIN feedback_adjudications a ON a.id = (
                SELECT latest.id
                FROM feedback_adjudications latest
                WHERE latest.feedback_id = f.id
                ORDER BY latest.created_at DESC, latest.id DESC
                LIMIT 1
            )
            WHERE f.consent_training = 1
              AND COALESCE(a.changed_squares, f.changed_squares) > 0
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

    def _iter_rows(
        self,
        query: str,
        parameters: tuple[object, ...] = (),
    ) -> Iterator[dict[str, Any]]:
        with closing(self._connect()) as connection:
            cursor = connection.execute(query, parameters)
            while rows := cursor.fetchmany(100):
                yield from (dict(row) for row in rows)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection


def _json_object(raw: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError) as error:
        raise ValueError(f"Stored {label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"Stored {label} must be a JSON object")
    return value


def _json_array(raw: str, *, label: str) -> list[Any]:
    try:
        value = json.loads(raw)
    except (ValueError, RecursionError) as error:
        raise ValueError(f"Stored {label} is invalid JSON") from error
    if not isinstance(value, list):
        raise ValueError(f"Stored {label} must be a JSON array")
    return value


def _served_position_review(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Stored position review must be a JSON object")
    if payload.get("schema_version") != "position-analysis-4":
        return payload
    adapted = json.loads(json.dumps(payload))
    adapted["schema_version"] = "position-analysis-5"
    attempt = adapted.get("attempt")
    if isinstance(attempt, dict) and "headline" not in attempt:
        move = attempt.get("move")
        if not isinstance(move, dict):
            raise ValueError("Legacy position-review attempt has no move")
        attempt["headline"] = review_attempt_headline(
            san=str(move.get("san", "")),
            verdict=str(attempt.get("verdict", "")),
            equivalent=bool(attempt.get("equivalent")),
            lost_forced_mate=bool(attempt.get("lost_forced_mate")),
        )
    hint = adapted.get("hint")
    if isinstance(hint, dict):
        hint["id"] = "hint"
    explanation = adapted.get("explanation")
    if isinstance(explanation, list):
        for index, annotation in enumerate(explanation, start=1):
            if isinstance(annotation, dict):
                annotation["id"] = f"explanation-{index}"
    return PositionReviewResponse.model_validate(adapted).model_dump(mode="json")


def _commentary_run_record(
    row: Mapping[str, Any],
    *,
    review_id: str,
    review: PositionReviewResponse,
    prefix: str = "",
) -> CommentaryRunRecord:
    def value(column: str) -> Any:
        return row[f"{prefix}{column}"]

    response_json, claim_candidates_json, adapted_v1 = _normalized_commentary_storage(
        response_json=str(value("response_json")),
        claim_candidates_json=str(value("claim_candidates_json")),
        raw_output=value("raw_output"),
        review=review,
        run_id=str(value("id")),
        stored_status=str(value("status")),
        stored_planner_version=str(value("planner_version")),
    )
    response = PositionCoachingResponse.model_validate_json(response_json)
    expected = {
        "run_id": str(value("id")),
        "review_id": str(value("review_id")),
        "status": str(value("status")),
        "planner_version": (
            "commentary-planner-2" if adapted_v1 else str(value("planner_version"))
        ),
    }
    response_payload = response.model_dump(mode="json")
    if expected["review_id"] != review_id or any(
        response_payload.get(key) != expected_value for key, expected_value in expected.items()
    ):
        raise ValueError(f"Stored position coaching has conflicting metadata: {review_id}")
    record = CommentaryRunRecord(
        response=response,
        provider=str(value("provider")),
        model=str(value("model")),
        prompt_version=str(value("prompt_version")),
        request=_json_object(str(value("request_json")), label="planner request"),
        raw_output=value("raw_output"),
        accepted_claim_ids=tuple(
            _json_array(str(value("accepted_claims_json")), label="accepted claims")
        ),
        claim_candidates=tuple(_json_array(claim_candidates_json, label="claim candidates")),
        latency_ms=int(value("latency_ms")),
        input_tokens=value("input_tokens"),
        output_tokens=value("output_tokens"),
        error_code=value("error_code"),
        provider_called=bool(value("provider_called")),
    )
    _validate_commentary_run_record(record, legacy_v1=adapted_v1)
    _validate_commentary_candidates(record, review=review)
    return record


def _normalized_commentary_storage(
    *,
    response_json: str,
    claim_candidates_json: str,
    raw_output: str | None,
    review: PositionReviewResponse,
    run_id: str,
    stored_status: str,
    stored_planner_version: str,
) -> tuple[str, str, bool]:
    response = _json_object(response_json, label="planner response")
    is_v1 = response.get("schema_version") == "commentary-planner-1"
    if is_v1:
        _validate_v1_commentary_metadata(
            response,
            review=review,
            run_id=run_id,
            stored_status=stored_status,
            stored_planner_version=stored_planner_version,
            raw_output=raw_output,
        )

    normalized_candidates_json = claim_candidates_json
    legacy_lessons = response.get("lessons")
    if isinstance(legacy_lessons, list):
        candidates = commentary_claim_candidates(review.evidence, review.explanation)
        selected_lesson_ids: list[str] = []
        for legacy_lesson in legacy_lessons:
            match = next(
                (
                    candidate
                    for candidate in candidates
                    if candidate.lesson.model_dump(mode="json", exclude={"id"}) == legacy_lesson
                ),
                None,
            )
            if match is None:
                raise ValueError("Stored legacy coaching lesson is absent from its review")
            selected_lesson_ids.append(match.lesson.id)
        response["lesson_ids"] = selected_lesson_ids
        del response["lessons"]
        normalized_candidates_json = json.dumps(
            [candidate.model_dump(mode="json") for candidate in candidates],
            separators=(",", ":"),
        )

    if is_v1:
        return (
            _adapt_v1_commentary_response(
                response,
                raw_output=raw_output,
                review=review,
                run_id=run_id,
            ),
            normalized_candidates_json,
            True,
        )
    if not isinstance(legacy_lessons, list):
        return response_json, claim_candidates_json, False

    response["run_id"] = run_id
    return (
        json.dumps(response, separators=(",", ":")),
        normalized_candidates_json,
        False,
    )


def _validate_v1_commentary_metadata(
    response: dict[str, Any],
    *,
    review: PositionReviewResponse,
    run_id: str,
    stored_status: str,
    stored_planner_version: str,
    raw_output: str | None,
) -> None:
    has_embedded_lessons = "lessons" in response
    has_lesson_ids = "lesson_ids" in response
    if has_embedded_lessons == has_lesson_ids:
        raise ValueError("Stored V1 coaching has an invalid lesson shape")
    lesson_key = "lessons" if has_embedded_lessons else "lesson_ids"
    lessons = response.get(lesson_key)
    if not isinstance(lessons, list) or len(lessons) > COMMENTARY_MAX_LESSONS:
        raise ValueError("Stored V1 coaching has invalid lessons")
    if has_lesson_ids and (
        any(not isinstance(lesson_id, str) for lesson_id in lessons)
        or len(lessons) != len(set(lessons))
    ):
        raise ValueError("Stored V1 coaching has invalid lesson IDs")
    if has_embedded_lessons and any(not isinstance(lesson, dict) for lesson in lessons):
        raise ValueError("Stored V1 coaching has invalid embedded lessons")

    allowed_keys = {
        "schema_version",
        "review_id",
        "run_id",
        "status",
        "planner_version",
        "headline",
        lesson_key,
        "message",
    }
    required_keys = allowed_keys - ({"run_id"} if has_embedded_lessons else set())
    if not required_keys <= set(response) or not set(response) <= allowed_keys:
        raise ValueError("Stored V1 coaching has unexpected fields")
    if (
        response.get("schema_version") != "commentary-planner-1"
        or response.get("review_id") != review.review_id
        or response.get("status") != stored_status
        or response.get("planner_version") != "commentary-planner-1"
        or stored_planner_version != "commentary-planner-1"
        or ("run_id" in response and response.get("run_id") != run_id)
    ):
        raise ValueError("Stored V1 coaching has conflicting metadata")

    status = response.get("status")
    if status == "accepted":
        focus = _v1_commentary_focus(raw_output)
        if (
            not lessons
            or response.get("headline") != _V1_FOCUS_HEADLINES[focus]
            or response.get("message") is not None
        ):
            raise ValueError("Stored accepted V1 coaching has invalid fixed copy")
        return
    if status != "fallback":
        raise ValueError("Stored V1 coaching has an invalid status")
    expected_message = _V1_FALLBACK_MESSAGE if lessons else _V1_NO_CLAIM_MESSAGE
    if (
        response.get("headline") != _V1_REVIEW_HEADLINE
        or response.get("message") != expected_message
    ):
        raise ValueError("Stored fallback V1 coaching has invalid fixed copy")


def _adapt_v1_commentary_response(
    response: dict[str, Any],
    *,
    raw_output: str | None,
    review: PositionReviewResponse,
    run_id: str,
) -> str:
    lesson_ids = response.get("lesson_ids")
    status = response.get("status")
    if (
        not isinstance(lesson_ids, list)
        or any(not isinstance(lesson_id, str) for lesson_id in lesson_ids)
        or status not in {"accepted", "fallback"}
    ):
        raise ValueError("Stored V1 coaching response is invalid")
    lessons_by_id = {lesson.id: lesson for lesson in review.explanation}
    try:
        lessons = [lessons_by_id[lesson_id] for lesson_id in lesson_ids]
    except KeyError as error:
        raise ValueError("Stored V1 coaching lesson is absent from its review") from error

    focus = _v1_commentary_focus(raw_output) if status == "accepted" else "cause"
    sections = build_coaching_sections(review, lessons, focus=focus)
    adapted = PositionCoachingResponse(
        review_id=response.get("review_id"),
        run_id=run_id,
        status=status,
        planner_version="commentary-planner-2",
        focus=focus if sections else None,
        headline=sections[0].title if sections else COMMENTARY_REVIEW_HEADLINE,
        lesson_ids=lesson_ids,
        sections=sections,
        message=(
            None
            if status == "accepted"
            else COMMENTARY_FALLBACK_MESSAGE
            if sections
            else COMMENTARY_NO_CLAIM_MESSAGE
        ),
    )
    return adapted.model_dump_json()


def _v1_commentary_focus(raw_output: str | None) -> str:
    if raw_output is None:
        raise ValueError("Stored accepted V1 coaching has no raw provider output")
    selection = _json_object(raw_output, label="accepted V1 planner output")
    focus = selection.get("focus")
    if focus not in {"cause", "concept", "comparison"}:
        raise ValueError("Stored accepted V1 coaching has an invalid focus")
    return str(focus)


def _commentary_snapshot(
    record: CommentaryRunRecord,
    *,
    created_at: str,
    stored_response: dict[str, Any],
) -> dict[str, Any]:
    response = record.response
    return {
        "run_id": response.run_id,
        "created_at": created_at,
        "status": response.status,
        "provider": record.provider,
        "model": record.model,
        "prompt_version": record.prompt_version,
        "planner_version": stored_response.get("planner_version", response.planner_version),
        "serving_planner_version": response.planner_version,
        "request": record.request,
        "raw_output": record.raw_output,
        "accepted_claim_ids": list(record.accepted_claim_ids),
        "claim_candidates": [
            candidate.model_dump(mode="json") for candidate in record.claim_candidates
        ],
        "response": response.model_dump(mode="json"),
        "stored_response": stored_response,
        "latency_ms": record.latency_ms,
        "input_tokens": record.input_tokens,
        "output_tokens": record.output_tokens,
        "error_code": record.error_code,
        "provider_called": record.provider_called,
    }


def _validate_commentary_candidates(
    record: CommentaryRunRecord,
    *,
    review: PositionReviewResponse,
) -> None:
    if review.review_id != record.response.review_id:
        raise ValueError("Stored commentary candidates reference a different review")
    expected = commentary_claim_candidates(review.evidence, review.explanation)
    if record.claim_candidates != expected:
        raise ValueError(
            f"Stored commentary candidates contradict their review: {record.response.review_id}"
        )
    validate_commentary_response(review, record.response)


def _validate_commentary_run_record(
    record: CommentaryRunRecord,
    *,
    legacy_v1: bool = False,
) -> None:
    response = record.response
    if response.status == "disabled":
        raise ValueError("Disabled coaching is not a planner run")
    candidate_ids = [candidate.id for candidate in record.claim_candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Commentary claim candidates must be unique")
    if response.status == "accepted":
        if not 1 <= len(record.accepted_claim_ids) <= COMMENTARY_MAX_LESSONS:
            raise ValueError("Accepted commentary requires one or two claim IDs")
        if len(record.accepted_claim_ids) != len(set(record.accepted_claim_ids)):
            raise ValueError("Accepted commentary claim IDs must be unique")
        candidates = {candidate.id: candidate.lesson for candidate in record.claim_candidates}
        try:
            selected_lessons = [candidates[claim_id] for claim_id in record.accepted_claim_ids]
        except KeyError as exc:
            raise ValueError("Accepted commentary contains an unsupported claim ID") from exc
        if [lesson.id for lesson in selected_lessons] != response.lesson_ids:
            raise ValueError("Accepted claim order contradicts the presented lessons")
        if record.raw_output is None:
            raise ValueError("Accepted commentary requires raw provider output")
        try:
            selected_claim_ids, selected_focus = verified_commentary_selection(
                record.raw_output,
                record.claim_candidates,
                require_causal_primary=not legacy_v1,
            )
        except CommentarySelectionError as error:
            raise ValueError("Stored accepted planner output was never valid") from error
        if selected_claim_ids != record.accepted_claim_ids or selected_focus != response.focus:
            raise ValueError("Accepted planner output contradicts the presented selection")
    elif record.accepted_claim_ids:
        raise ValueError("Fallback commentary cannot contain accepted claim IDs")
    else:
        expected_fallback = [candidate.lesson.id for candidate in record.claim_candidates[:1]]
        if response.lesson_ids != expected_fallback:
            raise ValueError("Fallback lessons contradict the deterministic policy")
    if response.status == "accepted" and record.error_code is not None:
        raise ValueError("Accepted commentary cannot contain a planner error")
    if response.status == "accepted" and not record.provider_called:
        raise ValueError("Accepted commentary requires a provider call")


def _is_foreign_key_violation(error: sqlite3.IntegrityError) -> bool:
    return getattr(error, "sqlite_errorname", None) == "SQLITE_CONSTRAINT_FOREIGNKEY"


def _limited_feedback_ids(
    rows: list[sqlite3.Row],
    *,
    max_boards_per_client: int,
) -> list[str]:
    feedback_ids: list[str] = []
    client_counts: Counter[str] = Counter()
    image_hashes: set[str] = set()
    for row in rows:
        feedback_id = str(row["id"])
        client = str(row["client_session_id"] or "anonymous")
        image_hash = str(row["image_sha256"])
        if client_counts[client] >= max_boards_per_client or image_hash in image_hashes:
            continue
        feedback_ids.append(feedback_id)
        client_counts[client] += 1
        image_hashes.add(image_hash)
    return feedback_ids


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _enable_wal(connection: sqlite3.Connection) -> None:
    deadline = time.monotonic() + 30
    while True:
        try:
            mode = connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]
            if str(mode).lower() == "wal":
                return
        except sqlite3.OperationalError as exc:
            if getattr(exc, "sqlite_errorcode", None) not in {
                sqlite3.SQLITE_BUSY,
                sqlite3.SQLITE_LOCKED,
            }:
                raise
        if time.monotonic() >= deadline:
            raise sqlite3.OperationalError("Timed out while enabling SQLite WAL mode")
        time.sleep(0.05)


def _migrate_commentary_tables(connection: sqlite3.Connection) -> None:
    if int(connection.execute("PRAGMA user_version").fetchone()[0]) >= _DATABASE_SCHEMA_VERSION:
        return
    connection.commit()
    connection.execute("BEGIN EXCLUSIVE")
    try:
        if int(connection.execute("PRAGMA user_version").fetchone()[0]) >= _DATABASE_SCHEMA_VERSION:
            connection.commit()
            return
        planner_columns = _table_columns(connection, "commentary_planner_runs")
        if "claim_candidates_json" not in planner_columns:
            connection.execute(
                """
                ALTER TABLE commentary_planner_runs
                ADD COLUMN claim_candidates_json TEXT NOT NULL DEFAULT '[]'
                """
            )
        _migrate_commentary_admission_tables(connection)
        columns = _table_columns(connection, "position_review_feedback")
        if "coaching_status" not in columns:
            connection.execute(
                """
                ALTER TABLE position_review_feedback
                ADD COLUMN coaching_status TEXT NOT NULL DEFAULT 'not_shown'
                    CHECK (
                        coaching_status IN (
                            'not_shown', 'loading', 'accepted', 'fallback',
                            'unavailable', 'disabled'
                        )
                    )
                """
            )
        if "commentary_run_id" not in columns:
            connection.execute(
                """
                ALTER TABLE position_review_feedback
                ADD COLUMN commentary_run_id TEXT REFERENCES commentary_planner_runs(id)
                """
            )
        _execute_sql_script(connection, _COMMENTARY_FEEDBACK_TRIGGERS)
        connection.execute(f"PRAGMA user_version = {_DATABASE_SCHEMA_VERSION}")
        connection.commit()
    except Exception:
        connection.rollback()
        raise


def _migrate_commentary_admission_tables(connection: sqlite3.Connection) -> None:
    attempt_columns = _table_columns(connection, "commentary_planner_attempts")
    if "feedback_id" in attempt_columns:
        connection.execute("DROP INDEX IF EXISTS idx_commentary_attempts_feedback")
        connection.execute("DROP INDEX IF EXISTS idx_commentary_attempts_review")
        connection.execute("DROP INDEX IF EXISTS idx_commentary_attempts_admitted")
        _execute_sql_script(
            connection,
            """
            ALTER TABLE commentary_planner_attempts RENAME TO legacy_commentary_attempts;
            CREATE TABLE commentary_planner_attempts (
                id TEXT PRIMARY KEY,
                review_id TEXT NOT NULL REFERENCES position_review_runs(id),
                admitted_at TEXT NOT NULL
            );
            INSERT INTO commentary_planner_attempts (id, review_id, admitted_at)
            SELECT id, review_id, admitted_at FROM legacy_commentary_attempts;
            DROP TABLE legacy_commentary_attempts;
            """,
        )

    reservation_columns = _table_columns(connection, "commentary_planner_reservations")
    connection.execute("DROP INDEX IF EXISTS idx_commentary_reservations_lease")
    connection.execute("DROP INDEX IF EXISTS idx_commentary_reservations_reserved")
    if "feedback_id" in reservation_columns:
        connection.execute("DROP INDEX IF EXISTS idx_commentary_reservations_feedback")
        connection.execute("DROP INDEX IF EXISTS idx_commentary_reservations_expiry")
        connection.execute("DROP INDEX IF EXISTS idx_commentary_reservations_created")
        _execute_sql_script(
            connection,
            """
            ALTER TABLE commentary_planner_reservations
            RENAME TO legacy_commentary_reservations;
            CREATE TABLE commentary_planner_reservations (
                review_id TEXT PRIMARY KEY REFERENCES position_review_runs(id),
                reservation_id TEXT NOT NULL UNIQUE,
                reserved_at TEXT NOT NULL,
                lease_expires_at TEXT NOT NULL
            );
            INSERT INTO commentary_planner_reservations (
                review_id, reservation_id, reserved_at, lease_expires_at
            )
            SELECT review_id, reservation_id, reserved_at, lease_expires_at
            FROM legacy_commentary_reservations;
            DROP TABLE legacy_commentary_reservations;
            """,
        )

    _execute_sql_script(
        connection,
        """
        CREATE INDEX IF NOT EXISTS idx_commentary_attempts_review
        ON commentary_planner_attempts(review_id);
        CREATE INDEX IF NOT EXISTS idx_commentary_attempts_admitted
        ON commentary_planner_attempts(admitted_at);
        CREATE INDEX IF NOT EXISTS idx_commentary_reservations_expiry
        ON commentary_planner_reservations(lease_expires_at);
        CREATE INDEX IF NOT EXISTS idx_commentary_reservations_created
        ON commentary_planner_reservations(reserved_at);
        """,
    )


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    if not table_name.isidentifier():
        raise ValueError("SQLite table name must be an identifier")
    return {
        str(row["name"])
        for row in connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    }


def _execute_sql_script(connection: sqlite3.Connection, script: str) -> None:
    statement = ""
    for line in script.splitlines():
        statement += f"{line}\n"
        if sqlite3.complete_statement(statement):
            connection.execute(statement)
            statement = ""
    if statement.strip():
        raise ValueError("Incomplete SQLite migration statement")


def _migrate_scans_table(connection: sqlite3.Connection) -> None:
    columns = _table_columns(connection, "scans")
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


_COMMENTARY_FEEDBACK_TRIGGERS = """
CREATE TRIGGER IF NOT EXISTS validate_review_feedback_coaching_insert
BEFORE INSERT ON position_review_feedback
WHEN (
    (NEW.coaching_status IN ('accepted', 'fallback')) != (NEW.commentary_run_id IS NOT NULL)
    OR (
        NEW.commentary_run_id IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM commentary_planner_runs planner
            WHERE planner.id = NEW.commentary_run_id
              AND planner.review_id = NEW.review_id
              AND planner.status = NEW.coaching_status
        )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'invalid coaching feedback snapshot');
END;

CREATE TRIGGER IF NOT EXISTS validate_review_feedback_coaching_update
BEFORE UPDATE OF review_id, coaching_status, commentary_run_id ON position_review_feedback
WHEN (
    (NEW.coaching_status IN ('accepted', 'fallback')) != (NEW.commentary_run_id IS NOT NULL)
    OR (
        NEW.commentary_run_id IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM commentary_planner_runs planner
            WHERE planner.id = NEW.commentary_run_id
              AND planner.review_id = NEW.review_id
              AND planner.status = NEW.coaching_status
        )
    )
)
BEGIN
    SELECT RAISE(ABORT, 'invalid coaching feedback snapshot');
END;
"""


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

CREATE TABLE IF NOT EXISTS feedback_adjudications (
    id TEXT PRIMARY KEY,
    feedback_id TEXT NOT NULL REFERENCES feedback_events(id),
    created_at TEXT NOT NULL,
    corrected_labels_json TEXT NOT NULL,
    corrected_fen TEXT NOT NULL,
    changed_squares INTEGER NOT NULL,
    reason TEXT NOT NULL
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

CREATE TABLE IF NOT EXISTS learning_cycles (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    state TEXT NOT NULL
        CHECK (state IN ('training', 'benchmarking', 'shadowing', 'promoted', 'rejected')),
    base_model_version TEXT NOT NULL REFERENCES model_versions(version),
    candidate_model_version TEXT REFERENCES model_versions(version),
    shadow_started_at TEXT,
    shadow_target_boards INTEGER NOT NULL,
    decision_reason TEXT,
    metrics_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learning_cycle_feedback (
    cycle_id TEXT NOT NULL REFERENCES learning_cycles(id),
    feedback_id TEXT NOT NULL REFERENCES feedback_events(id),
    role TEXT NOT NULL CHECK (role IN ('replay', 'batch')),
    PRIMARY KEY (cycle_id, feedback_id)
);

CREATE TABLE IF NOT EXISTS feedback_learning_pool (
    feedback_id TEXT PRIMARY KEY REFERENCES feedback_events(id),
    state TEXT NOT NULL CHECK (state IN ('accepted', 'quarantined')),
    cycle_id TEXT NOT NULL REFERENCES learning_cycles(id),
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS position_review_runs (
    id TEXT PRIMARY KEY,
    feedback_id TEXT NOT NULL REFERENCES feedback_events(id),
    created_at TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    engine TEXT NOT NULL,
    request_json TEXT NOT NULL,
    response_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS commentary_planner_runs (
    id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL UNIQUE REFERENCES position_review_runs(id),
    created_at TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('accepted', 'fallback')),
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    planner_version TEXT NOT NULL,
    request_json TEXT NOT NULL,
    raw_output TEXT,
    accepted_claims_json TEXT NOT NULL,
    claim_candidates_json TEXT NOT NULL,
    response_json TEXT NOT NULL,
    latency_ms INTEGER NOT NULL CHECK (latency_ms >= 0),
    input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
    output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
    error_code TEXT,
    provider_called INTEGER NOT NULL DEFAULT 1 CHECK (provider_called IN (0, 1))
);

CREATE TABLE IF NOT EXISTS commentary_planner_attempts (
    id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL REFERENCES position_review_runs(id),
    admitted_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_commentary_attempts_review
ON commentary_planner_attempts(review_id);

CREATE TABLE IF NOT EXISTS commentary_planner_reservations (
    review_id TEXT PRIMARY KEY REFERENCES position_review_runs(id),
    reservation_id TEXT NOT NULL UNIQUE,
    reserved_at TEXT NOT NULL,
    lease_expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS position_review_feedback (
    id TEXT PRIMARY KEY,
    review_id TEXT NOT NULL REFERENCES position_review_runs(id),
    created_at TEXT NOT NULL,
    rating TEXT NOT NULL CHECK (rating IN ('helpful', 'unhelpful')),
    reason TEXT NOT NULL CHECK (
        reason IN (
            'correct', 'incorrect_chess', 'irrelevant_topic', 'unclear',
            'equivalent_move_rejected', 'too_verbose', 'missing_detail', 'other'
        )
    ),
    detail TEXT,
    coaching_status TEXT NOT NULL DEFAULT 'not_shown' CHECK (
        coaching_status IN (
            'not_shown', 'loading', 'accepted', 'fallback', 'unavailable', 'disabled'
        )
    ),
    commentary_run_id TEXT REFERENCES commentary_planner_runs(id),
    CHECK (
        (coaching_status IN ('accepted', 'fallback') AND commentary_run_id IS NOT NULL)
        OR (coaching_status NOT IN ('accepted', 'fallback') AND commentary_run_id IS NULL)
    )
);

CREATE TABLE IF NOT EXISTS position_review_feedback_adjudications (
    id TEXT PRIMARY KEY,
    review_feedback_id TEXT NOT NULL REFERENCES position_review_feedback(id),
    created_at TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    disposition TEXT NOT NULL CHECK (
        disposition IN ('confirmed_issue', 'rejected', 'duplicate', 'approved_fix')
    ),
    notes TEXT NOT NULL,
    regression_fixture TEXT,
    CHECK (
        disposition != 'approved_fix'
        OR (regression_fixture IS NOT NULL AND length(trim(regression_fixture)) > 0)
    )
);

CREATE TABLE IF NOT EXISTS shadow_evaluations (
    cycle_id TEXT NOT NULL REFERENCES learning_cycles(id),
    feedback_id TEXT NOT NULL REFERENCES feedback_events(id),
    evaluated_at TEXT NOT NULL,
    perceptual_hash TEXT NOT NULL,
    candidate_labels_json TEXT NOT NULL,
    active_square_errors INTEGER NOT NULL,
    candidate_square_errors INTEGER NOT NULL,
    active_non_empty_errors INTEGER NOT NULL,
    candidate_non_empty_errors INTEGER NOT NULL,
    active_board_exact INTEGER NOT NULL CHECK (active_board_exact IN (0, 1)),
    candidate_board_exact INTEGER NOT NULL CHECK (candidate_board_exact IN (0, 1)),
    PRIMARY KEY (cycle_id, feedback_id)
);

CREATE INDEX IF NOT EXISTS idx_feedback_training
ON feedback_events(consent_training, created_at);

CREATE INDEX IF NOT EXISTS idx_feedback_adjudications_latest
ON feedback_adjudications(feedback_id, created_at, id);

CREATE INDEX IF NOT EXISTS idx_learning_cycles_active
ON learning_cycles(state, created_at);

CREATE INDEX IF NOT EXISTS idx_position_review_feedback_id
ON position_review_runs(feedback_id, created_at);

CREATE INDEX IF NOT EXISTS idx_position_review_ratings
ON position_review_feedback(review_id, created_at);

CREATE INDEX IF NOT EXISTS idx_commentary_planner_created
ON commentary_planner_runs(created_at);

CREATE INDEX IF NOT EXISTS idx_commentary_attempts_admitted
ON commentary_planner_attempts(admitted_at);

CREATE INDEX IF NOT EXISTS idx_commentary_reservations_expiry
ON commentary_planner_reservations(lease_expires_at);

CREATE INDEX IF NOT EXISTS idx_commentary_reservations_created
ON commentary_planner_reservations(reserved_at);

CREATE INDEX IF NOT EXISTS idx_position_review_adjudications
ON position_review_feedback_adjudications(review_feedback_id, created_at);
"""
