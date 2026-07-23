"""Application-level scan, correction, and feedback workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from chess_scan.board import build_full_fen, lichess_analysis_url, validate_full_fen
from chess_scan.classifier import BoardPrediction, ModelManager
from chess_scan.commentary_planner import (
    CommentaryCoach,
    CommentaryPlannerRun,
    eligible_commentary_lessons,
)
from chess_scan.config import Settings
from chess_scan.database import Database
from chess_scan.errors import (
    CommentaryBudgetExceededError,
    CommentaryBusyError,
    StoredDataIntegrityError,
)
from chess_scan.geometry import (
    DETECTION_MAX_DIMENSION,
    board_grid_fits,
    detect_board_corners,
    order_corners,
    project_board_grid,
    rectify_board,
)
from chess_scan.image_io import decode_uploaded_image, write_jpeg
from chess_scan.review import build_position_review
from chess_scan.schemas import (
    BoardDetectionResponse,
    ConfirmRequest,
    ConfirmResponse,
    LearningStatusResponse,
    PositionCoachingResponse,
    PositionReviewFeedbackRequest,
    PositionReviewFeedbackResponse,
    PositionReviewRequest,
    PositionReviewResponse,
    ReviewPositionResponse,
    ScanResponse,
)

logger = logging.getLogger(__name__)
_CLEANUP_RETRY_SECONDS = 60 * 60


@dataclass(frozen=True, slots=True)
class CleanupResult:
    removed: int
    backlog: bool


@dataclass(frozen=True, slots=True)
class ImmediatePositionCoaching:
    response: PositionCoachingResponse


@dataclass(frozen=True, slots=True)
class PreparedPositionCoaching:
    review: PositionReviewResponse


PositionCoachingPreflight = ImmediatePositionCoaching | PreparedPositionCoaching


class ScannerService:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        models: ModelManager,
        commentary_coach: CommentaryCoach | None = None,
    ) -> None:
        self.settings = settings
        self.database = database
        self.models = models
        self.commentary_coach = commentary_coach or CommentaryCoach.from_settings(settings)

    def close(self) -> None:
        self.commentary_coach.close()

    def detect(self, file_bytes: bytes) -> BoardDetectionResponse:
        source = decode_uploaded_image(
            file_bytes,
            max_dimension=min(self.settings.max_image_dimension, DETECTION_MAX_DIMENSION),
        )
        source_height, source_width = source.shape[:2]
        detection = detect_board_corners(source)
        found = detection.method != "manual_adjustment_needed"
        corners = [[float(x), float(y)] for x, y in detection.corners] if found else []
        return BoardDetectionResponse(
            found=found,
            confidence=detection.confidence,
            method=detection.method,
            image_width=source_width,
            image_height=source_height,
            corners=corners,
            grid_points=project_board_grid(corners) if found else [],
        )

    def scan(
        self,
        file_bytes: bytes,
        *,
        corners: list[list[float]] | None = None,
        detection_method: str | None = None,
    ) -> ScanResponse:
        source = decode_uploaded_image(
            file_bytes,
            max_dimension=self.settings.max_image_dimension,
        )
        source_height, source_width = source.shape[:2]
        if corners is None:
            if detection_method is not None:
                raise ValueError("A detection method cannot be used without captured corners")
            detection = detect_board_corners(source)
            selected = order_corners(np.asarray(detection.corners, dtype=np.float32))
            method = detection.method
        else:
            if detection_method not in {"checkerboard", "contour"}:
                raise ValueError("Captured corners must come from an automatic board detection")
            selected = order_corners(_corners_array(corners))
            if not _corners_are_inside_image(selected, source.shape):
                raise ValueError("Captured board corners must stay inside the image")
            method = (
                detection_method
                if board_grid_fits(source, selected)
                else "manual_adjustment_needed"
            )

        selected_corners = [[float(x), float(y)] for x, y in selected]
        rectified = rectify_board(source, selected)
        classifier = self.models.active()
        prediction = classifier.predict(rectified)

        scan_id = uuid.uuid4().hex
        source_path = self.settings.data_dir / "source-temp" / f"{scan_id}.jpg"
        rectified_path = self.settings.data_dir / "rectified" / f"{scan_id}.jpg"
        write_jpeg(source_path, source)
        write_jpeg(rectified_path, rectified)
        try:
            self.database.create_scan(
                scan_id=scan_id,
                image_sha256=hashlib.sha256(file_bytes).hexdigest(),
                source_width=source_width,
                source_height=source_height,
                source_image_path=source_path,
                rectified_image_path=rectified_path,
                corners=selected_corners,
                detection_method=method,
                model_version=classifier.version,
                labels=prediction.labels,
                probabilities=prediction.probabilities,
                board_fen=prediction.board_fen,
            )
        except Exception:
            source_path.unlink(missing_ok=True)
            rectified_path.unlink(missing_ok=True)
            raise
        return _scan_response(
            scan_id=scan_id,
            source_width=source_width,
            source_height=source_height,
            corners=selected_corners,
            detection_method=method,
            prediction=prediction,
            model_version=classifier.version,
        )

    def get_scan(self, scan_id: str) -> ScanResponse:
        scan = self.database.scan_for_display(scan_id)
        probabilities = [
            [float(value) for value in square]
            for square in json.loads(scan["predicted_probabilities_json"])
        ]
        prediction = BoardPrediction(
            labels=[int(value) for value in json.loads(scan["predicted_labels_json"])],
            probabilities=probabilities,
            confidences=[max(square) for square in probabilities],
            board_fen=str(scan["predicted_board_fen"]),
        )
        return _scan_response(
            scan_id=scan_id,
            source_width=int(scan["source_width"]),
            source_height=int(scan["source_height"]),
            corners=json.loads(scan["corners_json"]),
            detection_method=str(scan["detection_method"]),
            prediction=prediction,
            model_version=str(scan["model_version"]),
        )

    def reprocess(self, scan_id: str, corners: list[list[float]]) -> ScanResponse:
        scan = self.database.scan_for_reprocessing(scan_id)
        source_path = Path(scan["source_image_path"])
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError("The original photograph is no longer available for adjustment")

        ordered = order_corners(_corners_array(corners))
        if not _corners_are_inside_image(ordered, source.shape):
            raise ValueError("The selected corners must stay inside the photograph")
        ordered_corners = [[float(x), float(y)] for x, y in ordered]
        rectified = rectify_board(source, ordered)
        classifier = self.models.active()
        prediction = classifier.predict(rectified)
        rectified_path = Path(scan["rectified_image_path"]).with_name(
            f"{scan_id}-{uuid.uuid4().hex[:8]}.jpg"
        )
        write_jpeg(rectified_path, rectified)
        try:
            displaced_path = self.database.update_scan_prediction(
                scan_id=scan_id,
                rectified_image_path=rectified_path,
                corners=ordered_corners,
                detection_method="manual",
                model_version=classifier.version,
                labels=prediction.labels,
                probabilities=prediction.probabilities,
                board_fen=prediction.board_fen,
            )
        except Exception:
            rectified_path.unlink(missing_ok=True)
            raise
        displaced_path.unlink(missing_ok=True)
        return _scan_response(
            scan_id=scan_id,
            source_width=int(scan["source_width"]),
            source_height=int(scan["source_height"]),
            corners=ordered_corners,
            detection_method="manual",
            prediction=prediction,
            model_version=classifier.version,
        )

    def confirm(self, scan_id: str, request: ConfirmRequest) -> ConfirmResponse:
        full_fen = build_full_fen(
            request.labels,
            orientation=request.orientation,
            side_to_move=request.side_to_move,
            castling=request.castling,
            en_passant=request.en_passant,
        )
        validate_full_fen(full_fen)
        feedback_id = uuid.uuid4().hex
        confirmed = self.database.confirm_scan(
            feedback_id=feedback_id,
            scan_id=scan_id,
            labels=request.labels,
            orientation=request.orientation,
            side_to_move=request.side_to_move,
            castling=request.castling,
            en_passant=request.en_passant,
            full_fen=full_fen,
            consent_training=request.consent_training,
            client_session_id=request.client_session_id,
        )

        try:
            Path(confirmed["source_image_path"]).unlink(missing_ok=True)
            if not request.consent_training:
                Path(confirmed["rectified_image_path"]).unlink(missing_ok=True)
        except OSError:
            logger.exception("Failed to remove confirmed files for scan %s", scan_id)
        else:
            self.database.complete_scan_cleanup([scan_id])
        return ConfirmResponse(
            feedback_id=feedback_id,
            full_fen=full_fen,
            lichess_url=lichess_analysis_url(full_fen, orientation=request.orientation),
            changed_squares=int(confirmed["changed_squares"]),
            warnings=[],
            coaching_available=self.commentary_coach.enabled,
        )

    def review_position(self, feedback_id: str) -> ReviewPositionResponse:
        review = self.database.review_position(feedback_id)
        try:
            orientation = str(review["orientation"])
            if orientation not in {"white", "black"}:
                raise ValueError(f"Invalid stored review orientation: {orientation}")
            full_fen = str(review["final_fen"])
            validate_full_fen(full_fen)
        except ValueError as exc:
            raise StoredDataIntegrityError(f"Stored review data is invalid: {feedback_id}") from exc
        return ReviewPositionResponse(
            feedback_id=str(review["feedback_id"]),
            full_fen=full_fen,
            orientation=orientation,
            changed_squares=int(review["changed_squares"]),
            lichess_url=lichess_analysis_url(full_fen, orientation=orientation),
            coaching_available=self.commentary_coach.enabled,
        )

    def create_position_review(self, request: PositionReviewRequest) -> PositionReviewResponse:
        if request.feedback_id is None:
            raise ValueError("A confirmed feedback ID is required")
        review_id = uuid.uuid4().hex
        response = build_position_review(request, review_id=review_id)
        self.database.save_position_review(
            review_id=review_id,
            feedback_id=request.feedback_id,
            fen=response.fen,
            schema_version=response.schema_version,
            engine=response.engine,
            request=request.model_dump(mode="json"),
            response=response.model_dump(mode="json"),
        )
        return response

    def get_position_review(self, review_id: str) -> PositionReviewResponse:
        try:
            return PositionReviewResponse.model_validate(
                self.database.position_review_run(review_id)
            )
        except ValueError as exc:
            raise StoredDataIntegrityError(
                f"Stored position review is invalid: {review_id}"
            ) from exc

    def preflight_position_coaching(
        self,
        review_id: str,
    ) -> PositionCoachingPreflight:
        review = self.get_position_review(review_id)
        if not self.commentary_coach.enabled:
            return ImmediatePositionCoaching(
                response=self.commentary_coach.plan(review).response,
            )

        stored = self._stored_position_coaching(review)
        if stored is not None:
            return ImmediatePositionCoaching(response=stored)

        lessons = eligible_commentary_lessons(review)
        if lessons:
            return PreparedPositionCoaching(review=review)

        run = self.commentary_coach.plan(review)
        if not isinstance(run, CommentaryPlannerRun):
            raise RuntimeError("Enabled coaching returned a disabled result")
        response = self._persist_commentary_run(review, run)
        return ImmediatePositionCoaching(response=response)

    def create_position_coaching(
        self,
        review_id: str,
        *,
        preflight: PreparedPositionCoaching | None = None,
    ) -> PositionCoachingResponse:
        prepared = preflight or self.preflight_position_coaching(review_id)
        if isinstance(prepared, ImmediatePositionCoaching):
            return prepared.response

        review = prepared.review
        if review.review_id != review_id:
            raise ValueError("Prepared coaching references a different review")
        run_id = uuid.uuid4().hex
        reservation = self.database.reserve_commentary_planner_run(
            review_id=review_id,
            reservation_id=run_id,
            lease_seconds=self.settings.commentary_planner_timeout_seconds + 35,
            max_runs_per_feedback=self.settings.commentary_planner_max_runs_per_feedback,
            max_runs_per_hour=self.settings.commentary_planner_max_runs_per_hour,
            max_concurrent=self.settings.commentary_planner_max_concurrent,
        )
        if reservation == "stored":
            stored = self._stored_position_coaching(review)
            if stored is None:
                raise CommentaryBusyError("Coaching result is being committed")
            return stored
        if reservation == "busy":
            raise CommentaryBusyError("Coaching is already being prepared")
        if reservation == "global_busy":
            raise CommentaryBusyError("Coaching is busy; try again shortly")
        if reservation == "budget_exhausted":
            raise CommentaryBudgetExceededError(
                "This position has reached its external coaching limit"
            )
        if reservation == "global_budget_exhausted":
            raise CommentaryBudgetExceededError(
                "The external coaching budget is temporarily exhausted"
            )
        if reservation != "reserved":
            raise RuntimeError(f"Unknown commentary reservation outcome: {reservation}")

        run: CommentaryPlannerRun | None = None
        try:
            planned = self.commentary_coach.plan(
                review,
                run_id=run_id,
            )
            if not isinstance(planned, CommentaryPlannerRun):
                raise RuntimeError("Enabled coaching returned a disabled result")
            run = planned
            return self._persist_commentary_run(
                review,
                run,
                reservation_id=run_id,
            )
        except Exception:
            if run is None or run.provider_completion is None:
                self.database.release_commentary_planner_reservation(
                    review_id=review_id,
                    reservation_id=run_id,
                )
            raise

    def _persist_commentary_run(
        self,
        review: PositionReviewResponse,
        run: CommentaryPlannerRun,
        *,
        reservation_id: str | None = None,
    ) -> PositionCoachingResponse:
        if review.review_id is None:
            raise ValueError("Stored review ID is required for coaching")
        response = run.response
        if response.run_id is None:
            raise RuntimeError("Persisted coaching requires a run ID")
        try:
            inserted = self.database.save_commentary_planner_run(
                run.record,
                reservation_id=reservation_id,
                release_reservation=run.provider_completion is None,
            )
        finally:
            if reservation_id is not None and run.provider_completion is not None:
                run.provider_completion.add_done_callback(
                    lambda _future: self._release_commentary_reservation(
                        review.review_id,
                        reservation_id,
                    )
                )
        if inserted:
            return response
        if reservation_id is not None and run.provider_completion is None:
            self.database.release_commentary_planner_reservation(
                review_id=review.review_id,
                reservation_id=reservation_id,
            )
        stored = self._stored_position_coaching(review)
        if stored is None:
            raise CommentaryBusyError("Coaching result is being committed")
        return stored

    def _release_commentary_reservation(
        self,
        review_id: str,
        reservation_id: str,
    ) -> None:
        try:
            self.database.release_commentary_planner_reservation(
                review_id=review_id,
                reservation_id=reservation_id,
            )
        except Exception:
            logger.exception("Could not release commentary reservation %s", reservation_id)

    def _stored_position_coaching(
        self,
        review: PositionReviewResponse,
    ) -> PositionCoachingResponse | None:
        if review.review_id is None:
            raise ValueError("Stored review ID is required for coaching")
        try:
            return self.database.position_coaching(review.review_id, review=review)
        except ValueError as exc:
            raise StoredDataIntegrityError(
                f"Stored position coaching is invalid: {review.review_id}"
            ) from exc

    def add_position_review_feedback(
        self,
        review_id: str,
        request: PositionReviewFeedbackRequest,
    ) -> PositionReviewFeedbackResponse:
        feedback_id = uuid.uuid4().hex
        self.database.append_position_review_feedback(
            feedback_id=feedback_id,
            review_id=review_id,
            rating=request.rating,
            reason=request.reason,
            detail=request.detail,
            coaching_status=request.coaching_status,
            commentary_run_id=request.commentary_run_id,
        )
        return PositionReviewFeedbackResponse(feedback_id=feedback_id)

    def source_path(self, scan_id: str) -> Path:
        return self.database.source_image_path(scan_id)

    def rectified_path(self, scan_id: str) -> Path:
        return self.database.rectified_image_path(scan_id)

    def learning_status(self) -> LearningStatusResponse:
        return LearningStatusResponse(**self.database.learning_status())

    def remove_stale_sources(
        self,
        *,
        older_than_seconds: int = 24 * 60 * 60,
        max_seconds: float = 5.0,
        batch_size: int = 200,
    ) -> CleanupResult:
        now = datetime.now(UTC)
        cutoff = now - timedelta(seconds=older_than_seconds)
        retry_before = now - timedelta(seconds=_CLEANUP_RETRY_SECONDS)
        deadline = time.monotonic() + max_seconds
        removed = 0

        while time.monotonic() < deadline:
            rows = self.database.scan_files_for_cleanup(
                created_before=cutoff.isoformat(),
                retry_before=retry_before.isoformat(),
                limit=batch_size,
            )
            if not rows:
                break
            completed: list[str] = []
            for row in rows:
                try:
                    Path(row["source_image_path"]).unlink(missing_ok=True)
                    if row["delete_rectified"]:
                        Path(row["rectified_image_path"]).unlink(missing_ok=True)
                except OSError:
                    logger.exception("Failed to remove retained files for scan %s", row["id"])
                    continue
                completed.append(str(row["id"]))
            self.database.complete_scan_cleanup(completed)
            removed += len(completed)

        orphan_count, orphan_backlog = self._remove_orphaned_files(
            cutoff=cutoff.timestamp(),
            deadline=deadline,
        )
        removed += orphan_count
        database_backlog = self.database.has_cleanup_work(
            created_before=cutoff.isoformat(),
            retry_before=retry_before.isoformat(),
        )
        return CleanupResult(removed=removed, backlog=database_backlog or orphan_backlog)

    def _remove_orphaned_files(self, *, cutoff: float, deadline: float) -> tuple[int, bool]:
        referenced = self.database.referenced_file_paths()
        removed = 0
        for directory in (
            self.settings.data_dir / "source-temp",
            self.settings.data_dir / "rectified",
        ):
            for path in directory.glob("*.jpg"):
                if time.monotonic() >= deadline:
                    return removed, True
                if path.resolve() in referenced:
                    continue
                try:
                    if path.stat().st_mtime >= cutoff:
                        continue
                    path.unlink(missing_ok=True)
                except FileNotFoundError:
                    continue
                except OSError:
                    logger.exception("Failed to remove orphaned scan file %s", path)
                    continue
                removed += 1
        return removed, False


def _scan_response(
    *,
    scan_id: str,
    source_width: int,
    source_height: int,
    corners: list[list[float]],
    detection_method: str,
    prediction: BoardPrediction,
    model_version: str,
) -> ScanResponse:
    cache_key = uuid.uuid4().hex[:8]
    return ScanResponse(
        scan_id=scan_id,
        source_width=source_width,
        source_height=source_height,
        corners=corners,
        detection_method=detection_method,
        labels=prediction.labels,
        probabilities=prediction.probabilities,
        confidences=prediction.confidences,
        board_fen=prediction.board_fen,
        model_version=model_version,
        prediction_revision=_prediction_revision(
            corners=corners,
            detection_method=detection_method,
            prediction=prediction,
            model_version=model_version,
        ),
        source_image_url=f"/api/scans/{scan_id}/source?v={cache_key}",
        rectified_image_url=f"/api/scans/{scan_id}/rectified?v={cache_key}",
    )


def _prediction_revision(
    *,
    corners: list[list[float]],
    detection_method: str,
    prediction: BoardPrediction,
    model_version: str,
) -> str:
    payload = json.dumps(
        {
            "corners": corners,
            "detection_method": detection_method,
            "labels": prediction.labels,
            "model_version": model_version,
            "probabilities": prediction.probabilities,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _corners_array(corners: list[list[float]]) -> np.ndarray:
    return np.asarray(corners, dtype=np.float32)


def _corners_are_inside_image(corners: np.ndarray, image_shape: tuple[int, ...]) -> bool:
    height, width = image_shape[:2]
    return bool(
        np.isfinite(corners).all()
        and (corners[:, 0] >= 0).all()
        and (corners[:, 0] <= width - 1).all()
        and (corners[:, 1] >= 0).all()
        and (corners[:, 1] <= height - 1).all()
    )
