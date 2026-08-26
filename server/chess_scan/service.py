"""Application-level scan, correction, and feedback workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
import numpy as np

from chess_scan.board import (
    SQUARE_COUNT,
    build_full_fen,
    fen_warnings,
    labels_to_board_fen,
    lichess_analysis_url,
    validate_full_fen,
)
from chess_scan.capture_quality import (
    BoardCaptureQuality,
    board_capture_quality,
    compose_best_board,
)
from chess_scan.classifier import (
    BoardPrediction,
    ClassifierRoute,
    ModelManager,
    MultiViewPrediction,
)
from chess_scan.commentary_planner import (
    CommentaryCoach,
    CommentaryPlannerRun,
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
    board_corners_overlap,
    board_grid_fits,
    detect_board_corners,
    order_corners,
    project_board_grid,
    rectify_board,
    track_board_corners,
)
from chess_scan.image_io import decode_uploaded_image, write_jpeg
from chess_scan.review import build_position_attempt, build_position_review
from chess_scan.schemas import (
    BoardDetectionResponse,
    CaptureDiagnosticMetadata,
    CaptureDiagnosticResponse,
    CaptureReadinessContext,
    ConfirmRequest,
    ConfirmResponse,
    LearningStatusResponse,
    PositionAttemptRequest,
    PositionCoachingResponse,
    PositionReviewFeedbackRequest,
    PositionReviewFeedbackResponse,
    PositionReviewRequest,
    PositionReviewResponse,
    PositionStateRequest,
    PositionValidationResponse,
    PreviewPredictionResponse,
    ReviewAttemptResponse,
    ReviewPositionResponse,
    ScanResponse,
)

logger = logging.getLogger(__name__)
_CLEANUP_RETRY_SECONDS = 60 * 60
_CAPTURE_DIAGNOSTIC_CONTENT_TYPES = {"video/mp4", "video/webm"}
_CAPTURE_DIAGNOSTIC_MAX_BYTES = 8 * 1024 * 1024
_MIN_SUPPLEMENTAL_BOARD_OVERLAP = 0.45


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


@dataclass(frozen=True, slots=True)
class CapturedBoardFrame:
    capture_index: int
    payload: bytes
    source: np.ndarray
    corners: np.ndarray
    method: str
    rectified: np.ndarray
    quality: BoardCaptureQuality


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
        started = time.monotonic()
        source = decode_uploaded_image(
            file_bytes,
            max_dimension=min(self.settings.max_image_dimension, DETECTION_MAX_DIMENSION),
        )
        source_height, source_width = source.shape[:2]
        detection = detect_board_corners(source, fast=True)
        found = detection.method != "manual_adjustment_needed"
        corners = [[float(x), float(y)] for x, y in detection.corners] if found else []
        preview_prediction: PreviewPredictionResponse | None = None
        if found:
            rectified = rectify_board(source, detection.corners)
            quality = board_capture_quality(rectified)
            classifier = self.models.active()
            prediction = classifier.predict(rectified)
            preview_prediction = PreviewPredictionResponse(
                labels=prediction.labels,
                probabilities=prediction.probabilities,
                confidences=prediction.confidences,
                model_version=classifier.version,
                glare_score=quality.glare_score,
                glare_squares=list(quality.glare_squares),
                sharpness_score=quality.sharpness_score,
            )
        return BoardDetectionResponse(
            found=found,
            confidence=detection.confidence,
            method=detection.method,
            image_width=source_width,
            image_height=source_height,
            corners=corners,
            grid_points=project_board_grid(corners) if found else [],
            prediction=preview_prediction,
            processing_ms=max(0, round((time.monotonic() - started) * 1000)),
        )

    def scan(
        self,
        file_bytes: bytes,
        *,
        corners: list[list[float]] | None = None,
        detection_method: str | None = None,
        supplemental_file_bytes: list[bytes] | None = None,
        capture_readiness: CaptureReadinessContext | None = None,
    ) -> ScanResponse:
        source = decode_uploaded_image(
            file_bytes,
            max_dimension=self.settings.max_image_dimension,
        )
        if corners is None:
            if detection_method is not None:
                raise ValueError("A detection method cannot be used without captured corners")
            if capture_readiness is not None:
                raise ValueError("Capture readiness requires automatically detected corners")
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

        submitted_frames = [file_bytes, *(supplemental_file_bytes or [])[:3]]
        primary = _captured_board_frame(0, file_bytes, source, selected, method)
        captured_frames = [primary]
        rejected_frames: list[dict[str, object]] = []
        for capture_index, supplemental_bytes in enumerate(submitted_frames[1:], start=1):
            supplemental = decode_uploaded_image(
                supplemental_bytes,
                max_dimension=self.settings.max_image_dimension,
            )
            geometry, rejection_reason = _supplemental_frame_geometry(primary, supplemental)
            if geometry is None:
                rejected_frames.append({"capture_index": capture_index, "reason": rejection_reason})
                continue
            supplemental_corners, supplemental_method = geometry
            captured_frames.append(
                _captured_board_frame(
                    capture_index,
                    supplemental_bytes,
                    supplemental,
                    supplemental_corners,
                    supplemental_method,
                )
            )

        retained_frame = max(captured_frames, key=_capture_frame_utility)
        if len(captured_frames) == 1:
            rectified = retained_frame.rectified
            selected_frames = (0,) * SQUARE_COUNT
        else:
            rectified, selected_frames = compose_best_board(
                [frame.rectified for frame in captured_frames]
            )
        quality = board_capture_quality(rectified)
        source = retained_frame.source
        source_height, source_width = source.shape[:2]
        selected = retained_frame.corners
        selected_corners = [[float(x), float(y)] for x, y in selected]
        method = retained_frame.method
        classifier = self.models.active()
        inference = classifier.predict_multiview(
            rectified,
            [frame.rectified for frame in captured_frames],
        )
        prediction = inference.prediction
        prediction_evidence = inference.evidence()
        prediction_evidence["capture"] = _capture_evidence(
            submitted_frame_count=len(submitted_frames),
            captured_frames=captured_frames,
            rejected_frames=rejected_frames,
            selected_frames=selected_frames,
            capture_readiness=capture_readiness,
        )

        scan_id = uuid.uuid4().hex
        source_path = self.settings.data_dir / "source-temp" / f"{scan_id}.jpg"
        rectified_path = self.settings.data_dir / "rectified" / f"{scan_id}.jpg"
        write_jpeg(source_path, source)
        write_jpeg(rectified_path, rectified)
        try:
            self.database.create_scan(
                scan_id=scan_id,
                image_sha256=hashlib.sha256(retained_frame.payload).hexdigest(),
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
                prediction_evidence=prediction_evidence,
                glare_score=quality.glare_score,
                glare_squares=list(quality.glare_squares),
                sharpness_score=quality.sharpness_score,
                capture_frame_count=len(captured_frames),
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
            inference=inference,
            model_version=classifier.version,
            glare_score=quality.glare_score,
            glare_squares=list(quality.glare_squares),
            sharpness_score=quality.sharpness_score,
            capture_frame_count=len(captured_frames),
            submitted_frame_count=len(submitted_frames),
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
        inference = _stored_multiview_prediction(
            prediction,
            scan["prediction_evidence_json"],
        )
        return _scan_response(
            scan_id=scan_id,
            source_width=int(scan["source_width"]),
            source_height=int(scan["source_height"]),
            corners=json.loads(scan["corners_json"]),
            detection_method=str(scan["detection_method"]),
            inference=inference,
            model_version=str(scan["model_version"]),
            glare_score=float(scan["glare_score"]),
            glare_squares=[int(value) for value in json.loads(scan["glare_squares_json"])],
            sharpness_score=(
                float(scan["sharpness_score"]) if scan["sharpness_score"] is not None else None
            ),
            capture_frame_count=int(scan["capture_frame_count"]),
            submitted_frame_count=_stored_submitted_frame_count(
                scan["prediction_evidence_json"],
                fallback=int(scan["capture_frame_count"]),
            ),
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
        quality = board_capture_quality(rectified)
        classifier = self.models.active()
        inference = classifier.predict_multiview(rectified, [rectified])
        prediction = inference.prediction
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
                prediction_evidence=inference.evidence(),
                glare_score=quality.glare_score,
                glare_squares=list(quality.glare_squares),
                sharpness_score=quality.sharpness_score,
                capture_frame_count=1,
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
            inference=inference,
            model_version=classifier.version,
            glare_score=quality.glare_score,
            glare_squares=list(quality.glare_squares),
            sharpness_score=quality.sharpness_score,
            capture_frame_count=1,
            submitted_frame_count=1,
        )

    def validate_position(self, request: PositionStateRequest) -> PositionValidationResponse:
        full_fen = build_full_fen(
            request.labels,
            orientation=request.orientation,
            side_to_move=request.side_to_move,
            castling=request.castling,
            en_passant=request.en_passant,
        )
        problems = fen_warnings(full_fen)
        return PositionValidationResponse(
            valid=not problems,
            full_fen=full_fen,
            problems=problems,
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

    def save_capture_diagnostic(
        self,
        feedback_id: str,
        *,
        recording: bytes,
        content_type: str,
        metadata: CaptureDiagnosticMetadata,
        user_agent: str | None,
    ) -> CaptureDiagnosticResponse:
        if not self.settings.capture_diagnostics_enabled:
            raise ValueError("Camera diagnostics are disabled")
        normalized_content_type = content_type.partition(";")[0].strip().lower()
        if normalized_content_type not in _CAPTURE_DIAGNOSTIC_CONTENT_TYPES:
            raise ValueError("Camera diagnostics must be MP4 or WebM video")
        if not recording:
            raise ValueError("Camera diagnostic is empty")
        if len(recording) > _CAPTURE_DIAGNOSTIC_MAX_BYTES:
            raise ValueError("Camera diagnostic exceeds the 8 MB limit")
        if not _capture_diagnostic_matches_type(recording, normalized_content_type):
            raise ValueError("Camera diagnostic does not match its declared video type")

        diagnostic_id = uuid.uuid4().hex
        suffix = ".mp4" if normalized_content_type == "video/mp4" else ".webm"
        recording_path = self.settings.data_dir / "capture-diagnostics" / f"{diagnostic_id}{suffix}"
        expires_at = datetime.now(UTC) + timedelta(
            days=self.settings.capture_diagnostic_retention_days
        )
        recording_path.parent.mkdir(parents=True, exist_ok=True)
        recording_path.write_bytes(recording)
        try:
            self.database.save_capture_diagnostic(
                diagnostic_id=diagnostic_id,
                feedback_id=feedback_id,
                recording_path=recording_path,
                content_type=normalized_content_type,
                byte_count=len(recording),
                sha256=hashlib.sha256(recording).hexdigest(),
                metadata=metadata.model_dump(mode="json"),
                user_agent=user_agent,
                expires_at=expires_at.isoformat(),
            )
        except Exception:
            recording_path.unlink(missing_ok=True)
            raise
        return CaptureDiagnosticResponse(
            diagnostic_id=diagnostic_id,
            expires_at=expires_at.isoformat(),
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

    def compare_position_attempt(
        self,
        request: PositionAttemptRequest,
    ) -> ReviewAttemptResponse:
        return build_position_attempt(request)

    def create_position_review(self, request: PositionReviewRequest) -> PositionReviewResponse:
        if request.feedback_id is None:
            return build_position_review(request)

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

        if self.commentary_coach.selection_required(review):
            return PreparedPositionCoaching(review=review)

        run = self.commentary_coach.plan(review, provider_selection=False)
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

        diagnostic_count, diagnostic_backlog = self._remove_expired_capture_diagnostics(
            expires_before=now.isoformat(),
            deadline=deadline,
            batch_size=batch_size,
        )
        removed += diagnostic_count
        orphan_count, orphan_backlog = self._remove_orphaned_files(
            cutoff=cutoff.timestamp(),
            deadline=deadline,
        )
        removed += orphan_count
        database_backlog = self.database.has_cleanup_work(
            created_before=cutoff.isoformat(),
            retry_before=retry_before.isoformat(),
        )
        return CleanupResult(
            removed=removed,
            backlog=(database_backlog or diagnostic_backlog or orphan_backlog),
        )

    def _remove_expired_capture_diagnostics(
        self,
        *,
        expires_before: str,
        deadline: float,
        batch_size: int,
    ) -> tuple[int, bool]:
        removed = 0
        while time.monotonic() < deadline:
            rows = self.database.capture_diagnostics_for_cleanup(
                expires_before=expires_before,
                limit=batch_size,
            )
            if not rows:
                return removed, False
            completed: list[str] = []
            for row in rows:
                try:
                    Path(row["recording_path"]).unlink(missing_ok=True)
                except OSError:
                    logger.exception(
                        "Failed to remove capture diagnostic %s",
                        row["id"],
                    )
                    continue
                completed.append(str(row["id"]))
            self.database.delete_capture_diagnostics(completed)
            removed += len(completed)
            if len(rows) < batch_size or not completed:
                return removed, self.database.has_expired_capture_diagnostics(
                    expires_before=expires_before
                )
        return removed, self.database.has_expired_capture_diagnostics(expires_before=expires_before)

    def _remove_orphaned_files(self, *, cutoff: float, deadline: float) -> tuple[int, bool]:
        referenced = self.database.referenced_file_paths()
        removed = 0
        directories = (
            (self.settings.data_dir / "source-temp", ("*.jpg",)),
            (self.settings.data_dir / "rectified", ("*.jpg",)),
            (self.settings.data_dir / "capture-diagnostics", ("*.webm", "*.mp4")),
        )
        for directory, patterns in directories:
            for pattern in patterns:
                for path in directory.glob(pattern):
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


def _captured_board_frame(
    capture_index: int,
    payload: bytes,
    source: np.ndarray,
    corners: np.ndarray,
    method: str,
) -> CapturedBoardFrame:
    rectified = rectify_board(source, corners)
    return CapturedBoardFrame(
        capture_index=capture_index,
        payload=payload,
        source=source,
        corners=order_corners(corners),
        method=method,
        rectified=rectified,
        quality=board_capture_quality(rectified),
    )


def _supplemental_frame_geometry(
    primary: CapturedBoardFrame,
    supplemental: np.ndarray,
) -> tuple[tuple[np.ndarray, str] | None, str | None]:
    if supplemental.shape != primary.source.shape:
        return None, "image_dimensions_changed"

    tracked = track_board_corners(primary.source, supplemental, primary.corners)
    if tracked is not None and board_corners_overlap(primary.corners, tracked) >= (
        _MIN_SUPPLEMENTAL_BOARD_OVERLAP
    ):
        return (tracked, "tracked"), None

    detection = detect_board_corners(supplemental, fast=True)
    if detection.method == "manual_adjustment_needed":
        return None, "board_not_found"
    detected = order_corners(np.asarray(detection.corners, dtype=np.float32))
    if (
        primary.method != "manual_adjustment_needed"
        and board_corners_overlap(primary.corners, detected) < _MIN_SUPPLEMENTAL_BOARD_OVERLAP
    ):
        return None, "different_board"
    return (detected, detection.method), None


def _capture_evidence(
    *,
    submitted_frame_count: int,
    captured_frames: list[CapturedBoardFrame],
    rejected_frames: list[dict[str, object]],
    selected_frames: tuple[int, ...],
    capture_readiness: CaptureReadinessContext | None,
) -> dict[str, object]:
    selected_counts = Counter(selected_frames)
    evidence: dict[str, object] = {
        "schema_version": "capture-evidence-1",
        "submitted_frame_count": submitted_frame_count,
        "accepted_frame_count": len(captured_frames),
        "accepted_frames": [
            {
                "capture_index": frame.capture_index,
                "geometry_method": frame.method,
                "sharpness_score": frame.quality.sharpness_score,
                "glare_score": frame.quality.glare_score,
                "glare_squares": list(frame.quality.glare_squares),
                "selected_square_count": selected_counts[index],
            }
            for index, frame in enumerate(captured_frames)
        ],
        "rejected_frames": rejected_frames,
    }
    if capture_readiness is not None:
        evidence["readiness"] = capture_readiness.model_dump(mode="json")
    return evidence


def _stored_submitted_frame_count(evidence_json: str | None, *, fallback: int) -> int:
    if evidence_json is None:
        return fallback
    evidence = json.loads(evidence_json)
    capture = evidence.get("capture")
    if not isinstance(capture, dict):
        return fallback
    submitted = capture.get("submitted_frame_count")
    return int(submitted) if isinstance(submitted, int) and 1 <= submitted <= 4 else fallback


def _capture_frame_utility(frame: CapturedBoardFrame) -> float:
    return float(np.log1p(frame.quality.sharpness_score)) - 5.0 * frame.quality.glare_score


def _stored_multiview_prediction(
    prediction: BoardPrediction,
    evidence_json: str | None,
) -> MultiViewPrediction:
    if evidence_json is None:
        return MultiViewPrediction(
            prediction=prediction,
            raw_prediction=prediction,
            ensemble_prediction=prediction,
            frame_agreements=[1.0] * SQUARE_COUNT,
            view_agreements=[1.0] * SQUARE_COUNT,
            frame_labels=[prediction.labels],
            frame_probabilities=[prediction.probabilities],
            view_labels=[prediction.labels],
            inference_view_count=1,
            method="single-view-v1",
        )
    evidence = json.loads(evidence_json)
    raw_probabilities = [
        [float(value) for value in square] for square in evidence["raw_probabilities"]
    ]
    raw_labels = [int(value) for value in evidence["raw_labels"]]
    raw_prediction = BoardPrediction(
        labels=raw_labels,
        probabilities=raw_probabilities,
        confidences=[max(square) for square in raw_probabilities],
        board_fen=labels_to_board_fen(raw_labels),
    )
    if "ensemble_probabilities" in evidence:
        ensemble_probabilities = [
            [float(value) for value in square] for square in evidence["ensemble_probabilities"]
        ]
        ensemble_labels = [int(value) for value in evidence["ensemble_labels"]]
        ensemble_prediction = BoardPrediction(
            labels=ensemble_labels,
            probabilities=ensemble_probabilities,
            confidences=[max(square) for square in ensemble_probabilities],
            board_fen=labels_to_board_fen(ensemble_labels),
        )
    else:
        ensemble_prediction = (
            prediction if evidence["method"] == "multi-view-mean-v1" else raw_prediction
        )
    return MultiViewPrediction(
        prediction=prediction,
        raw_prediction=raw_prediction,
        ensemble_prediction=ensemble_prediction,
        frame_agreements=[float(value) for value in evidence["frame_agreements"]],
        view_agreements=[float(value) for value in evidence["view_agreements"]],
        frame_labels=[[int(value) for value in labels] for labels in evidence["frame_labels"]],
        frame_probabilities=[
            [[float(value) for value in square] for square in probabilities]
            for probabilities in evidence.get("frame_probabilities", [])
        ],
        view_labels=[[int(value) for value in labels] for labels in evidence["view_labels"]],
        inference_view_count=int(evidence["inference_view_count"]),
        method=str(evidence["method"]),
        classifier_route=_stored_classifier_route(evidence),
        classifier_route_score=_stored_classifier_route_score(evidence),
    )


def _stored_classifier_route(evidence: dict[str, object]) -> ClassifierRoute:
    return (
        "print-specialist" if evidence.get("classifier_route") == "print-specialist" else "general"
    )


def _stored_classifier_route_score(evidence: dict[str, object]) -> float | None:
    score = evidence.get("classifier_route_score")
    if isinstance(score, bool) or not isinstance(score, int | float):
        return None
    return float(score)


def _scan_response(
    *,
    scan_id: str,
    source_width: int,
    source_height: int,
    corners: list[list[float]],
    detection_method: str,
    inference: MultiViewPrediction,
    model_version: str,
    glare_score: float,
    glare_squares: list[int],
    sharpness_score: float | None,
    capture_frame_count: int,
    submitted_frame_count: int,
) -> ScanResponse:
    cache_key = uuid.uuid4().hex[:8]
    prediction = inference.prediction
    raw_prediction = inference.raw_prediction
    return ScanResponse(
        scan_id=scan_id,
        source_width=source_width,
        source_height=source_height,
        corners=corners,
        detection_method=detection_method,
        labels=prediction.labels,
        probabilities=prediction.probabilities,
        confidences=prediction.confidences,
        raw_labels=raw_prediction.labels,
        raw_probabilities=raw_prediction.probabilities,
        raw_confidences=raw_prediction.confidences,
        frame_agreements=inference.frame_agreements,
        view_agreements=inference.view_agreements,
        prediction_method=inference.method,
        classifier_route=inference.classifier_route,
        inference_view_count=inference.inference_view_count,
        board_fen=prediction.board_fen,
        model_version=model_version,
        glare_score=glare_score,
        glare_squares=glare_squares,
        sharpness_score=sharpness_score,
        capture_frame_count=capture_frame_count,
        submitted_frame_count=submitted_frame_count,
        prediction_revision=_prediction_revision(
            corners=corners,
            detection_method=detection_method,
            inference=inference,
            model_version=model_version,
        ),
        source_image_url=f"/api/scans/{scan_id}/source?v={cache_key}",
        rectified_image_url=f"/api/scans/{scan_id}/rectified?v={cache_key}",
    )


def _prediction_revision(
    *,
    corners: list[list[float]],
    detection_method: str,
    inference: MultiViewPrediction,
    model_version: str,
) -> str:
    prediction = inference.prediction
    revision = {
        "corners": corners,
        "detection_method": detection_method,
        "labels": prediction.labels,
        "model_version": model_version,
        "probabilities": prediction.probabilities,
    }
    if inference.method != "single-view-v1":
        revision.update(
            {
                "prediction_method": inference.method,
                "raw_labels": inference.raw_prediction.labels,
            }
        )
    payload = json.dumps(
        revision,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _capture_diagnostic_matches_type(recording: bytes, content_type: str) -> bool:
    if content_type == "video/webm":
        return recording.startswith(b"\x1a\x45\xdf\xa3")
    if content_type == "video/mp4":
        return len(recording) >= 12 and recording[4:8] == b"ftyp"
    return False


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
