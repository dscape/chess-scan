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

from chess_scan.board import build_full_fen, fen_warnings, lichess_analysis_url, validate_full_fen
from chess_scan.classifier import BoardPrediction, ModelManager
from chess_scan.config import Settings
from chess_scan.database import Database
from chess_scan.geometry import (
    DETECTION_MAX_DIMENSION,
    board_grid_fits,
    detect_board_corners,
    order_corners,
    project_board_grid,
    rectify_board,
)
from chess_scan.image_io import decode_uploaded_image, write_jpeg
from chess_scan.schemas import (
    BoardDetectionResponse,
    ConfirmRequest,
    ConfirmResponse,
    LearningStatusResponse,
    ScanResponse,
)

logger = logging.getLogger(__name__)
_CLEANUP_RETRY_SECONDS = 60 * 60


@dataclass(frozen=True, slots=True)
class CleanupResult:
    removed: int
    backlog: bool


class ScannerService:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        models: ModelManager,
    ) -> None:
        self.settings = settings
        self.database = database
        self.models = models

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

    def scan(self, file_bytes: bytes) -> ScanResponse:
        source = decode_uploaded_image(
            file_bytes,
            max_dimension=self.settings.max_image_dimension,
        )
        source_height, source_width = source.shape[:2]
        detection = detect_board_corners(source)
        if detection.method == "manual_adjustment_needed":
            raise ValueError(
                "No complete, aligned 8x8 chess board was found. "
                "Retake the photo with all four board corners visible."
            )
        corners = [[float(x), float(y)] for x, y in detection.corners]
        rectified = rectify_board(source, corners)
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
                corners=corners,
                detection_method=detection.method,
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
            corners=corners,
            detection_method=detection.method,
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
        if not board_grid_fits(source, ordered):
            raise ValueError(
                "The selected corners do not tightly align with a complete 8x8 chess board"
            )
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
            warnings=fen_warnings(full_fen),
        )

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
        source_image_url=f"/api/scans/{scan_id}/source?v={cache_key}",
        rectified_image_url=f"/api/scans/{scan_id}/rectified?v={cache_key}",
    )


def _corners_array(corners: list[list[float]]) -> np.ndarray:
    return np.asarray(corners, dtype=np.float32)
