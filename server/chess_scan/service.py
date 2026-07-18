"""Application-level scan, correction, and feedback workflow."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from chess_scan.board import build_full_fen, fen_warnings, lichess_analysis_url
from chess_scan.classifier import BoardPrediction, ModelManager
from chess_scan.config import Settings
from chess_scan.database import Database
from chess_scan.geometry import (
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
            max_dimension=min(self.settings.max_image_dimension, 1400),
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
        corners = [[float(x), float(y)] for x, y in detection.corners]
        rectified = rectify_board(source, corners)
        classifier = self.models.active()
        prediction = classifier.predict(rectified)

        scan_id = uuid.uuid4().hex
        source_path = self.settings.data_dir / "source-temp" / f"{scan_id}.jpg"
        rectified_path = self.settings.data_dir / "rectified" / f"{scan_id}.jpg"
        write_jpeg(source_path, source)
        write_jpeg(rectified_path, rectified)
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
        return _scan_response(
            scan_id=scan_id,
            source_width=source_width,
            source_height=source_height,
            corners=corners,
            detection_method=detection.method,
            prediction=prediction,
            model_version=classifier.version,
        )

    def reprocess(self, scan_id: str, corners: list[list[float]]) -> ScanResponse:
        scan = self.database.get_scan(scan_id)
        source_path = Path(scan["source_image_path"])
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError("The original photograph is no longer available for adjustment")

        ordered = order_corners(_corners_array(corners))
        ordered_corners = [[float(x), float(y)] for x, y in ordered]
        rectified = rectify_board(source, ordered)
        classifier = self.models.active()
        prediction = classifier.predict(rectified)
        rectified_path = Path(scan["rectified_image_path"])
        write_jpeg(rectified_path, rectified)
        self.database.update_scan_prediction(
            scan_id=scan_id,
            rectified_image_path=rectified_path,
            corners=ordered_corners,
            detection_method="manual",
            model_version=classifier.version,
            labels=prediction.labels,
            probabilities=prediction.probabilities,
            board_fen=prediction.board_fen,
        )
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
        scan = self.database.get_scan(scan_id)
        full_fen = build_full_fen(
            request.labels,
            orientation=request.orientation,
            side_to_move=request.side_to_move,
            castling=request.castling,
            en_passant=request.en_passant,
        )
        predicted_labels = [int(value) for value in json.loads(scan["predicted_labels_json"])]
        changed_squares = sum(
            predicted != final
            for predicted, final in zip(predicted_labels, request.labels, strict=True)
        )
        feedback_id = uuid.uuid4().hex
        self.database.confirm_scan(
            feedback_id=feedback_id,
            scan_id=scan_id,
            labels=request.labels,
            orientation=request.orientation,
            side_to_move=request.side_to_move,
            castling=request.castling,
            en_passant=request.en_passant,
            full_fen=full_fen,
            changed_squares=changed_squares,
            consent_training=request.consent_training,
            client_session_id=request.client_session_id,
        )

        source_path = Path(scan["source_image_path"])
        source_path.unlink(missing_ok=True)
        if not request.consent_training:
            Path(scan["rectified_image_path"]).unlink(missing_ok=True)
        return ConfirmResponse(
            feedback_id=feedback_id,
            full_fen=full_fen,
            lichess_url=lichess_analysis_url(full_fen, orientation=request.orientation),
            changed_squares=changed_squares,
            warnings=fen_warnings(full_fen),
        )

    def rectified_path(self, scan_id: str) -> Path:
        return Path(self.database.get_scan(scan_id)["rectified_image_path"])

    def learning_status(self) -> LearningStatusResponse:
        return LearningStatusResponse(**self.database.learning_status())

    def remove_stale_sources(self, *, older_than_seconds: int = 24 * 60 * 60) -> int:
        cutoff = time.time() - older_than_seconds
        removed = 0
        for path in (self.settings.data_dir / "source-temp").glob("*.jpg"):
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
                (self.settings.data_dir / "rectified" / path.name).unlink(missing_ok=True)
                removed += 1
        return removed


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
        rectified_image_url=f"/api/scans/{scan_id}/rectified?v={cache_key}",
    )


def _corners_array(corners: list[list[float]]) -> np.ndarray:
    return np.asarray(corners, dtype=np.float32)
