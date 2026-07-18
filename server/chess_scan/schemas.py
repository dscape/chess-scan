"""HTTP request and response schemas."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ReprocessRequest(BaseModel):
    corners: list[list[float]] = Field(min_length=4, max_length=4)

    @field_validator("corners")
    @classmethod
    def validate_corners(cls, corners: list[list[float]]) -> list[list[float]]:
        if any(len(point) != 2 for point in corners):
            raise ValueError("Each corner must contain x and y coordinates")
        return corners


class ConfirmRequest(BaseModel):
    labels: list[int] = Field(min_length=64, max_length=64)
    orientation: Literal["white", "black"] = "white"
    side_to_move: Literal["w", "b"] = "w"
    castling: str = "-"
    en_passant: str = "-"
    consent_training: bool = True
    client_session_id: str | None = Field(default=None, max_length=120)

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, labels: list[int]) -> list[int]:
        if any(label < 0 or label > 12 for label in labels):
            raise ValueError("Every label must be between 0 and 12")
        return labels


class BoardDetectionResponse(BaseModel):
    found: bool
    confidence: float
    method: str
    image_width: int
    image_height: int
    corners: list[list[float]]
    grid_points: list[list[float]]


class ScanResponse(BaseModel):
    scan_id: str
    source_width: int
    source_height: int
    corners: list[list[float]]
    detection_method: str
    labels: list[int]
    probabilities: list[list[float]]
    confidences: list[float]
    board_fen: str
    model_version: str
    rectified_image_url: str


class ConfirmResponse(BaseModel):
    feedback_id: str
    full_fen: str
    lichess_url: str
    changed_squares: int
    warnings: list[str]


class LearningStatusResponse(BaseModel):
    confirmed_boards: int
    corrected_boards: int
    training_boards: int
    active_model: str
