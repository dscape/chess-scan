"""HTTP request and response schemas."""

from __future__ import annotations

from typing import Literal

import chess
from pydantic import BaseModel, Field, field_validator

from chess_scan.board import SQUARE_COUNT, validate_labels
from chess_scan.review_topics import Course, TopicCapability


class ReprocessRequest(BaseModel):
    corners: list[list[float]] = Field(min_length=4, max_length=4)

    @field_validator("corners")
    @classmethod
    def validate_corners(cls, corners: list[list[float]]) -> list[list[float]]:
        if any(len(point) != 2 for point in corners):
            raise ValueError("Each corner must contain x and y coordinates")
        return corners


class ConfirmRequest(BaseModel):
    labels: list[int] = Field(min_length=SQUARE_COUNT, max_length=SQUARE_COUNT)
    orientation: Literal["white", "black"] = "white"
    side_to_move: Literal["w", "b"] = "w"
    castling: str = Field(default="-", min_length=1, max_length=4)
    en_passant: str = Field(default="-", max_length=2, pattern=r"^(?:-|[a-h][36])$")
    consent_training: bool = True
    client_session_id: str | None = Field(default=None, max_length=120)

    @field_validator("labels")
    @classmethod
    def validate_labels(cls, labels: list[int]) -> list[int]:
        validate_labels(labels)
        return labels

    @field_validator("castling")
    @classmethod
    def validate_castling(cls, castling: str) -> str:
        if castling == "-":
            return castling
        canonical = "".join(right for right in "KQkq" if right in castling)
        if castling != canonical or len(set(castling)) != len(castling):
            raise ValueError("Castling rights must be '-' or an ordered subset of KQkq")
        return castling


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
    prediction_revision: str
    source_image_url: str
    rectified_image_url: str


class ConfirmResponse(BaseModel):
    feedback_id: str
    full_fen: str
    lichess_url: str
    changed_squares: int
    warnings: list[str]


class ReviewPositionResponse(BaseModel):
    feedback_id: str
    full_fen: str
    orientation: Literal["white", "black"]
    changed_squares: int
    lichess_url: str


class EngineScore(BaseModel):
    kind: Literal["cp", "mate"]
    value: int
    bound: Literal["lower", "upper"] | None = None


class EngineLineInput(BaseModel):
    multipv: int = Field(ge=1, le=5)
    depth: int = Field(ge=1)
    score: EngineScore
    wdl: list[int] | None = Field(default=None, min_length=3, max_length=3)
    pv: list[str] = Field(min_length=1, max_length=24)

    @field_validator("pv")
    @classmethod
    def validate_pv(cls, moves: list[str]) -> list[str]:
        if any(not _is_uci_move(move) for move in moves):
            raise ValueError("Principal variation contains an invalid UCI move")
        return moves

    @field_validator("wdl")
    @classmethod
    def validate_wdl(cls, wdl: list[int] | None) -> list[int] | None:
        if wdl is not None and (
            any(value < 0 or value > 1000 for value in wdl) or sum(wdl) != 1000
        ):
            raise ValueError("WDL values must be non-negative and total 1000")
        return wdl


class PositionReviewRequest(BaseModel):
    fen: str = Field(min_length=1, max_length=120)
    study_level: int = Field(default=2, ge=1, le=6)
    mode: Literal["general", "mix", "thinking_ahead"] = "general"
    lines: list[EngineLineInput] = Field(max_length=5)


class ReviewMove(BaseModel):
    uci: str
    san: str


class ReviewLine(BaseModel):
    multipv: int
    depth: int
    score: EngineScore
    wdl: list[int] | None
    moves: list[ReviewMove]


class ReviewEvidence(BaseModel):
    kind: str
    summary: str
    squares: list[str] = Field(default_factory=list)
    moves: list[str] = Field(default_factory=list)


class TopicFindingResponse(BaseModel):
    topic_id: str
    topic: str
    level: int
    confidence: float
    evidence: list[ReviewEvidence]


class PositionReviewResponse(BaseModel):
    fen: str
    engine: str
    evaluation: str
    best_move: ReviewMove | None
    lines: list[ReviewLine]
    primary_finding: TopicFindingResponse | None
    findings: list[TopicFindingResponse]
    explanation: str
    verbalizer: Literal["mock"] = "mock"


class ReviewTopicResponse(BaseModel):
    id: str
    name: str
    level: int
    course: Course
    capability: TopicCapability


class ReviewTopicRegistryResponse(BaseModel):
    version: str
    topics: list[ReviewTopicResponse]


class LearningStatusResponse(BaseModel):
    confirmed_boards: int
    corrected_boards: int
    training_boards: int
    active_model: str
    learning_state: Literal["collecting", "training", "benchmarking", "shadowing"]
    learning_progress: int
    learning_target: int
    candidate_model: str | None


def _is_uci_move(move: str) -> bool:
    try:
        parsed = chess.Move.from_uci(move)
    except ValueError:
        return False
    return parsed != chess.Move.null() and parsed.drop is None
