"""HTTP request and response schemas."""

from __future__ import annotations

from typing import Literal

import chess
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from chess_scan.board import SQUARE_COUNT, validate_full_fen, validate_labels


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
    model_config = ConfigDict(extra="forbid")

    kind: Literal["cp", "mate"]
    value: int
    bound: Literal["lower", "upper"] | None = None

    @model_validator(mode="after")
    def validate_value(self) -> EngineScore:
        if self.kind == "mate" and (self.value == 0 or abs(self.value) > 1000):
            raise ValueError("Mate scores must be non-zero and at most 1000 moves")
        if self.kind == "cp" and abs(self.value) > 100_000:
            raise ValueError("Centipawn scores must be between -100000 and 100000")
        return self


class EngineLineInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(ge=1, le=3)
    depth: int = Field(ge=8, le=255)
    score: EngineScore
    wdl: list[int] = Field(min_length=3, max_length=3)
    pv: list[str] = Field(min_length=1, max_length=16)
    stable: bool

    @field_validator("pv")
    @classmethod
    def validate_pv(cls, moves: list[str]) -> list[str]:
        if any(not _is_uci_move(move) for move in moves):
            raise ValueError("Principal variation contains an invalid UCI move")
        return moves

    @field_validator("wdl")
    @classmethod
    def validate_wdl(cls, wdl: list[int]) -> list[int]:
        if any(value < 0 or value > 1000 for value in wdl) or sum(wdl) != 1000:
            raise ValueError("WDL values must be non-negative and total 1000")
        return wdl

    @model_validator(mode="after")
    def validate_mate_wdl(self) -> EngineLineInput:
        if self.score.kind == "mate":
            decisive_index = 0 if self.score.value > 0 else 2
            if self.wdl[decisive_index] != 1000:
                raise ValueError("Mate scores require a decisive WDL result")
        return self


class ReviewAttemptInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    move: str
    line: EngineLineInput

    @field_validator("move")
    @classmethod
    def validate_move(cls, move: str) -> str:
        if not _is_uci_move(move):
            raise ValueError("Attempt contains an invalid UCI move")
        return move

    @model_validator(mode="after")
    def validate_line(self) -> ReviewAttemptInput:
        if self.line.rank != 1:
            raise ValueError("Attempt analysis must contain rank 1")
        if self.line.pv[0] != self.move:
            raise ValueError("Attempt analysis must begin with the attempted move")
        return self


class ReviewAnalysisInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_pov: Literal["side_to_move"]
    lines: list[EngineLineInput] = Field(min_length=1, max_length=3)
    attempt: ReviewAttemptInput | None = None

    @model_validator(mode="after")
    def validate_lines(self) -> ReviewAnalysisInput:
        ranks = [line.rank for line in self.lines]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("Engine candidate ranks must be ordered and contiguous from 1")
        if any(not line.stable for line in self.lines):
            raise ValueError("Engine candidates must be stable")
        if any(line.score.bound is not None for line in self.lines):
            raise ValueError("Engine candidates must have exact scores")
        first_moves = [line.pv[0] for line in self.lines]
        if len(set(first_moves)) != len(first_moves):
            raise ValueError("Engine candidates must begin with distinct moves")
        if self.attempt is not None and not self.attempt.line.stable:
            raise ValueError("Attempt analysis must be stable")
        if self.attempt is not None and self.attempt.line.score.bound is not None:
            raise ValueError("Attempt analysis must have an exact score")
        return self


class PositionReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fen: str = Field(min_length=1, max_length=120)
    feedback_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    analysis: ReviewAnalysisInput | None = None


class ReviewMove(BaseModel):
    uci: str
    san: str = Field(min_length=1, max_length=20)

    @field_validator("uci")
    @classmethod
    def validate_uci(cls, uci: str) -> str:
        if not _is_uci_move(uci):
            raise ValueError("Review move contains invalid UCI")
        return uci


class ReviewArrow(BaseModel):
    from_square: str
    to_square: str
    kind: Literal["move", "idea"] = "idea"

    @field_validator("from_square", "to_square")
    @classmethod
    def validate_square(cls, square: str) -> str:
        return _square_name(square)


class ReviewAnnotation(BaseModel):
    label: str
    text: str
    scope: Literal[
        "root",
        "best_line",
        "attempt_line",
        "attempt_refutation",
        "terminal",
    ] = "root"
    ply: int = Field(default=0, ge=0)
    squares: list[str] = Field(default_factory=list)
    arrows: list[ReviewArrow] = Field(default_factory=list)
    evidence_ids: list[str] = Field(min_length=1)


class PositionTopicResponse(BaseModel):
    id: str
    name: str


class ReviewPieceRef(BaseModel):
    color: Literal["white", "black"]
    piece: str
    square: str

    @field_validator("square")
    @classmethod
    def validate_square(cls, square: str) -> str:
        return _square_name(square)


class ReviewEvidenceResponse(BaseModel):
    id: str
    kind: str
    scope: Literal["best_line", "attempt_line", "attempt_refutation", "terminal"]
    proof: Literal["legal_geometry", "line_consequence", "direct_rule"]
    ply: int = Field(ge=0)
    actor: ReviewPieceRef | None
    targets: list[ReviewPieceRef]
    from_square: str | None = None
    to_square: str | None = None
    squares: list[str]
    moves: list[str]
    score: EngineScore | None = None
    wdl: list[int] | None = None
    expected_score_loss: float | None = Field(default=None, ge=0, le=1)
    centipawn_loss: int | None = Field(default=None, ge=0)
    lost_forced_mate: bool | None = None
    mate_delay: int | None = Field(default=None, ge=0)
    verdict: Literal["best", "excellent", "good", "inaccuracy", "mistake", "blunder"] | None = None

    @field_validator("from_square", "to_square")
    @classmethod
    def validate_optional_square(cls, square: str | None) -> str | None:
        return _square_name(square) if square is not None else None

    @field_validator("squares")
    @classmethod
    def validate_squares(cls, squares: list[str]) -> list[str]:
        return [_square_name(square) for square in squares]

    @field_validator("moves")
    @classmethod
    def validate_moves(cls, moves: list[str]) -> list[str]:
        if any(not _is_uci_move(move) for move in moves):
            raise ValueError("Review evidence contains an invalid UCI move")
        return moves

    @model_validator(mode="after")
    def validate_move_geometry(self) -> ReviewEvidenceResponse:
        if (self.from_square is None) != (self.to_square is None):
            raise ValueError("Review evidence move geometry requires both endpoint squares")
        return self


class ReviewFindingResponse(BaseModel):
    topic: PositionTopicResponse
    evidence_ids: list[str] = Field(min_length=1)


class ReviewLineResponse(BaseModel):
    role: Literal[
        "best_candidate",
        "alternative_candidate",
        "attempt_line",
        "attempt_refutation",
    ]
    rank: int = Field(ge=1, le=3)
    depth: int = Field(ge=8, le=255)
    score: EngineScore
    wdl: list[int] = Field(min_length=3, max_length=3)
    moves: list[ReviewMove] = Field(min_length=1, max_length=16)

    @model_validator(mode="after")
    def validate_engine_result(self) -> ReviewLineResponse:
        if sum(self.wdl) != 1000 or any(value < 0 or value > 1000 for value in self.wdl):
            raise ValueError("Review-line WDL values must total 1000")
        if self.score.bound is not None:
            raise ValueError("Review lines require exact scores")
        if self.score.kind == "mate":
            decisive_index = 0 if self.score.value > 0 else 2
            if self.wdl[decisive_index] != 1000:
                raise ValueError("Mate review lines require a decisive WDL result")
        return self


class ReviewAttemptResponse(BaseModel):
    move: ReviewMove
    verdict: Literal["best", "excellent", "good", "inaccuracy", "mistake", "blunder"]
    equivalent: bool
    expected_score_loss: float = Field(ge=0, le=1)
    centipawn_loss: int | None = Field(default=None, ge=0)
    lost_forced_mate: bool = False
    mate_delay: int | None = Field(default=None, ge=0)
    line: ReviewLineResponse


class PositionReviewResponse(BaseModel):
    schema_version: Literal["position-analysis-2"] = "position-analysis-2"
    review_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    fen: str
    engine: str
    evaluation: str
    score: EngineScore | None
    score_pov: Literal["side_to_move"] | None
    best_move: ReviewMove | None
    lines: list[ReviewLineResponse]
    attempt: ReviewAttemptResponse | None
    topic: PositionTopicResponse
    findings: list[ReviewFindingResponse]
    evidence: list[ReviewEvidenceResponse]
    hint: ReviewAnnotation
    explanation: list[ReviewAnnotation]

    @field_validator("fen")
    @classmethod
    def validate_fen(cls, fen: str) -> str:
        validate_full_fen(fen)
        return fen

    @model_validator(mode="after")
    def validate_evidence_contract(self) -> PositionReviewResponse:
        evidence_ids = [item.id for item in self.evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("Position review evidence IDs must be unique")
        known = set(evidence_ids)
        references = [
            *(evidence_id for finding in self.findings for evidence_id in finding.evidence_ids),
            *self.hint.evidence_ids,
            *(
                evidence_id
                for annotation in self.explanation
                for evidence_id in annotation.evidence_ids
            ),
        ]
        if not set(references) <= known:
            raise ValueError("Position review references unknown evidence")
        if self.score is None:
            if (
                self.score_pov is not None
                or self.best_move is not None
                or self.lines
                or self.attempt is not None
                or any(item.score is not None or item.wdl is not None for item in self.evidence)
            ):
                raise ValueError("Terminal reviews cannot contain engine candidates")
            return self
        if self.score_pov != "side_to_move" or self.best_move is None or not self.lines:
            raise ValueError("Playable reviews require a scored best line")
        ranks = [line.rank for line in self.lines]
        if ranks != list(range(1, len(ranks) + 1)):
            raise ValueError("Review candidate ranks must be contiguous from 1")
        if self.lines[0].role != "best_candidate" or any(
            line.role != "alternative_candidate" for line in self.lines[1:]
        ):
            raise ValueError("Review candidate roles do not match their ranks")
        best_line = self.lines[0]
        for line in self.lines:
            _validate_review_line(self.fen, line)
        if self.score != best_line.score or self.best_move != best_line.moves[0]:
            raise ValueError("Review best move or score does not match the first candidate")
        best_evidence = [item for item in self.evidence if item.kind == "engine_candidate"]
        if len(best_evidence) != 1:
            raise ValueError("Playable reviews require exactly one engine-candidate evidence item")
        _validate_engine_line_evidence(
            best_evidence[0],
            best_line,
            expected_scope="best_line",
        )

        attempt_evidence = [item for item in self.evidence if item.kind == "engine_comparison"]
        if self.attempt is None:
            if attempt_evidence:
                raise ValueError("Review without an attempt cannot contain attempt-engine evidence")
            return self
        expected_role = (
            "attempt_refutation"
            if self.attempt.verdict in {"mistake", "blunder"}
            else "attempt_line"
        )
        if (
            self.attempt.line.role != expected_role
            or self.attempt.line.moves[0] != self.attempt.move
        ):
            raise ValueError("Review attempt does not match its checked line or verdict")
        _validate_review_line(self.fen, self.attempt.line)
        if len(attempt_evidence) != 1:
            raise ValueError("Review attempts require exactly one attempt-engine evidence item")
        evidence = attempt_evidence[0]
        _validate_engine_line_evidence(
            evidence,
            self.attempt.line,
            expected_scope=expected_role,
        )
        if (
            evidence.expected_score_loss != self.attempt.expected_score_loss
            or evidence.centipawn_loss != self.attempt.centipawn_loss
            or evidence.lost_forced_mate != self.attempt.lost_forced_mate
            or evidence.mate_delay != self.attempt.mate_delay
            or evidence.verdict != self.attempt.verdict
        ):
            raise ValueError("Review attempt evidence contradicts the checked attempt")
        return self


class PositionReviewFeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rating: Literal["helpful", "unhelpful"]
    reason: Literal[
        "correct",
        "incorrect_chess",
        "irrelevant_topic",
        "unclear",
        "equivalent_move_rejected",
        "too_verbose",
        "missing_detail",
        "other",
    ]
    detail: str | None = Field(default=None, max_length=500)

    @model_validator(mode="after")
    def validate_reason(self) -> PositionReviewFeedbackRequest:
        if self.rating == "helpful" and self.reason != "correct":
            raise ValueError("helpful feedback requires the correct reason")
        if self.rating == "unhelpful" and self.reason == "correct":
            raise ValueError("unhelpful feedback requires a problem reason")
        return self


class PositionReviewFeedbackResponse(BaseModel):
    feedback_id: str


class LearningStatusResponse(BaseModel):
    confirmed_boards: int
    corrected_boards: int
    training_boards: int
    active_model: str
    learning_state: Literal["collecting", "training", "benchmarking", "shadowing"]
    learning_progress: int
    learning_target: int
    candidate_model: str | None


def _validate_review_line(fen: str, line: ReviewLineResponse) -> None:
    board = validate_full_fen(fen)
    for reviewed_move in line.moves:
        move = chess.Move.from_uci(reviewed_move.uci)
        if move not in board.legal_moves or board.san(move) != reviewed_move.san:
            raise ValueError("Review line contains an illegal move or contradictory SAN")
        board.push(move)


def _validate_engine_line_evidence(
    evidence: ReviewEvidenceResponse,
    line: ReviewLineResponse,
    *,
    expected_scope: Literal["best_line", "attempt_line", "attempt_refutation"],
) -> None:
    if (
        evidence.scope != expected_scope
        or evidence.score != line.score
        or evidence.wdl != line.wdl
        or evidence.moves != [move.uci for move in line.moves]
    ):
        raise ValueError("Review engine evidence contradicts its canonical line")


def _square_name(square: str) -> str:
    try:
        chess.parse_square(square)
    except ValueError as exc:
        raise ValueError(f"Invalid review square: {square}") from exc
    return square


def _is_uci_move(move: str) -> bool:
    try:
        parsed = chess.Move.from_uci(move)
    except ValueError:
        return False
    return parsed != chess.Move.null() and parsed.drop is None
