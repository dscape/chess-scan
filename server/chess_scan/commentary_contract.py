"""Shared immutable contracts for optional commentary providers and persistence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from chess_scan.commentary_limits import (
    COMMENTARY_MAX_LESSONS,
    COMMENTARY_MODEL_MAX_LENGTH,
    COMMENTARY_PROVIDER_MAX_LENGTH,
    normalize_commentary_identity,
)
from chess_scan.commentary_narrative import (
    build_coaching_sections,
    unstructured_san_references,
)
from chess_scan.schemas import (
    COMMENTARY_FOCUS_VALUES,
    ENGINE_ONLY_EVIDENCE_KINDS,
    CoachingTextSegment,
    PositionCoachingResponse,
    PositionReviewResponse,
    ReviewAnnotation,
    ReviewEvidenceResponse,
    review_line_for_scope,
)


class CommentarySelectionError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


class CommentaryClaimRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^claim-[1-9][0-9]*$", max_length=32)
    lesson: ReviewAnnotation


def commentary_claim_candidates(
    evidence: Sequence[ReviewEvidenceResponse],
    lessons: Sequence[ReviewAnnotation],
) -> tuple[CommentaryClaimRecord, ...]:
    evidence_by_id = {item.id: item for item in evidence}
    eligible = (
        lesson
        for lesson in lessons
        if any(
            evidence_by_id[evidence_id].kind not in ENGINE_ONLY_EVIDENCE_KINDS
            for evidence_id in lesson.evidence_ids
        )
    )
    return tuple(
        CommentaryClaimRecord(id=f"claim-{index}", lesson=lesson)
        for index, lesson in enumerate(eligible, start=1)
    )


def required_primary_claim_id(
    candidates: Sequence[CommentaryClaimRecord],
) -> str | None:
    return next(
        (
            candidate.id
            for candidate in candidates
            if candidate.lesson.scope == "attempt_refutation"
        ),
        None,
    )


def verified_commentary_selection(
    raw_output: str,
    candidates: Sequence[CommentaryClaimRecord],
    *,
    require_causal_primary: bool = True,
) -> tuple[tuple[str, ...], str]:
    try:
        encoded = raw_output.encode("utf-8")
    except UnicodeEncodeError as error:
        raise CommentarySelectionError("invalid_unicode") from error
    if len(encoded) > 10 * 1024:
        raise CommentarySelectionError("output_too_large")
    try:
        decoded = json.loads(raw_output, object_pairs_hook=_unique_json_object)
    except CommentarySelectionError:
        raise
    except (ValueError, RecursionError) as error:
        raise CommentarySelectionError("invalid_json") from error
    if not isinstance(decoded, dict) or set(decoded) != {"claim_ids", "focus"}:
        raise CommentarySelectionError("invalid_shape")
    claim_ids = decoded["claim_ids"]
    focus = decoded["focus"]
    if (
        not isinstance(claim_ids, list)
        or not 1 <= len(claim_ids) <= COMMENTARY_MAX_LESSONS
        or any(not isinstance(claim_id, str) for claim_id in claim_ids)
        or len(set(claim_ids)) != len(claim_ids)
    ):
        raise CommentarySelectionError("invalid_claim_ids")
    allowed = {candidate.id for candidate in candidates}
    if not set(claim_ids) <= allowed:
        raise CommentarySelectionError("unsupported_claim")
    required_primary = required_primary_claim_id(candidates)
    if require_causal_primary and required_primary is not None and claim_ids[0] != required_primary:
        raise CommentarySelectionError("causal_claim_required")
    if not isinstance(focus, str) or focus not in COMMENTARY_FOCUS_VALUES:
        raise CommentarySelectionError("invalid_focus")
    return tuple(claim_ids), focus


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    keys = [key for key, _value in pairs]
    if len(keys) != len(set(keys)):
        raise CommentarySelectionError("invalid_shape")
    return dict(pairs)


def validate_commentary_response(
    review: PositionReviewResponse,
    response: PositionCoachingResponse,
) -> None:
    if response.review_id != review.review_id:
        raise ValueError("Stored coaching references a different review")
    allowed_lessons = {
        candidate.lesson.id
        for candidate in commentary_claim_candidates(review.evidence, review.explanation)
    }
    if not set(response.lesson_ids) <= allowed_lessons:
        raise ValueError("Stored coaching contains an unsupported lesson")

    evidence_ids = {item.id for item in review.evidence}
    best_line = review.lines[0] if review.lines else None
    attempt_line = review.attempt.line if review.attempt else None
    for section in response.sections:
        if not set(section.evidence_ids) <= evidence_ids:
            raise ValueError("Stored coaching section references unsupported evidence")
        for segment in section.segments:
            if isinstance(segment, CoachingTextSegment):
                if unstructured_san_references(segment.text):
                    raise ValueError("Stored coaching contains an unstructured SAN move")
                continue
            line = review_line_for_scope(
                segment.scope,
                best_line=best_line,
                attempt_line=attempt_line,
            )
            if (
                line is None
                or segment.ply >= len(line.moves)
                or segment.move != line.moves[segment.ply]
            ):
                raise ValueError("Stored coaching move contradicts its checked line")

    if response.sections:
        lessons_by_id = {lesson.id: lesson for lesson in review.explanation}
        selected_lessons = [lessons_by_id[lesson_id] for lesson_id in response.lesson_ids]
        expected_sections = build_coaching_sections(
            review,
            selected_lessons,
            focus=response.focus or "cause",
        )
        if response.sections != expected_sections:
            raise ValueError("Stored coaching narrative contradicts its verified evidence")


class CommentaryRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    response: PositionCoachingResponse
    provider: str = Field(min_length=1, max_length=COMMENTARY_PROVIDER_MAX_LENGTH)
    model: str = Field(min_length=1, max_length=COMMENTARY_MODEL_MAX_LENGTH)
    prompt_version: str = Field(min_length=1, max_length=80)
    request: dict[str, Any]
    raw_output: str | None
    accepted_claim_ids: tuple[str, ...]
    claim_candidates: tuple[CommentaryClaimRecord, ...]
    latency_ms: int = Field(ge=0)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    error_code: str | None = Field(default=None, max_length=80)
    provider_called: bool

    @field_validator("provider")
    @classmethod
    def normalize_provider(cls, value: str) -> str:
        return normalize_commentary_identity(
            value,
            label="provider name",
            max_length=COMMENTARY_PROVIDER_MAX_LENGTH,
        )

    @field_validator("model")
    @classmethod
    def normalize_model(cls, value: str) -> str:
        return normalize_commentary_identity(
            value,
            label="model name",
            max_length=COMMENTARY_MODEL_MAX_LENGTH,
        )


__all__ = [
    "COMMENTARY_MAX_LESSONS",
    "COMMENTARY_MODEL_MAX_LENGTH",
    "COMMENTARY_PROVIDER_MAX_LENGTH",
    "CommentaryClaimRecord",
    "CommentarySelectionError",
    "CommentaryRunRecord",
    "commentary_claim_candidates",
    "required_primary_claim_id",
    "validate_commentary_response",
    "verified_commentary_selection",
    "normalize_commentary_identity",
]
