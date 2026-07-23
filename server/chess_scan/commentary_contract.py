"""Shared immutable contracts for optional commentary providers and persistence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from chess_scan.commentary_limits import (
    COMMENTARY_MAX_LESSONS,
    COMMENTARY_MODEL_MAX_LENGTH,
    COMMENTARY_PROVIDER_MAX_LENGTH,
    normalize_commentary_identity,
)
from chess_scan.schemas import (
    ENGINE_ONLY_EVIDENCE_KINDS,
    PositionCoachingResponse,
    ReviewAnnotation,
    ReviewEvidenceResponse,
)


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
    "CommentaryRunRecord",
    "commentary_claim_candidates",
    "normalize_commentary_identity",
]
