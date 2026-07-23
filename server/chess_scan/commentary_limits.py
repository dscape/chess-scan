"""Dependency-free limits shared by commentary schemas and providers."""

from __future__ import annotations

COMMENTARY_PROVIDER_MAX_LENGTH = 80
COMMENTARY_MODEL_MAX_LENGTH = 160
COMMENTARY_MAX_LESSONS = 2


def normalize_commentary_identity(value: str, *, label: str, max_length: int) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > max_length:
        raise ValueError(f"Commentary planner {label} must contain 1 to {max_length} characters")
    return normalized
