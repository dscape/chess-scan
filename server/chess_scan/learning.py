"""Automatic candidate evaluation and promotion rules."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

INITIAL_TRAINING_BOARDS = 100
NEW_TRAINING_BOARDS = 40
MIN_SHADOW_BOARDS = 40
MAX_SHADOW_BOARDS = 120
MIN_SHADOW_CLIENTS = 8
MAX_BOARDS_PER_CLIENT = 5
MIN_ERROR_IMPROVEMENT = 3
MIN_RELATIVE_ERROR_IMPROVEMENT = 0.05
PERCEPTUAL_HASH_MAX_DISTANCE = 5


@dataclass(frozen=True, slots=True)
class BoardComparison:
    active_square_errors: int
    candidate_square_errors: int
    active_non_empty_errors: int
    candidate_non_empty_errors: int
    active_board_exact: bool
    candidate_board_exact: bool


@dataclass(frozen=True, slots=True)
class ShadowSummary:
    boards: int
    clients: int
    active_square_errors: int
    candidate_square_errors: int
    active_non_empty_errors: int
    candidate_non_empty_errors: int
    active_exact_boards: int
    candidate_exact_boards: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def compare_labels(
    active: list[int],
    candidate: list[int],
    expected: list[int],
) -> BoardComparison:
    if len(active) != 64 or len(candidate) != 64 or len(expected) != 64:
        raise ValueError("Board comparisons require 64 active, candidate, and expected labels")
    active_errors = [
        predicted != target for predicted, target in zip(active, expected, strict=True)
    ]
    candidate_errors = [
        predicted != target for predicted, target in zip(candidate, expected, strict=True)
    ]
    non_empty = [target != 0 for target in expected]
    return BoardComparison(
        active_square_errors=sum(active_errors),
        candidate_square_errors=sum(candidate_errors),
        active_non_empty_errors=sum(
            error and occupied for error, occupied in zip(active_errors, non_empty, strict=True)
        ),
        candidate_non_empty_errors=sum(
            error and occupied for error, occupied in zip(candidate_errors, non_empty, strict=True)
        ),
        active_board_exact=not any(active_errors),
        candidate_board_exact=not any(candidate_errors),
    )


def diverse_shadow_rows(
    rows: list[dict[str, Any]],
    *,
    max_per_client: int = MAX_BOARDS_PER_CLIENT,
    max_hash_distance: int = PERCEPTUAL_HASH_MAX_DISTANCE,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    client_counts: Counter[str] = Counter()
    fingerprints: list[int] = []
    image_hashes: set[str] = set()
    for row in rows:
        client = str(row.get("client_session_id") or "anonymous")
        image_hash = str(row["image_sha256"])
        fingerprint = int(str(row["perceptual_hash"]), 16)
        if client_counts[client] >= max_per_client or image_hash in image_hashes:
            continue
        if any((fingerprint ^ prior).bit_count() <= max_hash_distance for prior in fingerprints):
            continue
        selected.append(row)
        client_counts[client] += 1
        image_hashes.add(image_hash)
        fingerprints.append(fingerprint)
    return selected


def summarize_shadow(rows: list[dict[str, Any]]) -> ShadowSummary:
    return ShadowSummary(
        boards=len(rows),
        clients=len({str(row.get("client_session_id") or "anonymous") for row in rows}),
        active_square_errors=sum(int(row["active_square_errors"]) for row in rows),
        candidate_square_errors=sum(int(row["candidate_square_errors"]) for row in rows),
        active_non_empty_errors=sum(int(row["active_non_empty_errors"]) for row in rows),
        candidate_non_empty_errors=sum(int(row["candidate_non_empty_errors"]) for row in rows),
        active_exact_boards=sum(bool(row["active_board_exact"]) for row in rows),
        candidate_exact_boards=sum(bool(row["candidate_board_exact"]) for row in rows),
    )


def promotion_decision(
    summary: ShadowSummary,
    *,
    minimum_boards: int = MIN_SHADOW_BOARDS,
    minimum_clients: int = MIN_SHADOW_CLIENTS,
    minimum_error_improvement: int = MIN_ERROR_IMPROVEMENT,
    minimum_relative_improvement: float = MIN_RELATIVE_ERROR_IMPROVEMENT,
) -> tuple[bool, str]:
    if summary.boards < minimum_boards:
        return False, f"need {minimum_boards - summary.boards} more diverse shadow boards"
    if summary.clients < minimum_clients:
        return False, f"need {minimum_clients - summary.clients} more independent clients"

    improvement = summary.active_square_errors - summary.candidate_square_errors
    required = max(
        minimum_error_improvement,
        math.ceil(summary.active_square_errors * minimum_relative_improvement),
    )
    if improvement < required:
        return False, f"candidate saved {improvement} square errors; need at least {required}"
    if summary.candidate_non_empty_errors > summary.active_non_empty_errors:
        return False, "candidate regressed on non-empty squares"
    if summary.candidate_exact_boards < summary.active_exact_boards:
        return False, "candidate regressed on exact boards"
    return True, f"candidate saved {improvement} square errors without a safety regression"


def perceptual_hash(board: np.ndarray) -> str:
    if board.ndim != 3 or board.shape[2] != 3:
        raise ValueError("Expected a BGR board image")
    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low_frequencies = cv2.dct(resized)[:8, :8].ravel()
    median = float(np.median(low_frequencies[1:]))
    fingerprint = 0
    for index, value in enumerate(low_frequencies):
        if value > median:
            fingerprint |= 1 << index
    return f"{fingerprint:016x}"
