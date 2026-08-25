"""Automatic candidate evaluation and promotion rules."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from chess_scan.board import CLASS_NAMES, validate_labels

INITIAL_TRAINING_BOARDS = 100
NEW_TRAINING_BOARDS = 40
MIN_SHADOW_BOARDS = 40
MAX_SHADOW_BOARDS = 120
MIN_SHADOW_CLIENTS = 8
MIN_SPECIALIST_SHADOW_BOARDS = 10
MIN_SPECIALIST_SHADOW_CLIENTS = 3
MAX_BOARDS_PER_CLIENT = 5
MIN_ERROR_IMPROVEMENT = 3
MIN_RELATIVE_ERROR_IMPROVEMENT = 0.05
PERCEPTUAL_HASH_MAX_DISTANCE = 5
_METRIC_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class BoardComparison:
    active_square_errors: int
    candidate_square_errors: int
    active_non_empty_errors: int
    candidate_non_empty_errors: int
    active_board_exact: bool
    candidate_board_exact: bool


@dataclass(frozen=True, slots=True)
class ShadowClassSummary:
    support: int
    active_true_positive: int
    active_false_positive: int
    active_false_negative: int
    active_f1: float
    candidate_true_positive: int
    candidate_false_positive: int
    candidate_false_negative: int
    candidate_f1: float


@dataclass(frozen=True, slots=True)
class ShadowRouteSummary:
    boards: int
    clients: int
    prediction_disagreements: int
    active_square_errors: int
    candidate_square_errors: int
    active_non_empty_errors: int
    candidate_non_empty_errors: int
    active_exact_boards: int
    candidate_exact_boards: int


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
    active_macro_f1: float
    candidate_macro_f1: float
    active_client_macro_accuracy: float
    candidate_client_macro_accuracy: float
    per_class: dict[str, ShadowClassSummary]
    routes: dict[str, ShadowRouteSummary]

    def as_dict(self) -> dict[str, Any]:
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
    active_labels: list[int] = []
    candidate_labels: list[int] = []
    expected_labels: list[int] = []
    comparisons: list[tuple[dict[str, Any], str, BoardComparison, int]] = []
    client_correct: dict[str, list[int]] = {}

    for row in rows:
        active = _stored_labels(row, "active_labels_json")
        candidate = _stored_labels(row, "candidate_labels_json")
        expected = _stored_labels(row, "final_labels_json")
        comparison = compare_labels(active, candidate, expected)
        client = str(row.get("client_session_id") or "anonymous")
        prediction_disagreements = sum(
            active_label != candidate_label
            for active_label, candidate_label in zip(active, candidate, strict=True)
        )
        comparisons.append((row, client, comparison, prediction_disagreements))
        active_labels.extend(active)
        candidate_labels.extend(candidate)
        expected_labels.extend(expected)
        client_metrics = client_correct.setdefault(client, [0, 0, 0])
        client_metrics[0] += 64 - comparison.active_square_errors
        client_metrics[1] += 64 - comparison.candidate_square_errors
        client_metrics[2] += 64

    per_class = _class_summaries(active_labels, candidate_labels, expected_labels)
    routes = {
        route: _route_summary(route_comparisons)
        for route, route_comparisons in _group_by_route(comparisons).items()
    }
    active_macro_f1 = _macro_f1(per_class, candidate=False)
    candidate_macro_f1 = _macro_f1(per_class, candidate=True)
    active_client_macro, candidate_client_macro = _client_macro_accuracies(client_correct)
    board_comparisons = [comparison for _row, _client, comparison, _disagreements in comparisons]
    return ShadowSummary(
        boards=len(rows),
        clients=len(client_correct),
        active_square_errors=sum(item.active_square_errors for item in board_comparisons),
        candidate_square_errors=sum(item.candidate_square_errors for item in board_comparisons),
        active_non_empty_errors=sum(item.active_non_empty_errors for item in board_comparisons),
        candidate_non_empty_errors=sum(
            item.candidate_non_empty_errors for item in board_comparisons
        ),
        active_exact_boards=sum(item.active_board_exact for item in board_comparisons),
        candidate_exact_boards=sum(item.candidate_board_exact for item in board_comparisons),
        active_macro_f1=active_macro_f1,
        candidate_macro_f1=candidate_macro_f1,
        active_client_macro_accuracy=active_client_macro,
        candidate_client_macro_accuracy=candidate_client_macro,
        per_class=per_class,
        routes=routes,
    )


def has_routed_runtime_policy(metadata: dict[str, Any]) -> bool:
    policy = metadata.get("runtime_policy")
    return isinstance(policy, dict) and policy.get("kind") == "directional-print-router-v1"


def requires_general_label_parity(
    candidate_metadata: dict[str, Any],
    baseline_metadata: dict[str, Any],
) -> bool:
    candidate_hash = candidate_metadata.get("artifact_sha256")
    return isinstance(candidate_hash, str) and candidate_hash == baseline_metadata.get(
        "artifact_sha256"
    )


def promotion_decision(
    summary: ShadowSummary,
    *,
    minimum_boards: int = MIN_SHADOW_BOARDS,
    minimum_clients: int = MIN_SHADOW_CLIENTS,
    minimum_specialist_boards: int = 0,
    minimum_specialist_clients: int = 0,
    require_general_label_parity: bool = False,
    minimum_error_improvement: int = MIN_ERROR_IMPROVEMENT,
    minimum_relative_improvement: float = MIN_RELATIVE_ERROR_IMPROVEMENT,
) -> tuple[bool, str]:
    if summary.boards < minimum_boards:
        return False, f"need {minimum_boards - summary.boards} more diverse shadow boards"
    if summary.clients < minimum_clients:
        return False, f"need {minimum_clients - summary.clients} more independent clients"

    specialist = summary.routes.get("print-specialist")
    specialist_boards = 0 if specialist is None else specialist.boards
    specialist_clients = 0 if specialist is None else specialist.clients
    if specialist_boards < minimum_specialist_boards:
        return False, (
            f"need {minimum_specialist_boards - specialist_boards} more specialist-routed boards"
        )
    if specialist_clients < minimum_specialist_clients:
        return False, (
            f"need {minimum_specialist_clients - specialist_clients} more independent "
            "specialist clients"
        )

    general = summary.routes.get("general")
    if (
        require_general_label_parity
        and general is not None
        and general.prediction_disagreements > 0
    ):
        return False, "byte-identical general model changed labels"

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
    if (
        summary.candidate_client_macro_accuracy + _METRIC_TOLERANCE
        < summary.active_client_macro_accuracy
    ):
        return False, "candidate regressed on client-grouped square accuracy"
    if summary.candidate_macro_f1 + _METRIC_TOLERANCE < summary.active_macro_f1:
        return False, "candidate regressed on macro-F1"
    for class_name, metrics in summary.per_class.items():
        if metrics.candidate_true_positive < metrics.active_true_positive:
            return False, f"candidate regressed on {class_name} correct squares"
        if metrics.candidate_f1 + _METRIC_TOLERANCE < metrics.active_f1:
            return False, f"candidate regressed on {class_name} F1"
    for route, metrics in summary.routes.items():
        if metrics.candidate_square_errors > metrics.active_square_errors:
            return False, f"candidate regressed on {route} square errors"
        if metrics.candidate_non_empty_errors > metrics.active_non_empty_errors:
            return False, f"candidate regressed on {route} non-empty squares"
        if metrics.candidate_exact_boards < metrics.active_exact_boards:
            return False, f"candidate regressed on {route} exact boards"
    return True, f"candidate saved {improvement} square errors without a safety regression"


def _stored_labels(row: dict[str, Any], key: str) -> list[int]:
    value = row[key]
    labels = json.loads(value) if isinstance(value, str) else value
    if not isinstance(labels, list) or any(not isinstance(label, int) for label in labels):
        raise ValueError(f"Shadow evaluation has invalid {key}")
    validate_labels(labels)
    return labels


def _class_summaries(
    active: list[int],
    candidate: list[int],
    expected: list[int],
) -> dict[str, ShadowClassSummary]:
    summaries: dict[str, ShadowClassSummary] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        support = sum(label == class_id for label in expected)
        active_tp, active_fp, active_fn = _class_counts(active, expected, class_id)
        candidate_tp, candidate_fp, candidate_fn = _class_counts(candidate, expected, class_id)
        summaries[class_name] = ShadowClassSummary(
            support=support,
            active_true_positive=active_tp,
            active_false_positive=active_fp,
            active_false_negative=active_fn,
            active_f1=_f1(active_tp, active_fp, active_fn),
            candidate_true_positive=candidate_tp,
            candidate_false_positive=candidate_fp,
            candidate_false_negative=candidate_fn,
            candidate_f1=_f1(candidate_tp, candidate_fp, candidate_fn),
        )
    return summaries


def _class_counts(
    predicted: list[int],
    expected: list[int],
    class_id: int,
) -> tuple[int, int, int]:
    true_positive = sum(
        actual == class_id and target == class_id
        for actual, target in zip(predicted, expected, strict=True)
    )
    false_positive = sum(
        actual == class_id and target != class_id
        for actual, target in zip(predicted, expected, strict=True)
    )
    false_negative = sum(
        actual != class_id and target == class_id
        for actual, target in zip(predicted, expected, strict=True)
    )
    return true_positive, false_positive, false_negative


def _f1(true_positive: int, false_positive: int, false_negative: int) -> float:
    denominator = 2 * true_positive + false_positive + false_negative
    return 0.0 if denominator == 0 else 2 * true_positive / denominator


def _macro_f1(
    summaries: dict[str, ShadowClassSummary],
    *,
    candidate: bool,
) -> float:
    values = [
        metrics.candidate_f1 if candidate else metrics.active_f1 for metrics in summaries.values()
    ]
    return sum(values) / len(values) if values else 0.0


def _client_macro_accuracies(client_correct: dict[str, list[int]]) -> tuple[float, float]:
    if not client_correct:
        return 0.0, 0.0
    active = sum(values[0] / values[2] for values in client_correct.values())
    candidate = sum(values[1] / values[2] for values in client_correct.values())
    return active / len(client_correct), candidate / len(client_correct)


def _group_by_route(
    comparisons: list[tuple[dict[str, Any], str, BoardComparison, int]],
) -> dict[str, list[tuple[str, BoardComparison, int]]]:
    grouped: dict[str, list[tuple[str, BoardComparison, int]]] = {}
    for row, client, comparison, disagreements in comparisons:
        route = str(row.get("classifier_route") or "general")
        grouped.setdefault(route, []).append((client, comparison, disagreements))
    return grouped


def _route_summary(
    comparisons: list[tuple[str, BoardComparison, int]],
) -> ShadowRouteSummary:
    return ShadowRouteSummary(
        boards=len(comparisons),
        clients=len({client for client, _comparison, _count in comparisons}),
        prediction_disagreements=sum(count for _client, _item, count in comparisons),
        active_square_errors=sum(
            item.active_square_errors for _client, item, _count in comparisons
        ),
        candidate_square_errors=sum(
            item.candidate_square_errors for _client, item, _count in comparisons
        ),
        active_non_empty_errors=sum(
            item.active_non_empty_errors for _client, item, _count in comparisons
        ),
        candidate_non_empty_errors=sum(
            item.candidate_non_empty_errors for _client, item, _count in comparisons
        ),
        active_exact_boards=sum(item.active_board_exact for _client, item, _count in comparisons),
        candidate_exact_boards=sum(
            item.candidate_board_exact for _client, item, _count in comparisons
        ),
    )


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
