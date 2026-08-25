"""Aggregate confirmed capture quality without exporting private images or labels."""

from __future__ import annotations

import json
from collections import Counter
from statistics import median
from typing import Any

from chess_scan.board import validate_labels


def summarize_confirmed_captures(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_boards = 0
    square_errors = 0
    non_empty_errors = 0
    burst_boards = 0
    submitted_frames = 0
    accepted_frames = 0
    four_frame_boards = 0
    under_three_usable_boards = 0
    accepted_histogram: Counter[int] = Counter()
    rejection_reasons: Counter[str] = Counter()
    routes: Counter[str] = Counter()
    route_scores: list[float] = []
    sharpness_scores: list[float] = []
    glare_scores: list[float] = []
    installations: set[str] = set()

    for row in rows:
        predicted = _labels(row, "predicted_labels_json")
        expected = _labels(row, "final_labels_json")
        errors = [actual != wanted for actual, wanted in zip(predicted, expected, strict=True)]
        exact_boards += not any(errors)
        square_errors += sum(errors)
        non_empty_errors += sum(
            error and wanted != 0 for error, wanted in zip(errors, expected, strict=True)
        )
        installations.add(str(row.get("client_session_id") or "anonymous"))

        evidence = _evidence(row.get("prediction_evidence_json"))
        route = evidence.get("classifier_route", "general")
        routes[str(route) if route in {"general", "print-specialist"} else "unknown"] += 1
        route_score = evidence.get("classifier_route_score")
        if isinstance(route_score, int | float) and not isinstance(route_score, bool):
            route_scores.append(float(route_score))

        accepted = int(row["capture_frame_count"])
        capture = evidence.get("capture")
        submitted = accepted
        if isinstance(capture, dict):
            value = capture.get("submitted_frame_count")
            if isinstance(value, int) and 1 <= value <= 4:
                submitted = value
            rejected = capture.get("rejected_frames")
            if isinstance(rejected, list):
                for frame in rejected:
                    if isinstance(frame, dict) and isinstance(frame.get("reason"), str):
                        rejection_reasons[str(frame["reason"])] += 1
        if submitted > 1:
            burst_boards += 1
            submitted_frames += submitted
            accepted_frames += accepted
            accepted_histogram[accepted] += 1
            if submitted == 4:
                four_frame_boards += 1
                under_three_usable_boards += accepted < 3

        if row.get("sharpness_score") is not None:
            sharpness_scores.append(float(row["sharpness_score"]))
        glare_scores.append(float(row.get("glare_score") or 0.0))

    board_count = len(rows)
    return {
        "boards": board_count,
        "installations": len(installations),
        "exact_boards": exact_boards,
        "exact_board_accuracy": exact_boards / board_count if board_count else None,
        "square_errors": square_errors,
        "square_accuracy": (
            (board_count * 64 - square_errors) / (board_count * 64) if board_count else None
        ),
        "non_empty_errors": non_empty_errors,
        "burst_boards": burst_boards,
        "four_frame_boards": four_frame_boards,
        "under_three_usable_boards": under_three_usable_boards,
        "submitted_burst_frames": submitted_frames,
        "accepted_burst_frames": accepted_frames,
        "burst_frame_acceptance": (
            accepted_frames / submitted_frames if submitted_frames else None
        ),
        "accepted_frame_histogram": {
            str(count): boards for count, boards in sorted(accepted_histogram.items())
        },
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "classifier_routes": dict(sorted(routes.items())),
        "route_score": _distribution(route_scores),
        "sharpness_score": _distribution(sharpness_scores),
        "glare_score": _distribution(glare_scores),
    }


def _labels(row: dict[str, Any], key: str) -> list[int]:
    value = row[key]
    labels = json.loads(value) if isinstance(value, str) else value
    if not isinstance(labels, list) or any(not isinstance(label, int) for label in labels):
        raise ValueError(f"Capture audit has invalid {key}")
    validate_labels(labels)
    return labels


def _evidence(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    parsed = json.loads(value) if isinstance(value, str) else value
    return parsed if isinstance(parsed, dict) else {}


def _distribution(values: list[float]) -> dict[str, float | int] | None:
    if not values:
        return None
    return {
        "count": len(values),
        "min": min(values),
        "median": median(values),
        "max": max(values),
    }
