from __future__ import annotations

import json

from chess_scan.capture_audit import summarize_confirmed_captures


def test_capture_audit_reports_accuracy_burst_quality_and_routes() -> None:
    rows = [
        _row(
            predicted=[0] * 64,
            expected=[1] + [0] * 63,
            accepted=2,
            submitted=4,
            route="print-specialist",
            route_score=7.5,
            rejection_reasons=["grid_not_found", "insufficient_overlap"],
            client="client-a",
        ),
        _row(
            predicted=[0] * 64,
            expected=[0] * 64,
            accepted=4,
            submitted=4,
            route="general",
            route_score=2.0,
            client="client-b",
        ),
        _row(
            predicted=[0] * 64,
            expected=[0] * 64,
            accepted=1,
            submitted=1,
            route="general",
            route_score=None,
            client="client-b",
        ),
    ]

    summary = summarize_confirmed_captures(rows)

    assert summary["boards"] == 3
    assert summary["installations"] == 2
    assert summary["exact_boards"] == 2
    assert summary["square_errors"] == 1
    assert summary["non_empty_errors"] == 1
    assert summary["burst_boards"] == 2
    assert summary["four_frame_boards"] == 2
    assert summary["under_three_usable_boards"] == 1
    assert summary["submitted_burst_frames"] == 8
    assert summary["accepted_burst_frames"] == 6
    assert summary["burst_frame_acceptance"] == 0.75
    assert summary["accepted_frame_histogram"] == {"2": 1, "4": 1}
    assert summary["rejection_reasons"] == {
        "grid_not_found": 1,
        "insufficient_overlap": 1,
    }
    assert summary["auto_capture_boards"] == 2
    assert summary["auto_capture_modes"] == {"balanced": 2}
    assert summary["capture_readiness"] == {
        "count": 2,
        "min": 82.0,
        "median": 82.0,
        "max": 82.0,
    }
    assert summary["readiness_missed_detections"] == 2
    assert summary["classifier_routes"] == {"general": 2, "print-specialist": 1}
    assert summary["route_score"] == {
        "count": 2,
        "min": 2.0,
        "median": 4.75,
        "max": 7.5,
    }


def test_capture_audit_handles_an_empty_cohort() -> None:
    summary = summarize_confirmed_captures([])

    assert summary["boards"] == 0
    assert summary["exact_board_accuracy"] is None
    assert summary["burst_frame_acceptance"] is None
    assert summary["sharpness_score"] is None


def _row(
    *,
    predicted: list[int],
    expected: list[int],
    accepted: int,
    submitted: int,
    route: str,
    route_score: float | None,
    client: str,
    rejection_reasons: list[str] | None = None,
) -> dict[str, object]:
    capture = {
        "submitted_frame_count": submitted,
        "rejected_frames": [
            {"capture_index": index + accepted, "reason": reason}
            for index, reason in enumerate(rejection_reasons or [])
        ],
    }
    if submitted > 1:
        capture["readiness"] = {
            "schema_version": "scan-readiness-1",
            "capture_kind": "automatic",
            "auto_capture_mode": "balanced",
            "readiness_score": 82,
            "readiness_threshold": 80,
            "detection_attempts": 4,
            "missed_detections": 1,
        }
    evidence = {
        "classifier_route": route,
        "classifier_route_score": route_score,
        "capture": capture,
    }
    return {
        "predicted_labels_json": json.dumps(predicted),
        "final_labels_json": json.dumps(expected),
        "prediction_evidence_json": json.dumps(evidence),
        "capture_frame_count": accepted,
        "client_session_id": client,
        "sharpness_score": float(accepted * 100),
        "glare_score": 0.05,
    }
