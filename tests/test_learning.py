from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from chess_scan.database import Database
from chess_scan.learning import (
    ShadowClassSummary,
    ShadowRouteSummary,
    ShadowSummary,
    compare_labels,
    diverse_shadow_rows,
    has_routed_runtime_policy,
    perceptual_hash,
    promotion_decision,
    requires_general_label_parity,
    summarize_shadow,
)


def test_shadow_policy_requirements_come_from_verified_metadata() -> None:
    artifact_hash = "a" * 64
    candidate = {
        "artifact_sha256": artifact_hash,
        "runtime_policy": {"kind": "directional-print-router-v1"},
    }
    baseline = {"artifact_sha256": artifact_hash}

    assert has_routed_runtime_policy(candidate) is True
    assert requires_general_label_parity(candidate, baseline) is True
    assert has_routed_runtime_policy({"runtime_policy": {"kind": "other"}}) is False
    assert requires_general_label_parity(candidate, {"artifact_sha256": "b" * 64}) is False


def test_shadow_promotion_requires_strict_paired_improvement() -> None:
    active = [0] * 64
    candidate = [0] * 64
    expected = [0] * 64
    expected[:4] = [1, 2, 3, 4]
    candidate[:3] = expected[:3]

    comparison = compare_labels(active, candidate, expected)

    assert comparison.active_square_errors == 4
    assert comparison.candidate_square_errors == 1
    summary = _passing_summary()
    assert promotion_decision(summary)[0] is True
    assert promotion_decision(
        replace(
            summary,
            active_non_empty_errors=11,
            candidate_non_empty_errors=12,
        )
    ) == (False, "candidate regressed on non-empty squares")


def test_shadow_promotion_requires_group_class_and_route_safety() -> None:
    summary = _passing_summary()
    assert promotion_decision(replace(summary, candidate_client_macro_accuracy=0.98)) == (
        False,
        "candidate regressed on client-grouped square accuracy",
    )
    assert promotion_decision(replace(summary, candidate_macro_f1=0.89)) == (
        False,
        "candidate regressed on macro-F1",
    )

    fewer_correct = dict(summary.per_class)
    fewer_correct["Q"] = replace(fewer_correct["Q"], candidate_true_positive=8)
    assert promotion_decision(replace(summary, per_class=fewer_correct)) == (
        False,
        "candidate regressed on Q correct squares",
    )
    regressed_class = dict(summary.per_class)
    regressed_class["Q"] = replace(regressed_class["Q"], candidate_f1=0.8)
    assert promotion_decision(replace(summary, per_class=regressed_class)) == (
        False,
        "candidate regressed on Q F1",
    )

    parity_routes = dict(summary.routes)
    parity_routes["general"] = replace(
        parity_routes["general"],
        prediction_disagreements=1,
    )
    assert promotion_decision(
        replace(summary, routes=parity_routes),
        require_general_label_parity=True,
    ) == (False, "byte-identical general model changed labels")

    routes = dict(summary.routes)
    routes["general"] = replace(
        routes["general"],
        active_square_errors=4,
        candidate_square_errors=5,
    )
    assert promotion_decision(replace(summary, routes=routes)) == (
        False,
        "candidate regressed on general square errors",
    )


def test_routed_shadow_requires_independent_specialist_coverage() -> None:
    summary = _passing_summary()
    assert promotion_decision(
        summary,
        minimum_specialist_boards=10,
        minimum_specialist_clients=3,
    ) == (False, "need 1 more specialist-routed boards")

    specialist = replace(summary.routes["print-specialist"], boards=10, clients=2)
    two_client_summary = replace(
        summary,
        routes={**summary.routes, "print-specialist": specialist},
    )
    assert promotion_decision(
        two_client_summary,
        minimum_specialist_boards=10,
        minimum_specialist_clients=3,
    ) == (False, "need 1 more independent specialist clients")
    assert (
        promotion_decision(
            replace(
                two_client_summary,
                routes={
                    **summary.routes,
                    "print-specialist": replace(specialist, clients=3),
                },
            ),
            minimum_specialist_boards=10,
            minimum_specialist_clients=3,
        )[0]
        is True
    )


def test_shadow_summary_recomputes_metrics_from_resolved_labels() -> None:
    expected = [1, 2, 3, 4] + [0] * 60
    active = [0] * 64
    candidate = [1, 2, 3, 0] + [0] * 60
    rows = [
        {
            "client_session_id": "client-a",
            "classifier_route": "print-specialist",
            "active_labels_json": json.dumps(active),
            "candidate_labels_json": json.dumps(candidate),
            "final_labels_json": json.dumps(expected),
            "active_square_errors": 0,
            "candidate_square_errors": 64,
        }
    ]

    summary = summarize_shadow(rows)

    assert summary.active_square_errors == 4
    assert summary.candidate_square_errors == 1
    assert summary.active_non_empty_errors == 4
    assert summary.candidate_non_empty_errors == 1
    assert summary.routes["print-specialist"].boards == 1
    assert summary.routes["print-specialist"].clients == 1
    assert summary.routes["print-specialist"].prediction_disagreements == 3
    assert summary.per_class["P"].candidate_f1 == 1.0


def test_shadow_diversity_caps_clients_and_near_duplicate_images() -> None:
    rows = [
        {
            "feedback_id": "one",
            "client_session_id": "client-a",
            "image_sha256": "image-a",
            "perceptual_hash": "0000000000000000",
        },
        {
            "feedback_id": "near-duplicate",
            "client_session_id": "client-b",
            "image_sha256": "image-b",
            "perceptual_hash": "0000000000000001",
        },
        {
            "feedback_id": "same-client",
            "client_session_id": "client-a",
            "image_sha256": "image-c",
            "perceptual_hash": "ffffffffffffffff",
        },
    ]

    selected = diverse_shadow_rows(rows, max_per_client=1)
    anonymous = diverse_shadow_rows(
        [
            {
                "feedback_id": "anonymous-one",
                "client_session_id": None,
                "image_sha256": "anonymous-image-one",
                "perceptual_hash": "0000000000000000",
            },
            {
                "feedback_id": "anonymous-two",
                "client_session_id": None,
                "image_sha256": "anonymous-image-two",
                "perceptual_hash": "ffffffffffffffff",
            },
        ],
        max_per_client=1,
    )

    assert [row["feedback_id"] for row in selected] == ["one"]
    assert [row["feedback_id"] for row in anonymous] == ["anonymous-one"]
    board = np.zeros((64, 64, 3), dtype=np.uint8)
    assert len(perceptual_hash(board)) == 16


def test_database_migrates_legacy_shadow_audit_columns(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE shadow_evaluations (
                cycle_id TEXT NOT NULL,
                feedback_id TEXT NOT NULL,
                evaluated_at TEXT NOT NULL,
                perceptual_hash TEXT NOT NULL,
                candidate_labels_json TEXT NOT NULL,
                active_square_errors INTEGER NOT NULL,
                candidate_square_errors INTEGER NOT NULL,
                active_non_empty_errors INTEGER NOT NULL,
                candidate_non_empty_errors INTEGER NOT NULL,
                active_board_exact INTEGER NOT NULL,
                candidate_board_exact INTEGER NOT NULL,
                PRIMARY KEY (cycle_id, feedback_id)
            )
            """
        )
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")

    def initialize() -> None:
        Database(path).initialize(
            base_model_version="base",
            base_model_path=model_path,
            base_model_metadata={"artifact_sha256": _sha256(model_path)},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _index: initialize(), range(2)))

    with sqlite3.connect(path) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(shadow_evaluations)")}
    assert {
        "candidate_probabilities_json",
        "classifier_route",
        "classifier_route_score",
    } <= columns


def test_training_snapshot_caps_one_installation(tmp_path: Path) -> None:
    base_path = tmp_path / "base.onnx"
    base_path.write_bytes(b"base")
    database = Database(tmp_path / "db.sqlite3")
    database.initialize(
        base_model_version="base",
        base_model_path=base_path,
        base_model_metadata={"artifact_sha256": _sha256(base_path)},
    )
    for index in range(6):
        _add_feedback(
            database,
            tmp_path,
            f"feedback-{index}",
            changed=bool(index % 2),
            client_session_id="one-installation",
        )

    assert database.learning_feedback_snapshot(min_total_boards=6, min_new_boards=1) is None
    assert database.learning_feedback_snapshot(min_total_boards=5, min_new_boards=1) == {
        "accepted": [],
        "batch": [f"feedback-{index}" for index in range(5)],
    }


def test_feedback_adjudication_supersedes_labels_without_mutating_confirmation(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.onnx"
    base_path.write_bytes(b"base")
    database = Database(tmp_path / "db.sqlite3")
    database.initialize(
        base_model_version="base",
        base_model_path=base_path,
        base_model_metadata={"artifact_sha256": _sha256(base_path)},
    )
    _add_feedback(database, tmp_path, "feedback", changed=False)
    corrected = [1] + [0] * 63
    database.append_feedback_adjudication(
        adjudication_id="first-review",
        feedback_id="feedback",
        labels=corrected,
        full_fen="P7/8/8/8/8/8/8/8 w - - 0 1",
        reason="The source crop shows a white pawn",
    )

    with sqlite3.connect(database.path) as connection:
        original = connection.execute(
            "SELECT final_labels_json FROM feedback_events WHERE id = 'feedback'"
        ).fetchone()
    assert json.loads(original[0]) == [0] * 64
    assert json.loads(database.training_examples()[0]["final_labels_json"]) == corrected
    assert database.training_examples()[0]["changed_squares"] == 1

    revised = [2] + [0] * 63
    database.append_feedback_adjudication(
        adjudication_id="second-review",
        feedback_id="feedback",
        labels=revised,
        full_fen="N7/8/8/8/8/8/8/8 w - - 0 1",
        reason="A second review identifies the piece as a knight",
    )
    assert json.loads(database.training_examples()[0]["final_labels_json"]) == revised
    assert json.loads(next(database.iter_preference_examples())["final_labels_json"]) == revised
    capture_row = database.confirmed_capture_rows()[0]
    assert json.loads(capture_row["final_labels_json"]) == revised
    assert database.confirmed_capture_rows(created_after="9999-01-01T00:00:00+00:00") == []


def test_database_learning_cycle_accepts_only_a_promoted_training_batch(tmp_path: Path) -> None:
    base_path = tmp_path / "base.onnx"
    candidate_path = tmp_path / "candidate.onnx"
    base_path.write_bytes(b"base")
    candidate_path.write_bytes(b"candidate")
    database = Database(tmp_path / "db.sqlite3")
    database.initialize(
        base_model_version="base",
        base_model_path=base_path,
        base_model_metadata={"artifact_sha256": _sha256(base_path)},
    )
    _add_feedback(database, tmp_path, "first", changed=True)
    _add_feedback(database, tmp_path, "second", changed=False)

    snapshot = database.learning_feedback_snapshot(min_total_boards=2, min_new_boards=1)
    assert snapshot == {"accepted": [], "batch": ["first", "second"]}
    database.create_learning_cycle(
        cycle_id="cycle",
        base_model_version="base",
        accepted_feedback_ids=[],
        batch_feedback_ids=snapshot["batch"],
        shadow_target_boards=1,
    )
    database.register_candidate(
        version="candidate",
        artifact_path=candidate_path,
        metadata={
            "artifact_sha256": _sha256(candidate_path),
            "runtime_policy": {"kind": "directional-print-router-v1"},
        },
    )
    database.set_learning_candidate("cycle", "candidate")
    database.start_shadowing("cycle", {"fixed_qa": "passed"})
    _add_feedback(database, tmp_path, "shadow", changed=True)
    assert [row["feedback_id"] for row in database.shadow_examples("cycle")] == ["shadow"]
    database.record_shadow_evaluation(
        cycle_id="cycle",
        feedback_id="shadow",
        perceptual_hash="0123456789abcdef",
        candidate_labels=[1] + [0] * 63,
        candidate_probabilities=[[1.0] + [0.0] * 12 for _ in range(64)],
        classifier_route="print-specialist",
        classifier_route_score=7.5,
        active_square_errors=1,
        candidate_square_errors=0,
        active_non_empty_errors=1,
        candidate_non_empty_errors=0,
        active_board_exact=False,
        candidate_board_exact=True,
    )
    evaluation = database.shadow_evaluations("cycle")[0]
    assert evaluation["classifier_route"] == "print-specialist"
    assert evaluation["classifier_route_score"] == 7.5
    assert len(json.loads(evaluation["candidate_probabilities_json"])) == 64
    assert json.loads(evaluation["active_labels_json"]) == [0] * 64
    assert json.loads(evaluation["final_labels_json"]) == [1] + [0] * 63
    status = database.learning_status()
    assert status["shadow_installations"] == 1
    assert status["shadow_installation_target"] == 8
    assert status["specialist_progress"] == 1
    assert status["specialist_target"] == 10
    assert status["specialist_installations"] == 1
    assert status["specialist_installation_target"] == 3

    database.append_feedback_adjudication(
        adjudication_id="shadow-review",
        feedback_id="shadow",
        labels=[2] + [0] * 63,
        full_fen="N7/8/8/8/8/8/8/8 w - - 0 1",
        reason="The retained crop shows a knight",
    )
    revised_summary = summarize_shadow(database.shadow_evaluations("cycle"))
    assert revised_summary.active_square_errors == 1
    assert revised_summary.candidate_square_errors == 1
    assert revised_summary.candidate_exact_boards == 0

    promoted = database.promote_learning_cycle(
        "cycle",
        reason="candidate won",
        metrics={"shadow": {"boards": 1}},
    )

    assert promoted == "candidate"
    assert database.get_active_model()["version"] == "candidate"
    assert database.learning_feedback_snapshot(min_total_boards=2, min_new_boards=1) == {
        "accepted": ["first", "second"],
        "batch": ["shadow"],
    }


def _passing_summary() -> ShadowSummary:
    per_class = {
        class_name: ShadowClassSummary(
            support=10,
            active_true_positive=9,
            active_false_positive=1,
            active_false_negative=1,
            active_f1=0.9,
            candidate_true_positive=9,
            candidate_false_positive=1,
            candidate_false_negative=1,
            candidate_f1=0.9,
        )
        for class_name in ("empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
    }
    routes = {
        "general": ShadowRouteSummary(
            boards=31,
            clients=8,
            prediction_disagreements=0,
            active_square_errors=10,
            candidate_square_errors=10,
            active_non_empty_errors=5,
            candidate_non_empty_errors=5,
            active_exact_boards=25,
            candidate_exact_boards=25,
        ),
        "print-specialist": ShadowRouteSummary(
            boards=9,
            clients=3,
            prediction_disagreements=12,
            active_square_errors=10,
            candidate_square_errors=5,
            active_non_empty_errors=5,
            candidate_non_empty_errors=3,
            active_exact_boards=5,
            candidate_exact_boards=6,
        ),
    }
    return ShadowSummary(
        boards=40,
        clients=8,
        active_square_errors=20,
        candidate_square_errors=15,
        active_non_empty_errors=10,
        candidate_non_empty_errors=8,
        active_exact_boards=30,
        candidate_exact_boards=31,
        active_macro_f1=0.9,
        candidate_macro_f1=0.9,
        active_client_macro_accuracy=0.99,
        candidate_client_macro_accuracy=0.995,
        per_class=per_class,
        routes=routes,
    )


def _add_feedback(
    database: Database,
    directory: Path,
    feedback_id: str,
    *,
    changed: bool,
    client_session_id: str | None = None,
) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    rectified = directory / f"{feedback_id}.jpg"
    cv2.imwrite(str(rectified), image)
    predicted = [0] * 64
    final = [1] + [0] * 63 if changed else predicted
    database.create_scan(
        scan_id=f"scan-{feedback_id}",
        image_sha256=f"hash-{feedback_id}",
        source_width=64,
        source_height=64,
        source_image_path=directory / f"source-{feedback_id}.jpg",
        rectified_image_path=rectified,
        corners=[[0, 0], [63, 0], [63, 63], [0, 63]],
        detection_method="test",
        model_version=str(database.get_active_model()["version"]),
        labels=predicted,
        probabilities=[[1.0] + [0.0] * 12 for _ in range(64)],
        board_fen="8/8/8/8/8/8/8/8",
    )
    database.confirm_scan(
        feedback_id=feedback_id,
        scan_id=f"scan-{feedback_id}",
        labels=final,
        orientation="white",
        side_to_move="w",
        castling="-",
        en_passant="-",
        full_fen="8/8/8/8/8/8/8/8 w - - 0 1",
        consent_training=True,
        client_session_id=client_session_id or f"client-{feedback_id}",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
