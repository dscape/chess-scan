from __future__ import annotations

import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from chess_scan.bootstrap import initialize_database
from chess_scan.commentary_contract import CommentaryRunRecord
from chess_scan.config import PROJECT_ROOT, Settings
from chess_scan.database import Database
from chess_scan.errors import (
    ArtifactHashMismatchError,
    MissingArtifactHashError,
    PositionReviewNotFoundError,
    ScanAlreadyConfirmedError,
)
from chess_scan.review import build_position_review
from chess_scan.schemas import (
    PositionCoachingResponse,
    PositionReviewRequest,
    PositionReviewResponse,
)


@pytest.mark.parametrize(
    "base_version",
    [
        "argus-v2r5",
        "chess-steps-v1",
        "chess-steps-v1r1",
        "chess-steps-v2",
        "chess-steps-v3",
        "chess-steps-v4",
    ],
)
def test_bootstrap_replaces_old_base_but_preserves_newer_candidate(
    tmp_path: Path,
    base_version: str,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=PROJECT_ROOT / "models",
        web_dist=tmp_path / "web",
        max_upload_bytes=1024,
        max_image_dimension=1024,
        cors_origins=(),
    )
    database = Database(settings.data_dir / "chess-scan.sqlite3")
    database.initialize(
        base_model_version=base_version,
        base_model_path=settings.model_dir / f"{base_version}.onnx",
        base_model_metadata={},
    )

    database = initialize_database(settings)
    assert database.get_active_model()["version"] == "chess-steps-v5"

    candidate_path = settings.model_dir / "chess-steps-v5.onnx"
    database.register_candidate(
        version="feedback-candidate",
        artifact_path=candidate_path,
        metadata={"artifact_sha256": _sha256(candidate_path)},
    )
    database.promote_model("feedback-candidate")

    assert initialize_database(settings).get_active_model()["version"] == "feedback-candidate"


def test_bootstrap_falls_back_from_legacy_candidate_without_artifact_hash(
    tmp_path: Path,
) -> None:
    settings = Settings(
        data_dir=tmp_path / "data",
        model_dir=PROJECT_ROOT / "models",
        web_dist=tmp_path / "web",
        max_upload_bytes=1024,
        max_image_dimension=1024,
        cors_origins=(),
    )
    database = initialize_database(settings)
    candidate_path = settings.model_dir / "chess-steps-v5.onnx"
    database.register_candidate(
        version="legacy-candidate",
        artifact_path=candidate_path,
        metadata={"artifact_sha256": _sha256(candidate_path)},
    )
    database.promote_model("legacy-candidate")
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE model_versions SET metadata_json = '{}' WHERE version = ?",
            ("legacy-candidate",),
        )

    database = initialize_database(settings)

    assert database.get_active_model()["version"] == "chess-steps-v5"


def test_registering_new_base_does_not_replace_promoted_candidate(tmp_path: Path) -> None:
    database = Database(tmp_path / "db.sqlite3")
    base_path = tmp_path / "base.onnx"
    candidate_path = tmp_path / "candidate.onnx"
    next_base_path = tmp_path / "next-base.onnx"
    for path in (base_path, candidate_path, next_base_path):
        path.write_bytes(b"model")

    database.initialize(
        base_model_version="base",
        base_model_path=base_path,
        base_model_metadata={},
    )
    database.register_candidate(
        version="candidate",
        artifact_path=candidate_path,
        metadata={"artifact_sha256": _sha256(candidate_path)},
    )
    database.promote_model("candidate")
    database.initialize(
        base_model_version="next-base",
        base_model_path=next_base_path,
        base_model_metadata={},
    )

    assert database.get_active_model()["version"] == "candidate"


def test_confirmed_feedback_is_immutable_and_counted(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    database = Database(tmp_path / "db.sqlite3")
    database.initialize(
        base_model_version="base",
        base_model_path=model_path,
        base_model_metadata={"version": "base"},
    )
    source = tmp_path / "source.jpg"
    rectified = tmp_path / "rectified.jpg"
    rectified.write_bytes(b"rectified")
    database.create_scan(
        scan_id="scan",
        image_sha256="hash",
        source_width=100,
        source_height=100,
        source_image_path=source,
        rectified_image_path=rectified,
        corners=[[0, 0], [99, 0], [99, 99], [0, 99]],
        detection_method="test",
        model_version="base",
        labels=[0] * 64,
        probabilities=[[1.0] + [0.0] * 12 for _ in range(64)],
        board_fen="8/8/8/8/8/8/8/8",
    )
    final = [0] * 64
    final[0] = 12
    database.confirm_scan(
        feedback_id="feedback",
        scan_id="scan",
        labels=final,
        orientation="white",
        side_to_move="w",
        castling="-",
        en_passant="-",
        full_fen="k7/8/8/8/8/8/8/8 w - - 0 1",
        consent_training=True,
        client_session_id="client",
    )

    with pytest.raises(ScanAlreadyConfirmedError):
        database.scan_for_display("scan")
    with pytest.raises(ScanAlreadyConfirmedError):
        database.source_image_path("scan")
    with pytest.raises(ScanAlreadyConfirmedError):
        database.update_scan_prediction(
            scan_id="scan",
            rectified_image_path=tmp_path / "replacement.jpg",
            corners=[[0, 0], [99, 0], [99, 99], [0, 99]],
            detection_method="manual",
            model_version="base",
            labels=[1] * 64,
            probabilities=[[1.0] + [0.0] * 12 for _ in range(64)],
            board_fen="8/8/8/8/8/8/8/8",
        )

    status = database.learning_status()
    examples = database.training_examples()

    assert status["confirmed_boards"] == 1
    assert status["corrected_boards"] == 1
    assert status["training_boards"] == 1
    assert len(examples) == 1
    assert examples[0]["feedback_id"] == "feedback"
    database.save_feedback_split_assignments({"feedback": "gate"})
    assert database.feedback_split_assignments() == {"feedback": "gate"}
    with pytest.raises(ValueError):
        database.save_feedback_split_assignments({"feedback": "train"})

    legacy_review_id = "5" * 32
    legacy_review = build_position_review(
        PositionReviewRequest(fen="5Q1k/8/6K1/8/8/8/8/8 b - - 0 1")
    ).model_dump(mode="json")
    legacy_review["schema_version"] = "position-analysis-4"
    legacy_review["hint"].pop("id")
    for annotation in legacy_review["explanation"]:
        annotation.pop("id")
    with sqlite3.connect(database.path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            """
            INSERT INTO position_review_runs (
                id, feedback_id, created_at, schema_version, engine,
                request_json, response_json
            ) VALUES (?, 'feedback', 'now', 'position-analysis-4',
                      'Deterministic rules', '{}', ?)
            """,
            (legacy_review_id, json.dumps(legacy_review, separators=(",", ":"))),
        )
        connection.commit()
    database.initialize(
        base_model_version="base",
        base_model_path=model_path,
        base_model_metadata={"version": "base"},
    )
    migrated_review = PositionReviewResponse.model_validate(
        database.position_review_run(legacy_review_id)
    )
    assert migrated_review.schema_version == "position-analysis-5"
    assert migrated_review.hint.id == "hint"
    assert migrated_review.explanation[0].id == "explanation-1"
    with sqlite3.connect(database.path) as connection:
        stored_legacy_review = connection.execute(
            "SELECT schema_version, response_json FROM position_review_runs WHERE id = ?",
            (legacy_review_id,),
        ).fetchone()
    assert stored_legacy_review[0] == "position-analysis-4"
    assert "id" not in json.loads(stored_legacy_review[1])["hint"]

    review_id = "a" * 32
    planner_run_id = "b" * 32
    late_review_id = "c" * 32
    late_run_id = "d" * 32
    budget_review_id = "e" * 32
    expired_reservation_id = "f" * 32
    budget_run_id = "1" * 32
    over_budget_review_id = "2" * 32
    local_review_id = "3" * 32
    local_run_id = "4" * 32
    stored_review = build_position_review(
        PositionReviewRequest(fen="5Q1k/8/6K1/8/8/8/8/8 b - - 0 1")
    ).model_copy(update={"review_id": review_id})
    stored_response = stored_review.model_dump(mode="json")
    coaching_lesson = stored_review.explanation[0]
    database.save_position_review(
        review_id=review_id,
        feedback_id="feedback",
        fen="k7/8/8/8/8/8/8/8 w - - 0 1",
        schema_version="position-analysis-5",
        engine="Deterministic rules",
        request={"fen": "k7/8/8/8/8/8/8/8 w - - 0 1"},
        response=stored_response,
    )
    assert database.position_review_run(review_id) == stored_response
    coaching_response = PositionCoachingResponse(
        review_id=review_id,
        run_id=planner_run_id,
        status="accepted",
        planner_version="planner-1",
        headline="Follow the cause and effect.",
        lesson_ids=[coaching_lesson.id],
    )
    assert (
        database.reserve_commentary_planner_run(
            review_id=review_id,
            reservation_id=planner_run_id,
            lease_seconds=10,
            max_runs_per_feedback=50,
            max_runs_per_hour=50,
            max_concurrent=5,
        )
        == "reserved"
    )
    planner_record = CommentaryRunRecord(
        response=coaching_response,
        provider="test-provider",
        model="test-model",
        prompt_version="prompt-1",
        request={"evidence": ["f1-e1"]},
        raw_output='{"claim_ids":["claim-1"]}',
        accepted_claim_ids=("claim-1",),
        claim_candidates=({"id": "claim-1", "lesson": coaching_lesson},),
        latency_ms=25,
        input_tokens=100,
        output_tokens=10,
        error_code=None,
        provider_called=True,
    )
    normalized_record = planner_record.model_copy(
        update={"provider": " test-provider ", "model": " test-model "}
    )
    normalized_record = CommentaryRunRecord.model_validate(normalized_record.model_dump())
    assert normalized_record.provider == "test-provider"
    assert normalized_record.model == "test-model"
    with pytest.raises(ValueError, match="requires a provider call"):
        database.save_commentary_planner_run(
            planner_record.model_copy(update={"provider_called": False})
        )
    with pytest.raises(ValueError, match="reservation must match"):
        database.save_commentary_planner_run(
            planner_record.model_copy(
                update={"response": coaching_response.model_copy(update={"run_id": "9" * 32})}
            ),
            reservation_id=planner_run_id,
        )
    with pytest.raises(ValueError, match="claim IDs must be unique"):
        database.save_commentary_planner_run(
            planner_record.model_copy(update={"accepted_claim_ids": ("claim-1", "claim-1")}),
            reservation_id=planner_run_id,
        )
    assert (
        database.save_commentary_planner_run(
            planner_record,
            reservation_id=planner_run_id,
        )
        is True
    )
    assert (
        database.save_commentary_planner_run(
            planner_record,
            reservation_id=planner_run_id,
        )
        is False
    )
    fallback_response = PositionCoachingResponse(
        review_id=review_id,
        run_id="9" * 32,
        status="fallback",
        planner_version="planner-1",
        headline="Verified review",
        lesson_ids=[coaching_lesson.id],
        message=(
            "Deeper coaching is unavailable right now. "
            "The verified evidence-backed lesson is shown instead."
        ),
    )
    with pytest.raises(ValueError, match="cannot contain accepted claim IDs"):
        database.save_commentary_planner_run(
            planner_record.model_copy(
                update={
                    "response": fallback_response,
                    "provider_called": False,
                }
            )
        )
    with pytest.raises(ValueError, match="deterministic policy"):
        database.save_commentary_planner_run(
            planner_record.model_copy(
                update={
                    "response": fallback_response.model_copy(
                        update={"lesson_ids": ["other-lesson"]}
                    ),
                    "accepted_claim_ids": (),
                    "provider_called": False,
                }
            )
        )
    missing_response = fallback_response.model_copy(
        update={"review_id": "f" * 32, "run_id": "e" * 32}
    )
    with pytest.raises(PositionReviewNotFoundError, match="Unknown position review"):
        database.save_commentary_planner_run(
            planner_record.model_copy(
                update={
                    "response": missing_response,
                    "accepted_claim_ids": (),
                    "error_code": "provider_unavailable",
                    "provider_called": False,
                }
            )
        )
    assert database.position_coaching(review_id) == coaching_response
    with sqlite3.connect(database.path) as connection:
        attempt = connection.execute(
            "SELECT admitted_at FROM commentary_planner_attempts WHERE id = ?",
            (planner_run_id,),
        ).fetchone()
        connection.execute(
            "DELETE FROM commentary_planner_attempts WHERE id = ?",
            (planner_run_id,),
        )
        connection.commit()
    with pytest.raises(ValueError, match="no matching admission"):
        database.position_coaching(review_id)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            """
            INSERT INTO commentary_planner_attempts (
                id, review_id, admitted_at
            ) VALUES (?, ?, ?)
            """,
            (planner_run_id, review_id, attempt[0]),
        )
        connection.commit()
    with sqlite3.connect(database.path) as connection:
        corrupt_response = coaching_response.model_dump(mode="json")
        corrupt_response["run_id"] = None
        connection.execute(
            "UPDATE commentary_planner_runs SET response_json = ? WHERE review_id = ?",
            (json.dumps(corrupt_response), review_id),
        )
        connection.commit()
    with pytest.raises(ValueError, match="planner run ID"):
        database.position_coaching(review_id)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE commentary_planner_runs SET response_json = ?, status = 'fallback' "
            "WHERE review_id = ?",
            (coaching_response.model_dump_json(), review_id),
        )
        connection.commit()
    with pytest.raises(ValueError, match="conflicting metadata"):
        database.position_coaching(review_id)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE commentary_planner_runs SET status = 'accepted', accepted_claims_json = ? "
            "WHERE review_id = ?",
            ('["claim-2"]', review_id),
        )
        connection.commit()
    with pytest.raises(ValueError, match="unsupported claim ID"):
        database.commentary_planner_snapshot(review_id)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE commentary_planner_runs SET accepted_claims_json = 'null' WHERE review_id = ?",
            (review_id,),
        )
        connection.commit()
    with pytest.raises(ValueError, match="accepted claims must be a JSON array"):
        database.position_coaching(review_id)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            """
            UPDATE commentary_planner_runs
            SET accepted_claims_json = ?, response_json = '[]'
            WHERE review_id = ?
            """,
            ('["claim-1"]', review_id),
        )
        connection.commit()
    with pytest.raises(ValueError, match="planner response must be a JSON object"):
        database.position_coaching(review_id)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE commentary_planner_runs SET response_json = ? WHERE review_id = ?",
            (coaching_response.model_dump_json(), review_id),
        )
        connection.commit()
    planner_run = database.commentary_planner_snapshot(review_id)
    assert planner_run["provider"] == "test-provider"
    assert planner_run["accepted_claim_ids"] == ["claim-1"]
    assert planner_run["claim_candidates"][0]["id"] == "claim-1"
    assert planner_run["latency_ms"] == 25
    tampered_candidates = planner_run["claim_candidates"]
    tampered_candidates[0]["lesson"]["text"] = "Altered after presentation."
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE commentary_planner_runs SET claim_candidates_json = ? WHERE review_id = ?",
            (json.dumps(tampered_candidates), review_id),
        )
        connection.commit()
    with pytest.raises(ValueError, match="contradict their review"):
        database.commentary_planner_snapshot(review_id)
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE commentary_planner_runs SET claim_candidates_json = ? WHERE review_id = ?",
            (
                json.dumps([{"id": "claim-1", "lesson": coaching_lesson.model_dump(mode="json")}]),
                review_id,
            ),
        )
        connection.commit()
    with pytest.raises(PositionReviewNotFoundError, match="Unknown position review"):
        database.append_position_review_feedback(
            feedback_id="orphan-review-feedback",
            review_id="missing-review",
            rating="unhelpful",
            reason="incorrect_chess",
            detail=None,
            coaching_status="not_shown",
            commentary_run_id=None,
        )
    database.append_position_review_feedback(
        feedback_id="review-feedback",
        review_id=review_id,
        rating="unhelpful",
        reason="incorrect_chess",
        detail="The stated relationship is wrong.",
        coaching_status="accepted",
        commentary_run_id=planner_run_id,
    )
    with sqlite3.connect(database.path) as connection:
        row = connection.execute(
            "SELECT rating, reason FROM position_review_feedback WHERE id = ?",
            ("review-feedback",),
        ).fetchone()
    assert row == ("unhelpful", "incorrect_chess")
    review_feedback = list(database.iter_position_review_feedback())
    assert len(review_feedback) == 1
    assert review_feedback[0]["review_id"] == review_id
    assert review_feedback[0]["coaching_status"] == "accepted"
    assert review_feedback[0]["presented_commentary_run_id"] == planner_run_id
    assert review_feedback[0]["commentary_provider"] == "test-provider"
    assert review_feedback[0]["commentary_provider_called"] == 1
    assert review_feedback[0]["commentary_raw_output"] == '{"claim_ids":["claim-1"]}'
    export_snapshot = database.commentary_snapshot_from_feedback_row(review_feedback[0])
    assert export_snapshot == database.commentary_planner_snapshot(review_id)
    coaching_payload = coaching_response.model_dump(mode="json")
    legacy_response = dict(coaching_payload)
    legacy_response.pop("run_id")
    legacy_response["lessons"] = [coaching_lesson.model_dump(mode="json", exclude={"id"})]
    legacy_response.pop("lesson_ids")
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            """
            UPDATE commentary_planner_runs
            SET response_json = ?, claim_candidates_json = '[]'
            WHERE id = ?
            """,
            (json.dumps(legacy_response, separators=(",", ":")), planner_run_id),
        )
        connection.commit()
    database.initialize(
        base_model_version="base",
        base_model_path=model_path,
        base_model_metadata={"version": "base"},
    )
    legacy_row = list(database.iter_position_review_feedback())[0]
    legacy_snapshot = database.commentary_snapshot_from_feedback_row(legacy_row)
    assert legacy_snapshot["response"] == coaching_payload
    assert legacy_snapshot["stored_response"] == legacy_response
    assert legacy_snapshot["claim_candidates"][0]["lesson"]["id"] == "explanation-1"
    assert database.position_coaching(review_id) == coaching_response
    with sqlite3.connect(database.path) as connection:
        stored_legacy_coaching = json.loads(
            connection.execute(
                "SELECT response_json FROM commentary_planner_runs WHERE id = ?",
                (planner_run_id,),
            ).fetchone()[0]
        )
    assert "lessons" in stored_legacy_coaching
    assert "run_id" not in stored_legacy_coaching
    assert review_feedback[0]["response_json"] == json.dumps(
        stored_response,
        separators=(",", ":"),
    )
    assert list(database.iter_position_review_feedback(rating="helpful")) == []
    assert len(list(database.iter_position_review_feedback(rating="unhelpful"))) == 1
    with pytest.raises(ValueError, match="requires a regression fixture"):
        database.append_position_review_adjudication(
            adjudication_id="missing-fixture",
            review_feedback_id="review-feedback",
            reviewer="expert",
            disposition="approved_fix",
            notes="Verified the corrected explanation.",
            regression_fixture=None,
        )
    with pytest.raises(KeyError, match="Unknown position review feedback"):
        database.append_position_review_adjudication(
            adjudication_id="orphan-adjudication",
            review_feedback_id="missing-review-feedback",
            reviewer="expert",
            disposition="rejected",
            notes="No matching feedback exists.",
            regression_fixture=None,
        )
    database.append_position_review_adjudication(
        adjudication_id="review-adjudication",
        review_feedback_id="review-feedback",
        reviewer="expert",
        disposition="approved_fix",
        notes="Verified the corrected explanation.",
        regression_fixture=(
            "tests/test_review.py::test_review_returns_the_compact_annotation_contract"
        ),
    )
    adjudications = list(database.iter_position_review_adjudications())
    assert len(adjudications) == 1
    assert adjudications[0]["review_feedback_id"] == "review-feedback"
    assert adjudications[0]["regression_fixture"].startswith("tests/")
    export_adjudications = list(
        database.iter_position_review_adjudications_for_export(rating="unhelpful")
    )
    assert [row["adjudication_id"] for row in export_adjudications] == ["review-adjudication"]
    assert list(database.iter_position_review_adjudications_for_export(rating="helpful")) == []
    with pytest.raises(ValueError, match="does not match"):
        database.save_position_review(
            review_id="wrong-fen",
            feedback_id="feedback",
            fen="8/8/8/8/8/8/8/8 w - - 0 1",
            schema_version="position-analysis-3",
            engine="test",
            request={},
            response={},
        )

    no_claim_response = json.loads(json.dumps(stored_response))
    no_claim_response["review_id"] = local_review_id
    no_claim_response["explanation"] = []
    PositionReviewResponse.model_validate(no_claim_response)
    for additional_review_id in (
        late_review_id,
        budget_review_id,
        over_budget_review_id,
        local_review_id,
    ):
        database.save_position_review(
            review_id=additional_review_id,
            feedback_id="feedback",
            fen="k7/8/8/8/8/8/8/8 w - - 0 1",
            schema_version="position-analysis-5",
            engine="Deterministic rules",
            request={},
            response=(
                no_claim_response
                if additional_review_id == local_review_id
                else {**stored_response, "review_id": additional_review_id}
            ),
        )
    database.append_position_review_feedback(
        feedback_id="loading-feedback",
        review_id=late_review_id,
        rating="helpful",
        reason="correct",
        detail=None,
        coaching_status="loading",
        commentary_run_id=None,
    )
    assert (
        database.reserve_commentary_planner_run(
            review_id=late_review_id,
            reservation_id=late_run_id,
            lease_seconds=10,
            max_runs_per_feedback=50,
            max_runs_per_hour=50,
            max_concurrent=5,
        )
        == "reserved"
    )
    late_record = planner_record.model_copy(
        update={
            "response": coaching_response.model_copy(
                update={
                    "review_id": late_review_id,
                    "run_id": late_run_id,
                }
            )
        }
    )
    assert (
        database.save_commentary_planner_run(
            late_record,
            reservation_id=late_run_id,
        )
        is True
    )
    loading_row = next(
        row
        for row in database.iter_position_review_feedback()
        if row["review_feedback_id"] == "loading-feedback"
    )
    assert loading_row["coaching_status"] == "loading"
    assert loading_row["presented_commentary_run_id"] is None
    assert loading_row["commentary_id"] is None

    assert (
        database.reserve_commentary_planner_run(
            review_id=budget_review_id,
            reservation_id=expired_reservation_id,
            lease_seconds=10,
            max_runs_per_feedback=50,
            max_runs_per_hour=50,
            max_concurrent=5,
        )
        == "reserved"
    )
    assert (
        database.reserve_commentary_planner_run(
            review_id=over_budget_review_id,
            reservation_id="blocked-by-shared-limit",
            lease_seconds=10,
            max_runs_per_feedback=50,
            max_runs_per_hour=50,
            max_concurrent=1,
        )
        == "global_busy"
    )
    with sqlite3.connect(database.path) as connection:
        connection.execute(
            "UPDATE commentary_planner_reservations SET lease_expires_at = '2000-01-01' "
            "WHERE review_id = ?",
            (budget_review_id,),
        )
        connection.commit()
    assert (
        database.reserve_commentary_planner_run(
            review_id=budget_review_id,
            reservation_id=budget_run_id,
            lease_seconds=10,
            max_runs_per_feedback=50,
            max_runs_per_hour=50,
            max_concurrent=5,
        )
        == "reserved"
    )
    budget_record = planner_record.model_copy(
        update={
            "response": coaching_response.model_copy(
                update={
                    "review_id": budget_review_id,
                    "run_id": budget_run_id,
                }
            )
        }
    )
    assert (
        database.save_commentary_planner_run(
            budget_record,
            reservation_id=budget_run_id,
            release_reservation=False,
        )
        is True
    )
    with sqlite3.connect(database.path) as connection:
        retained_reservation = connection.execute(
            "SELECT reservation_id FROM commentary_planner_reservations WHERE review_id = ?",
            (budget_review_id,),
        ).fetchone()
    assert retained_reservation == (budget_run_id,)
    assert database.release_commentary_planner_reservation(
        review_id=budget_review_id,
        reservation_id=budget_run_id,
    )
    assert (
        database.reserve_commentary_planner_run(
            review_id=over_budget_review_id,
            reservation_id="global-budget",
            lease_seconds=10,
            max_runs_per_feedback=50,
            max_runs_per_hour=4,
            max_concurrent=5,
        )
        == "global_budget_exhausted"
    )
    assert (
        database.reserve_commentary_planner_run(
            review_id=over_budget_review_id,
            reservation_id="over-budget",
            lease_seconds=10,
            max_runs_per_feedback=4,
            max_runs_per_hour=50,
            max_concurrent=5,
        )
        == "budget_exhausted"
    )
    with sqlite3.connect(database.path) as connection:
        attempts = connection.execute(
            "SELECT COUNT(*) FROM commentary_planner_attempts"
        ).fetchone()[0]
    assert attempts == 4
    with sqlite3.connect(database.path) as connection:
        retried_attempts = connection.execute(
            "SELECT COUNT(*) FROM commentary_planner_attempts WHERE review_id = ?",
            (budget_review_id,),
        ).fetchone()[0]
    assert retried_attempts == 2
    local_response = PositionCoachingResponse(
        review_id=local_review_id,
        run_id=local_run_id,
        status="fallback",
        planner_version="planner-1",
        headline="Verified review",
        lesson_ids=[],
        message="No deeper evidence-backed lesson is available for this position.",
    )
    local_record = planner_record.model_copy(
        update={
            "response": local_response,
            "accepted_claim_ids": (),
            "claim_candidates": (),
            "error_code": "no_claim_candidates",
            "provider_called": False,
        }
    )
    assert database.save_commentary_planner_run(local_record)
    with sqlite3.connect(database.path) as connection:
        attempts_after_local_fallback = connection.execute(
            "SELECT COUNT(*) FROM commentary_planner_attempts"
        ).fetchone()[0]
    assert attempts_after_local_fallback == 4
    database.initialize(
        base_model_version="base",
        base_model_path=model_path,
        base_model_metadata={},
    )
    with sqlite3.connect(database.path) as connection:
        attempts_after_reinitialize = connection.execute(
            "SELECT COUNT(*) FROM commentary_planner_attempts"
        ).fetchone()[0]
    assert attempts_after_reinitialize == 4

    candidate_path = tmp_path / "candidate.onnx"
    candidate_path.write_bytes(b"candidate")
    database.start_training_run(
        run_id="run",
        base_model_version="base",
        training_example_count=1,
    )
    database.register_candidate(
        version="candidate",
        artifact_path=candidate_path,
        metadata={
            "eligible_for_promotion": True,
            "artifact_sha256": _sha256(candidate_path),
        },
    )
    database.complete_training_run(
        run_id="run",
        candidate_model_version="candidate",
        metrics={"board_exact": 1.0},
    )
    database.promote_model("candidate")

    assert database.get_active_model()["version"] == "candidate"


def test_commentary_feedback_migration_enforces_snapshot_integrity(tmp_path: Path) -> None:
    database_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE position_review_feedback (
                id TEXT PRIMARY KEY,
                review_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                rating TEXT NOT NULL,
                reason TEXT NOT NULL,
                detail TEXT
            );
            CREATE TABLE commentary_planner_runs (
                id TEXT PRIMARY KEY,
                review_id TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                status TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                planner_version TEXT NOT NULL,
                request_json TEXT NOT NULL,
                raw_output TEXT,
                accepted_claims_json TEXT NOT NULL,
                response_json TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                input_tokens INTEGER,
                output_tokens INTEGER,
                error_code TEXT,
                provider_called INTEGER NOT NULL DEFAULT 1
            );
            CREATE TABLE commentary_planner_attempts (
                id TEXT PRIMARY KEY,
                review_id TEXT NOT NULL,
                feedback_id TEXT NOT NULL,
                admitted_at TEXT NOT NULL
            );
            CREATE TABLE commentary_planner_reservations (
                review_id TEXT PRIMARY KEY,
                feedback_id TEXT NOT NULL,
                reservation_id TEXT NOT NULL UNIQUE,
                reserved_at TEXT NOT NULL,
                lease_expires_at TEXT NOT NULL
            );
            """
        )
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")

    def initialize() -> None:
        Database(database_path).initialize(
            base_model_version="base",
            base_model_path=model_path,
            base_model_metadata={},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _index: initialize(), range(2)))

    with sqlite3.connect(database_path) as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM model_versions WHERE version = 'base'"
            ).fetchone()[0]
            == 1
        )
        planner_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(commentary_planner_runs)").fetchall()
        }
        assert "claim_candidates_json" in planner_columns
        attempt_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(commentary_planner_attempts)"
            ).fetchall()
        }
        reservation_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(commentary_planner_reservations)"
            ).fetchall()
        }
        assert "feedback_id" not in attempt_columns
        assert "feedback_id" not in reservation_columns
        reservation_indexes = {
            row[1]
            for row in connection.execute(
                "PRAGMA index_list(commentary_planner_reservations)"
            ).fetchall()
        }
        assert "idx_commentary_reservations_expiry" in reservation_indexes
        assert "idx_commentary_reservations_created" in reservation_indexes
        assert "idx_commentary_reservations_lease" not in reservation_indexes
        assert "idx_commentary_reservations_reserved" not in reservation_indexes
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError, match="invalid coaching feedback snapshot"):
            connection.execute(
                """
                INSERT INTO position_review_feedback (
                    id, review_id, created_at, rating, reason, detail,
                    coaching_status, commentary_run_id
                ) VALUES ('invalid', 'review', 'now', 'helpful', 'correct', NULL,
                          'accepted', NULL)
                """
            )


def test_reprocessing_returns_the_path_displaced_inside_the_transaction(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")
    database = Database(tmp_path / "db.sqlite3")
    database.initialize(
        base_model_version="base",
        base_model_path=model_path,
        base_model_metadata={},
    )
    original = tmp_path / "original.jpg"
    first = tmp_path / "first.jpg"
    second = tmp_path / "second.jpg"
    database.create_scan(
        scan_id="scan",
        image_sha256="hash",
        source_width=100,
        source_height=100,
        source_image_path=tmp_path / "source.jpg",
        rectified_image_path=original,
        corners=[[0, 0], [99, 0], [99, 99], [0, 99]],
        detection_method="test",
        model_version="base",
        labels=[0] * 64,
        probabilities=[[1.0] + [0.0] * 12 for _ in range(64)],
        board_fen="8/8/8/8/8/8/8/8",
    )

    displaced_first = database.update_scan_prediction(
        scan_id="scan",
        rectified_image_path=first,
        corners=[[0, 0], [99, 0], [99, 99], [0, 99]],
        detection_method="manual",
        model_version="base",
        labels=[0] * 64,
        probabilities=[[1.0] + [0.0] * 12 for _ in range(64)],
        board_fen="8/8/8/8/8/8/8/8",
    )
    displaced_second = database.update_scan_prediction(
        scan_id="scan",
        rectified_image_path=second,
        corners=[[0, 0], [99, 0], [99, 99], [0, 99]],
        detection_method="manual",
        model_version="base",
        labels=[0] * 64,
        probabilities=[[1.0] + [0.0] * 12 for _ in range(64)],
        board_fen="8/8/8/8/8/8/8/8",
    )

    assert displaced_first == original
    assert displaced_second == first


def test_candidate_artifact_is_verified_before_registration_and_promotion(tmp_path: Path) -> None:
    base_path = tmp_path / "base.onnx"
    base_path.write_bytes(b"base")
    database = Database(tmp_path / "db.sqlite3")
    database.initialize(
        base_model_version="base",
        base_model_path=base_path,
        base_model_metadata={"artifact_sha256": _sha256(base_path)},
    )
    candidate_path = tmp_path / "candidate.onnx"
    candidate_path.write_bytes(b"candidate")

    with pytest.raises(MissingArtifactHashError):
        database.register_candidate(
            version="missing-hash",
            artifact_path=candidate_path,
            metadata={},
        )

    database.register_candidate(
        version="candidate",
        artifact_path=candidate_path,
        metadata={"artifact_sha256": _sha256(candidate_path)},
    )
    candidate_path.write_bytes(b"corrupted")

    with pytest.raises(ArtifactHashMismatchError):
        database.promote_model("candidate")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
