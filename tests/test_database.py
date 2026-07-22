from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from chess_scan.bootstrap import initialize_database
from chess_scan.config import PROJECT_ROOT, Settings
from chess_scan.database import Database
from chess_scan.errors import (
    ArtifactHashMismatchError,
    MissingArtifactHashError,
    ScanAlreadyConfirmedError,
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

    stored_response = {"schema_version": "position-analysis-3", "review_id": "review"}
    database.save_position_review(
        review_id="review",
        feedback_id="feedback",
        fen="k7/8/8/8/8/8/8/8 w - - 0 1",
        schema_version="position-analysis-3",
        engine="Deterministic rules",
        request={"fen": "k7/8/8/8/8/8/8/8 w - - 0 1"},
        response=stored_response,
    )
    assert database.position_review_run("review") == stored_response
    with pytest.raises(KeyError, match="Unknown position review"):
        database.append_position_review_feedback(
            feedback_id="orphan-review-feedback",
            review_id="missing-review",
            rating="unhelpful",
            reason="incorrect_chess",
            detail=None,
        )
    database.append_position_review_feedback(
        feedback_id="review-feedback",
        review_id="review",
        rating="unhelpful",
        reason="incorrect_chess",
        detail="The stated relationship is wrong.",
    )
    with sqlite3.connect(database.path) as connection:
        row = connection.execute(
            "SELECT rating, reason FROM position_review_feedback WHERE id = ?",
            ("review-feedback",),
        ).fetchone()
    assert row == ("unhelpful", "incorrect_chess")
    review_feedback = list(database.iter_position_review_feedback())
    assert len(review_feedback) == 1
    assert review_feedback[0]["review_id"] == "review"
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
