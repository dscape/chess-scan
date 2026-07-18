from __future__ import annotations

from pathlib import Path

import pytest

from chess_scan.bootstrap import initialize_database
from chess_scan.config import PROJECT_ROOT, Settings
from chess_scan.database import Database


@pytest.mark.parametrize(
    "base_version",
    ["argus-v2r5", "chess-steps-v1", "chess-steps-v1r1"],
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
    assert database.get_active_model()["version"] == "chess-steps-v2"

    database.register_candidate(
        version="feedback-candidate",
        artifact_path=settings.model_dir / "chess-steps-v2.onnx",
        metadata={},
    )
    database.promote_model("feedback-candidate")

    assert initialize_database(settings).get_active_model()["version"] == "feedback-candidate"


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
        metadata={},
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
        changed_squares=1,
        consent_training=True,
        client_session_id="client",
    )

    status = database.learning_status()
    examples = database.training_examples()

    assert status["confirmed_boards"] == 1
    assert status["corrected_boards"] == 1
    assert status["training_boards"] == 1
    assert len(examples) == 1
    assert examples[0]["feedback_id"] == "feedback"
    assert database.learning_cycle_progress() == {
        "total_training_boards": 1,
        "boards_in_last_completed_run": 0,
        "new_training_boards": 1,
    }

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
        metadata={"eligible_for_promotion": True},
    )
    database.complete_training_run(
        run_id="run",
        candidate_model_version="candidate",
        metrics={"board_exact": 1.0},
    )
    database.promote_model("candidate")

    assert database.learning_cycle_progress()["new_training_boards"] == 0
    assert database.get_active_model()["version"] == "candidate"
