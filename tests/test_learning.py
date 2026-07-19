from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np

from chess_scan.database import Database
from chess_scan.learning import (
    ShadowSummary,
    compare_labels,
    diverse_shadow_rows,
    perceptual_hash,
    promotion_decision,
)


def test_shadow_promotion_requires_strict_paired_improvement() -> None:
    active = [0] * 64
    candidate = [0] * 64
    expected = [0] * 64
    expected[:4] = [1, 2, 3, 4]
    candidate[:3] = expected[:3]

    comparison = compare_labels(active, candidate, expected)

    assert comparison.active_square_errors == 4
    assert comparison.candidate_square_errors == 1
    summary = ShadowSummary(
        boards=40,
        clients=8,
        active_square_errors=20,
        candidate_square_errors=15,
        active_non_empty_errors=10,
        candidate_non_empty_errors=8,
        active_exact_boards=30,
        candidate_exact_boards=31,
    )
    assert promotion_decision(summary)[0] is True
    assert promotion_decision(
        ShadowSummary(
            boards=40,
            clients=8,
            active_square_errors=20,
            candidate_square_errors=15,
            active_non_empty_errors=11,
            candidate_non_empty_errors=12,
            active_exact_boards=30,
            candidate_exact_boards=31,
        )
    ) == (False, "candidate regressed on non-empty squares")


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
        metadata={"artifact_sha256": _sha256(candidate_path)},
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
        active_square_errors=1,
        candidate_square_errors=0,
        active_non_empty_errors=1,
        candidate_non_empty_errors=0,
        active_board_exact=False,
        candidate_board_exact=True,
    )

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
