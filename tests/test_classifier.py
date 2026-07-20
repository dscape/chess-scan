from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from chess_scan.classifier import (
    DiagramClassifier,
    ModelManager,
    normalize_board_contrast,
    preprocess_square_crops,
    split_board_squares,
)
from chess_scan.config import PROJECT_ROOT
from chess_scan.database import Database


def test_split_and_preprocess_board() -> None:
    board = np.zeros((512, 512, 3), dtype=np.uint8)
    board[:64, :64] = (10, 20, 30)

    crops = split_board_squares(board)
    batch = preprocess_square_crops(crops)

    assert len(crops) == 64
    assert all(crop.shape == (64, 64, 3) for crop in crops)
    assert batch.shape == (64, 3, 64, 64)
    assert batch.dtype == np.float32
    assert np.isclose(batch[0, 0, 0, 0], 30 / 255)
    assert np.isclose(batch[0, 2, 0, 0], 10 / 255)


def test_classifier_accepts_batched_preprocessed_boards() -> None:
    classifier = DiagramClassifier(
        PROJECT_ROOT / "models" / "chess-steps-v3.onnx",
        version="test",
    )
    inputs = np.zeros((128, 3, 64, 64), dtype=np.float32)

    predictions = classifier.predict_preprocessed(inputs)

    assert len(predictions) == 2
    assert all(len(prediction.labels) == 64 for prediction in predictions)


def test_normalizes_faded_board_luminance_only_when_needed() -> None:
    faded = np.full((64, 64, 3), 220, dtype=np.uint8)
    for row in range(8):
        for col in range(8):
            if (row + col) % 2:
                faded[row * 8 : (row + 1) * 8, col * 8 : (col + 1) * 8] = 205
    faded[24:40, 28:36] = 120
    high_contrast = np.full((64, 64, 3), 255, dtype=np.uint8)
    high_contrast[:, 32:] = 0

    normalized = normalize_board_contrast(faded)

    assert normalized.std() > faded.std() * 2
    assert np.array_equal(normalize_board_contrast(high_contrast), high_contrast)

    tiny = cv2.resize(faded, (4, 4), interpolation=cv2.INTER_AREA)
    blurry = cv2.resize(tiny, (64, 64), interpolation=cv2.INTER_CUBIC)
    assert np.array_equal(normalize_board_contrast(blurry), blurry)


def test_chess_steps_model_passed_recorded_king_gates() -> None:
    model_path = PROJECT_ROOT / "models" / "chess-steps-v2.onnx"
    metadata = json.loads((PROJECT_ROOT / "models" / "chess-steps-v2.json").read_text())

    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == metadata["artifact_sha256"]
    assert metadata["eligible_for_promotion"] is True
    assert metadata["gates"]["independent_manual_king_gate"] == 1.0
    assert metadata["gates"]["localized_board_exact_after_adjudication"] == 1.0
    assert (
        metadata["gates"]["argus_replay_candidate_accuracy"]
        >= (metadata["gates"]["argus_replay_base_accuracy"])
    )


def test_argus_recovery_model_passed_recorded_gates() -> None:
    model_path = PROJECT_ROOT / "models" / "chess-steps-v3.onnx"
    metadata = json.loads((PROJECT_ROOT / "models" / "chess-steps-v3.json").read_text())

    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == metadata["artifact_sha256"]
    assert metadata["eligible_for_promotion"] is True
    assert metadata["gates"]["official_online_exact_boards"] == 267
    assert (
        metadata["gates"]["chess_positions_candidate_correct"]
        > metadata["gates"]["chess_positions_base_correct"]
    )
    assert (
        metadata["gates"]["reviewed_feedback_candidate_exact_boards"]
        >= metadata["gates"]["reviewed_feedback_base_exact_boards"]
    )


def test_model_manager_reloads_promoted_model(tmp_path: Path) -> None:
    base_path = PROJECT_ROOT / "models" / "chess-steps-v3.onnx"
    candidate_path = tmp_path / "candidate.onnx"
    shutil.copyfile(base_path, candidate_path)
    database = Database(tmp_path / "db.sqlite3")
    database.initialize(
        base_model_version="base",
        base_model_path=base_path,
        base_model_metadata={
            "version": "base",
            "artifact_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        },
    )
    manager = ModelManager(database)

    assert manager.active().version == "base"
    database.register_candidate(
        version="candidate",
        artifact_path=candidate_path,
        metadata={
            "version": "candidate",
            "artifact_sha256": hashlib.sha256(candidate_path.read_bytes()).hexdigest(),
        },
    )
    database.promote_model("candidate")

    assert manager.active().version == "candidate"
