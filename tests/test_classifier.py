from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from pytest import MonkeyPatch

from chess_scan.classifier import (
    BoardPrediction,
    DiagramClassifier,
    ModelManager,
    RoutedDiagramClassifier,
    directional_print_texture_score,
    load_runtime_classifier,
    normalize_board_contrast,
    preprocess_square_crops,
    read_model_metadata,
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


def test_classifier_keeps_raw_labels_with_auditable_view_evidence(
    monkeypatch: MonkeyPatch,
) -> None:
    classifier = DiagramClassifier(
        PROJECT_ROOT / "models" / "chess-steps-v5.onnx",
        version="test",
    )
    board = np.zeros((512, 512, 3), dtype=np.uint8)
    for row in range(8):
        for column in range(8):
            value = 225 if (row + column) % 2 == 0 else 105
            board[row * 64 : (row + 1) * 64, column * 64 : (column + 1) * 64] = value

    inference_batch_sizes = []
    predict_preprocessed = classifier.predict_preprocessed

    def record_batch_size(inputs: np.ndarray) -> list[BoardPrediction]:
        inference_batch_sizes.append(len(inputs))
        return predict_preprocessed(inputs)

    monkeypatch.setattr(classifier, "predict_preprocessed", record_batch_size)
    inference = classifier.predict_multiview(board, [board] * 4)

    assert inference_batch_sizes == [9 * 64, 4 * 64]
    raw = classifier.predict(board)
    assert inference.method == "raw-center-evidence-v1"
    assert inference.inference_view_count == 13
    assert inference.prediction == inference.raw_prediction
    assert inference.raw_prediction.labels == raw.labels
    assert np.allclose(inference.raw_prediction.probabilities, raw.probabilities)
    assert len(inference.ensemble_prediction.labels) == 64
    assert len(inference.view_labels) == 9
    assert len(inference.frame_labels) == 4
    assert len(inference.frame_probabilities) == 4
    assert all(len(probabilities) == 64 for probabilities in inference.frame_probabilities)
    assert all(0 <= agreement <= 1 for agreement in inference.view_agreements)
    assert all(0 <= agreement <= 1 for agreement in inference.frame_agreements)
    assert np.allclose(
        np.asarray(inference.ensemble_prediction.probabilities).sum(axis=1),
        1.0,
    )
    evidence = inference.evidence()
    assert evidence["raw_labels"] == inference.prediction.labels
    assert evidence["ensemble_labels"] == inference.ensemble_prediction.labels
    assert evidence["ensemble_method"] == "multi-view-mean-v1"
    assert evidence["frame_probabilities"] == inference.frame_probabilities


def test_v6_runtime_policy_loads_verified_specialist() -> None:
    model_path = PROJECT_ROOT / "models" / "chess-steps-v6.onnx"
    metadata = read_model_metadata(model_path.with_suffix(".json"))

    classifier = load_runtime_classifier(
        model_path,
        version="chess-steps-v6",
        metadata=metadata,
    )

    assert isinstance(classifier, RoutedDiagramClassifier)
    assert classifier.general.version == "chess-steps-v6"
    assert classifier.print_specialist.version == "chess-steps-v6-print-specialist"


def test_runtime_router_uses_only_visual_hatching() -> None:
    general = DiagramClassifier(
        PROJECT_ROOT / "models" / "chess-steps-v5.onnx",
        version="test-router",
    )
    specialist = DiagramClassifier(
        PROJECT_ROOT / "models" / "chess-steps-v5.onnx",
        version="test-specialist",
    )
    router = RoutedDiagramClassifier(general, specialist, texture_threshold=5.0)
    plain = np.full((512, 512, 3), 235, dtype=np.uint8)
    hatched = plain.copy()
    y_coordinates, x_coordinates = np.indices((512, 512))
    diagonal_lines = (x_coordinates + y_coordinates) % 6 < 2
    for row in range(8):
        for column in range(8):
            if (row + column) % 2:
                square = hatched[row * 64 : (row + 1) * 64, column * 64 : (column + 1) * 64]
                mask = diagonal_lines[
                    row * 64 : (row + 1) * 64,
                    column * 64 : (column + 1) * 64,
                ]
                square[mask] = 145

    assert directional_print_texture_score(plain) < 1
    assert directional_print_texture_score(hatched) > 5
    assert router.predict_multiview(plain, [plain]).classifier_route == "general"
    assert router.predict_multiview(hatched, [hatched]).classifier_route == "print-specialist"


def test_view_consensus_cannot_replace_the_raw_center_labels(
    monkeypatch: MonkeyPatch,
) -> None:
    classifier = DiagramClassifier(
        PROJECT_ROOT / "models" / "chess-steps-v5.onnx",
        version="test",
    )
    calls = 0

    def prediction(label: int) -> BoardPrediction:
        probabilities = [[0.0] * 13 for _ in range(64)]
        for square in probabilities:
            square[label] = 1.0
        return BoardPrediction(
            labels=[label] * 64,
            probabilities=probabilities,
            confidences=[1.0] * 64,
            board_fen="test",
        )

    def disagreeing_views(inputs: np.ndarray) -> list[BoardPrediction]:
        nonlocal calls
        board_count = len(inputs) // 64
        if calls == 0:
            assert board_count == 9
            labels = [0, 0, 0, 0, 1, 0, 0, 0, 0]
        else:
            assert board_count == 1
            labels = [0]
        calls += 1
        return [prediction(label) for label in labels]

    monkeypatch.setattr(classifier, "predict_preprocessed", disagreeing_views)
    inference = classifier.predict_multiview(
        np.zeros((512, 512, 3), dtype=np.uint8),
        [np.zeros((512, 512, 3), dtype=np.uint8)],
    )

    assert inference.prediction.labels == [1] * 64
    assert inference.raw_prediction.labels == [1] * 64
    assert inference.ensemble_prediction.labels == [0] * 64
    assert inference.view_agreements == [1 / 9] * 64
    assert inference.frame_agreements == [0.0] * 64


def test_classifier_accepts_batched_preprocessed_boards() -> None:
    classifier = DiagramClassifier(
        PROJECT_ROOT / "models" / "chess-steps-v5.onnx",
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


def test_platform_model_passed_recorded_gates() -> None:
    model_path = PROJECT_ROOT / "models" / "chess-steps-v4.onnx"
    checkpoint_path = PROJECT_ROOT / "models" / "chess-steps-v4.pt"
    metadata = json.loads((PROJECT_ROOT / "models" / "chess-steps-v4.json").read_text())

    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == metadata["artifact_sha256"]
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == metadata["checkpoint_sha256"]
    assert metadata["eligible_for_promotion"] is True
    assert metadata["gates"]["platform_clean"]["candidate_exact_boards"] == 982
    assert metadata["gates"]["platform_camera"]["candidate_exact_boards"] == 980
    assert metadata["gates"]["latest_chess_com_correct_squares"] == 64
    assert metadata["gates"]["real_taketaketake_holdout_exact_boards"] == 2
    assert (
        metadata["gates"]["chess_positions_candidate_correct"]
        > metadata["gates"]["chess_positions_base_correct"]
    )
    assert (
        metadata["gates"]["synthetic_replay_candidate_correct"]
        >= metadata["gates"]["synthetic_replay_base_correct"]
    )


def test_print_recovery_model_passed_recorded_gates() -> None:
    model_path = PROJECT_ROOT / "models" / "chess-steps-v5.onnx"
    checkpoint_path = PROJECT_ROOT / "models" / "chess-steps-v5.pt"
    metadata = json.loads((PROJECT_ROOT / "models" / "chess-steps-v5.json").read_text())

    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == metadata["artifact_sha256"]
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() == metadata["checkpoint_sha256"]
    assert metadata["eligible_for_promotion"] is True
    assert metadata["gates"]["print_reference"]["candidate_correct_squares"] == 64
    assert metadata["gates"]["print_reference"]["candidate_exact_robustness_variants"] == 8
    assert (
        metadata["gates"]["platform_clean"]["candidate_exact_boards"]
        > metadata["gates"]["platform_clean"]["base_exact_boards"]
    )
    assert (
        metadata["gates"]["platform_camera"]["candidate_correct_squares"]
        >= metadata["gates"]["platform_camera"]["base_correct_squares"]
    )
    assert (
        metadata["gates"]["chess_positions_candidate_correct"]
        > metadata["gates"]["chess_positions_base_correct"]
    )
    assert (
        metadata["gates"]["synthetic_replay_candidate_correct"]
        > metadata["gates"]["synthetic_replay_base_correct"]
    )


def test_model_manager_reloads_promoted_model(tmp_path: Path) -> None:
    base_path = PROJECT_ROOT / "models" / "chess-steps-v5.onnx"
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
