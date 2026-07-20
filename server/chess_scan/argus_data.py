"""Prepare and evaluate the external Argus square-classifier corpus."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from chess_scan.board import CLASS_NAMES

EXPECTED_ARCHIVE_SHA256 = "4d795dd6734aa275de2fd515989328bfc362ac4a9bb7bf03b2ec8167e5858ac3"
EXPECTED_SOURCE_COUNTS = {"train": 80_000, "test": 20_000, "test_real": 13}
PIECE_DATASET_FILES = ("images.npy", "labels.npy", "metadata.json")
DEFAULT_EXPECTED_MANIFEST = Path(
    os.getenv("CHESS_SCAN_ARGUS_MANIFEST", "benchmarks/argus-training-corpus.json")
).resolve()


@dataclass(frozen=True, slots=True)
class PreparedSplit:
    images: np.ndarray
    labels: np.ndarray
    locators: np.ndarray
    board_names: tuple[str, ...]


def default_data_dir() -> Path:
    configured = os.getenv("CHESS_SCAN_ARGUS_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "chess-scan-training" / "argus-2026-03-29").resolve()


def labels_from_board_filename(filename: str) -> list[int]:
    placement = Path(filename).stem.replace("-", "/")
    labels: list[int] = []
    for character in placement:
        if character == "/":
            continue
        if character.isdigit():
            labels.extend([0] * int(character))
        else:
            labels.append(CLASS_NAMES.index(character))
    if len(labels) != 64:
        raise ValueError(f"Expected 64 labels in chess-positions filename: {filename}")
    return labels


def prepare_sampled_split(
    source_dir: Path,
    output_dir: Path,
    *,
    samples_per_piece_class: int,
    empty_multiplier: int = 6,
    seed: int = 42,
) -> dict[str, Any]:
    files = sorted(path.name for path in source_dir.glob("*.jpeg"))
    random.Random(seed).shuffle(files)
    targets = np.full(len(CLASS_NAMES), samples_per_piece_class, dtype=np.int64)
    targets[0] *= empty_multiplier
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    images: list[np.ndarray] = []
    labels: list[int] = []
    locators: list[tuple[int, int]] = []
    board_names: list[str] = []

    for filename in files:
        if np.all(counts >= targets):
            break
        board = cv2.imread(str(source_dir / filename), cv2.IMREAD_COLOR)
        if board is None or board.shape[:2] != (400, 400):
            continue
        board_index = len(board_names)
        board_names.append(filename)
        for square_index, label in enumerate(labels_from_board_filename(filename)):
            if counts[label] >= targets[label]:
                continue
            row, column = divmod(square_index, 8)
            square = board[row * 50 : (row + 1) * 50, column * 50 : (column + 1) * 50]
            images.append(cv2.resize(square, (64, 64), interpolation=cv2.INTER_LINEAR))
            labels.append(label)
            locators.append((board_index, square_index))
            counts[label] += 1

    if not np.array_equal(counts, targets):
        raise ValueError(
            f"Could not fill sampled split from {source_dir}: "
            f"expected {targets.tolist()}, got {counts.tolist()}"
        )

    order = np.random.RandomState(seed).permutation(len(labels))
    image_array = np.stack(images)[order]
    label_array = np.asarray(labels, dtype=np.int64)[order]
    locator_array = np.asarray(locators, dtype=np.int32)[order]
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = source_dir.name
    np.save(output_dir / f"{prefix}-images.npy", image_array)
    np.save(output_dir / f"{prefix}-labels.npy", label_array)
    np.save(output_dir / f"{prefix}-locators.npy", locator_array)
    (output_dir / f"{prefix}-boards.txt").write_text("\n".join(board_names) + "\n")
    return {
        "boards_scanned": len(board_names),
        "squares": len(label_array),
        "class_counts": counts.tolist(),
    }


def load_prepared_split(data_dir: Path, split: str, *, mmap: bool = True) -> PreparedSplit:
    prepared_dir = data_dir / "prepared"
    mode = "r" if mmap else None
    board_names = tuple(
        name for name in (prepared_dir / f"{split}-boards.txt").read_text().splitlines() if name
    )
    return PreparedSplit(
        images=np.load(prepared_dir / f"{split}-images.npy", mmap_mode=mode),
        labels=np.load(prepared_dir / f"{split}-labels.npy", mmap_mode=mode),
        locators=np.load(prepared_dir / f"{split}-locators.npy", mmap_mode=mode),
        board_names=board_names,
    )


def load_synthetic_replay(data_dir: Path, *, size: int = 64) -> tuple[np.ndarray, np.ndarray]:
    replay_dir = data_dir / "data" / "piece_classifier_dataset"
    images = np.load(replay_dir / "images.npy", mmap_mode="r")
    labels = np.load(replay_dir / "labels.npy", mmap_mode="r")
    resized = np.stack(
        [cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA) for image in images]
    )
    return resized, np.asarray(labels)


def evaluate_argus_model(model_path: Path, data_dir: Path) -> dict[str, Any]:
    verify_data_manifest(data_dir)
    return _evaluate_argus_model(model_path, data_dir)


def _evaluate_argus_model(model_path: Path, data_dir: Path) -> dict[str, Any]:
    split = load_prepared_split(data_dir, "test")
    chess_positions = classification_metrics(
        _predict(model_path, np.asarray(split.images)),
        np.asarray(split.labels),
    )
    replay_images, replay_labels = load_synthetic_replay(data_dir)
    synthetic_replay = classification_metrics(
        _predict(model_path, replay_images),
        replay_labels,
    )
    return {
        "model": model_path.stem,
        "chess_positions_test": chess_positions,
        "synthetic_replay": synthetic_replay,
    }


def evaluate_argus_pair(
    active_model_path: Path,
    candidate_model_path: Path,
    data_dir: Path,
) -> dict[str, Any]:
    verify_data_manifest(data_dir)
    active = _evaluate_argus_model(active_model_path, data_dir)
    candidate = _evaluate_argus_model(candidate_model_path, data_dir)
    passed, reasons = argus_pair_decision(active, candidate)
    return {
        "passed": passed,
        "reasons": reasons,
        "active": active,
        "candidate": candidate,
    }


def argus_pair_decision(
    active: dict[str, Any], candidate: dict[str, Any]
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    active_positions = active["chess_positions_test"]
    candidate_positions = candidate["chess_positions_test"]
    for metric in ("accuracy", "non_empty_accuracy"):
        if candidate_positions[metric] < active_positions[metric]:
            reasons.append(f"chess_positions_test {metric} regressed")
    for class_name, active_class in active_positions["per_class"].items():
        candidate_class = candidate_positions["per_class"][class_name]
        if candidate_class["correct"] < active_class["correct"]:
            reasons.append(f"chess_positions_test class {class_name} regressed")
    if candidate["synthetic_replay"]["correct"] < active["synthetic_replay"]["correct"]:
        reasons.append("synthetic replay accuracy regressed")
    return not reasons, reasons


def classification_metrics(predicted: np.ndarray, expected: np.ndarray) -> dict[str, Any]:
    correct = predicted == expected
    occupied = expected != 0
    per_class = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        selected = expected == class_index
        per_class[class_name or "empty"] = {
            "correct": int(correct[selected].sum()),
            "total": int(selected.sum()),
            "accuracy": float(correct[selected].mean()),
        }
    return {
        "correct": int(correct.sum()),
        "total": len(expected),
        "accuracy": float(correct.mean()),
        "non_empty_correct": int(correct[occupied].sum()),
        "non_empty_total": int(occupied.sum()),
        "non_empty_accuracy": float(correct[occupied].mean()),
        "per_class": per_class,
    }


def validate_source_data(data_dir: Path) -> dict[str, Any]:
    source = data_dir / "data" / "chess_positions"
    counts = {
        "train": len(list((source / "train").glob("*.jpeg"))),
        "test": len(list((source / "test").glob("*.jpeg"))),
        "test_real": len(list((source / "test_real").glob("*.jpg"))),
    }
    if counts != EXPECTED_SOURCE_COUNTS:
        raise ValueError(f"Unexpected chess_positions counts: {counts}")
    replay = data_dir / "data" / "piece_classifier_dataset"
    missing = [name for name in PIECE_DATASET_FILES if not (replay / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing piece-classifier replay files: {missing}")
    metadata = json.loads((replay / "metadata.json").read_text())
    if metadata.get("total_samples") != 19_500 or metadata.get("num_classes") != 13:
        raise ValueError(f"Unexpected piece-classifier metadata: {metadata}")
    return {"chess_positions": counts, "piece_classifier_dataset": metadata}


def verify_data_manifest(
    data_dir: Path,
    expected_manifest_path: Path = DEFAULT_EXPECTED_MANIFEST,
) -> None:
    expected = json.loads(expected_manifest_path.read_text())
    actual = json.loads((data_dir / "MANIFEST.json").read_text())
    for field in ("version", "source_archive_bytes", "source_archive_sha256", "source"):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"External Argus manifest field changed: {field}")
    expected_files = {record["path"]: record for record in expected["files"]}
    actual_files = {record["path"]: record for record in actual["files"]}
    if actual_files != expected_files:
        raise ValueError("External Argus manifest file inventory changed")
    for relative_path, record in expected_files.items():
        path = data_dir / relative_path
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"External Argus data failed verification: {relative_path}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, relative_to: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(relative_to)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _predict(model_path: Path, images_bgr: np.ndarray, *, batch_size: int = 512) -> np.ndarray:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    predictions: list[np.ndarray] = []
    for offset in range(0, len(images_bgr), batch_size):
        batch = np.ascontiguousarray(images_bgr[offset : offset + batch_size, :, :, ::-1])
        inputs = np.transpose(batch.astype(np.float32) / 255.0, (0, 3, 1, 2))
        logits = session.run(None, {input_name: inputs})[0]
        predictions.append(np.asarray(logits).argmax(axis=1))
    return np.concatenate(predictions)
