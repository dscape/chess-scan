"""Tiny ONNX classifier for 64 rectified diagram squares."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from chess_scan.board import CLASS_NAMES, labels_to_board_fen

INPUT_SIZE = 64
NUM_CLASSES = len(CLASS_NAMES)


@dataclass(frozen=True, slots=True)
class BoardPrediction:
    labels: list[int]
    probabilities: list[list[float]]
    confidences: list[float]
    board_fen: str


class DiagramClassifier:
    """Classify all squares in one rectified board with a single ONNX call."""

    def __init__(self, model_path: Path, *, version: str) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        self.model_path = model_path
        self.version = version
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 4
        self._session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name

    def predict(self, rectified_board_bgr: np.ndarray) -> BoardPrediction:
        normalized_board = normalize_board_contrast(rectified_board_bgr)
        crops = split_board_squares(normalized_board)
        inputs = preprocess_square_crops(crops)
        logits = np.asarray(self._session.run(None, {self._input_name: inputs})[0])
        if logits.shape != (64, NUM_CLASSES):
            raise RuntimeError(f"Expected model output (64, {NUM_CLASSES}), got {logits.shape}")

        probabilities = _softmax(logits)
        labels = probabilities.argmax(axis=1).astype(np.int64).tolist()
        confidences = probabilities.max(axis=1).astype(float).tolist()
        return BoardPrediction(
            labels=labels,
            probabilities=probabilities.astype(float).tolist(),
            confidences=confidences,
            board_fen=labels_to_board_fen(labels),
        )


def normalize_board_contrast(board_bgr: np.ndarray) -> np.ndarray:
    """Restore faded printed diagrams without changing normal high-contrast boards."""
    if board_bgr.ndim != 3 or board_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR board image")
    lab = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2LAB)
    luminance = lab[:, :, 0].astype(np.float32)
    low, lower_background, upper_background, high = np.percentile(
        luminance,
        (1.0, 25.0, 75.0, 99.0),
    )
    spread = float(high - low)
    checker_contrast = float(upper_background - lower_background)
    sharpness = float(cv2.Laplacian(luminance, cv2.CV_32F).var())
    if spread >= 180.0 or spread < 8.0 or checker_contrast >= 25.0 or sharpness < 10.0:
        return board_bgr

    lab[:, :, 0] = np.clip((luminance - low) * (255.0 / spread), 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def split_board_squares(board_bgr: np.ndarray) -> list[np.ndarray]:
    if board_bgr.ndim != 3 or board_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR board image")
    height, width = board_bgr.shape[:2]
    if height != width:
        raise ValueError("Rectified board image must be square")

    boundaries = [int(round(index * width / 8)) for index in range(9)]
    return [
        board_bgr[boundaries[row] : boundaries[row + 1], boundaries[col] : boundaries[col + 1]]
        for row in range(8)
        for col in range(8)
    ]


def preprocess_square_crops(crops: list[np.ndarray]) -> np.ndarray:
    resized: list[np.ndarray] = []
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        interpolation = cv2.INTER_AREA if min(rgb.shape[:2]) >= INPUT_SIZE else cv2.INTER_LINEAR
        resized.append(cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=interpolation))
    batch = np.stack(resized).astype(np.float32) / 255.0
    return np.transpose(batch, (0, 3, 1, 2))


def read_model_metadata(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


class ModelManager:
    """Reload the active model atomically after a registry promotion."""

    def __init__(self, database: Any) -> None:
        self._database = database
        self._classifier: DiagramClassifier | None = None
        self._version: str | None = None
        self._lock = Lock()

    def active(self) -> DiagramClassifier:
        model = self._database.get_active_model()
        version = str(model["version"])
        if self._classifier is not None and self._version == version:
            return self._classifier

        with self._lock:
            if self._classifier is None or self._version != version:
                self._classifier = DiagramClassifier(Path(model["artifact_path"]), version=version)
                self._version = version
        return self._classifier
