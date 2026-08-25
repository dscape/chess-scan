"""Tiny ONNX classifier for 64 rectified diagram squares."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import cv2
import numpy as np
import onnxruntime as ort

from chess_scan.board import CLASS_NAMES, SQUARE_COUNT, labels_to_board_fen
from chess_scan.model_artifact import is_sha256, sha256_file, verify_model_artifact

INPUT_SIZE = 64
NUM_CLASSES = len(CLASS_NAMES)
_MULTIVIEW_OFFSETS = tuple(
    (offset_x, offset_y) for offset_y in (-2, 0, 2) for offset_x in (-2, 0, 2)
)
_MULTIVIEW_CENTER_INDEX = _MULTIVIEW_OFFSETS.index((0, 0))
_ENSEMBLE_METHOD = "multi-view-mean-v1"
_RAW_CENTER_EVIDENCE_METHOD = "raw-center-evidence-v1"
# Bound ONNX activation memory while keeping the crop ensemble in one batch.
_MAX_BOARDS_PER_INFERENCE = len(_MULTIVIEW_OFFSETS)


ClassifierRoute = Literal["general", "print-specialist"]


@dataclass(frozen=True, slots=True)
class BoardPrediction:
    labels: list[int]
    probabilities: list[list[float]]
    confidences: list[float]
    board_fen: str


@dataclass(frozen=True, slots=True)
class RuntimeBoardPrediction:
    prediction: BoardPrediction
    classifier_route: ClassifierRoute
    classifier_route_score: float | None


@dataclass(frozen=True, slots=True)
class MultiViewPrediction:
    prediction: BoardPrediction
    raw_prediction: BoardPrediction
    ensemble_prediction: BoardPrediction
    frame_agreements: list[float]
    view_agreements: list[float]
    frame_labels: list[list[int]]
    frame_probabilities: list[list[list[float]]]
    view_labels: list[list[int]]
    inference_view_count: int
    method: str = _RAW_CENTER_EVIDENCE_METHOD
    classifier_route: ClassifierRoute = "general"
    classifier_route_score: float | None = None

    def evidence(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "raw_labels": self.raw_prediction.labels,
            "raw_probabilities": self.raw_prediction.probabilities,
            "ensemble_method": _ENSEMBLE_METHOD,
            "ensemble_labels": self.ensemble_prediction.labels,
            "ensemble_probabilities": self.ensemble_prediction.probabilities,
            "frame_agreements": self.frame_agreements,
            "view_agreements": self.view_agreements,
            "frame_labels": self.frame_labels,
            "frame_probabilities": self.frame_probabilities,
            "view_labels": self.view_labels,
            "inference_view_count": self.inference_view_count,
            "classifier_route": self.classifier_route,
            "classifier_route_score": self.classifier_route_score,
        }


class DiagramClassifier:
    """Classify the squares in one or more rectified boards with ONNX."""

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
        predictions = self.predict_preprocessed(preprocess_board(rectified_board_bgr))
        return predictions[0]

    def predict_with_routing(self, rectified_board_bgr: np.ndarray) -> RuntimeBoardPrediction:
        return RuntimeBoardPrediction(
            prediction=self.predict(rectified_board_bgr),
            classifier_route="general",
            classifier_route_score=None,
        )

    def predict_multiview(
        self,
        composite_board_bgr: np.ndarray,
        supporting_boards_bgr: list[np.ndarray],
    ) -> MultiViewPrediction:
        """Return the raw center prediction with visual stability evidence."""
        view_boards = [
            _shift_board(composite_board_bgr, offset_x, offset_y)
            for offset_x, offset_y in _MULTIVIEW_OFFSETS
        ]
        preprocessed = [preprocess_board(board) for board in [*view_boards, *supporting_boards_bgr]]
        predictions = []
        for start in range(0, len(preprocessed), _MAX_BOARDS_PER_INFERENCE):
            inputs = np.concatenate(preprocessed[start : start + _MAX_BOARDS_PER_INFERENCE])
            predictions.extend(self.predict_preprocessed(inputs))
        view_predictions = predictions[: len(view_boards)]
        frame_predictions = predictions[len(view_boards) :]
        view_probabilities = np.asarray(
            [prediction.probabilities for prediction in view_predictions],
            dtype=np.float32,
        )
        ensemble_prediction = _board_prediction(view_probabilities.mean(axis=0))
        raw_prediction = view_predictions[_MULTIVIEW_CENTER_INDEX]
        presented_labels = np.asarray(raw_prediction.labels, dtype=np.int64)
        view_labels = [candidate.labels for candidate in view_predictions]
        frame_labels = [candidate.labels for candidate in frame_predictions]
        return MultiViewPrediction(
            prediction=raw_prediction,
            raw_prediction=raw_prediction,
            ensemble_prediction=ensemble_prediction,
            frame_agreements=_label_agreements(frame_labels, presented_labels),
            view_agreements=_label_agreements(view_labels, presented_labels),
            frame_labels=frame_labels,
            frame_probabilities=[candidate.probabilities for candidate in frame_predictions],
            view_labels=view_labels,
            inference_view_count=len(predictions),
        )

    def predict_preprocessed(self, inputs: np.ndarray) -> list[BoardPrediction]:
        """Classify one or more boards represented as consecutive batches of 64 squares."""
        if inputs.ndim != 4 or inputs.shape[1:] != (3, INPUT_SIZE, INPUT_SIZE):
            raise ValueError(
                f"Expected square inputs shaped (n, 3, {INPUT_SIZE}, {INPUT_SIZE}), "
                f"got {inputs.shape}"
            )
        if len(inputs) == 0 or len(inputs) % SQUARE_COUNT != 0:
            raise ValueError(
                f"Preprocessed inputs must contain a positive multiple of {SQUARE_COUNT} squares"
            )

        logits = np.asarray(self._session.run(None, {self._input_name: inputs})[0])
        expected_shape = (len(inputs), NUM_CLASSES)
        if logits.shape != expected_shape:
            raise RuntimeError(f"Expected model output {expected_shape}, got {logits.shape}")

        probabilities = _softmax(logits).reshape(-1, SQUARE_COUNT, NUM_CLASSES)
        return [_board_prediction(board_probabilities) for board_probabilities in probabilities]


class RoutedDiagramClassifier:
    """Route visually hatched print boards to a specialist without chess assumptions."""

    def __init__(
        self,
        general: DiagramClassifier,
        print_specialist: DiagramClassifier,
        *,
        texture_threshold: float,
    ) -> None:
        if texture_threshold <= 0:
            raise ValueError("Print texture threshold must be positive")
        self.general = general
        self.print_specialist = print_specialist
        self.texture_threshold = texture_threshold
        self.version = general.version
        self.model_path = general.model_path

    def predict(self, rectified_board_bgr: np.ndarray) -> BoardPrediction:
        return self.predict_with_routing(rectified_board_bgr).prediction

    def predict_with_routing(self, rectified_board_bgr: np.ndarray) -> RuntimeBoardPrediction:
        classifier, route, score = self._classifier(rectified_board_bgr)
        return RuntimeBoardPrediction(
            prediction=classifier.predict(rectified_board_bgr),
            classifier_route=route,
            classifier_route_score=score,
        )

    def predict_multiview(
        self,
        composite_board_bgr: np.ndarray,
        supporting_boards_bgr: list[np.ndarray],
    ) -> MultiViewPrediction:
        classifier, route, score = self._classifier(composite_board_bgr)
        inference = classifier.predict_multiview(composite_board_bgr, supporting_boards_bgr)
        return replace(
            inference,
            classifier_route=route,
            classifier_route_score=score,
        )

    def predict_preprocessed(self, inputs: np.ndarray) -> list[BoardPrediction]:
        """Use the general model when the original board texture is unavailable."""
        return self.general.predict_preprocessed(inputs)

    def _classifier(
        self,
        board_bgr: np.ndarray,
    ) -> tuple[DiagramClassifier, ClassifierRoute, float]:
        score = directional_print_texture_score(board_bgr)
        if score > self.texture_threshold:
            return self.print_specialist, "print-specialist", score
        return self.general, "general", score


def directional_print_texture_score(board_bgr: np.ndarray) -> float:
    """Measure repeated diagonal hatching inside square interiors."""
    if board_bgr.ndim != 3 or board_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR board image")
    gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape
    row_boundaries = [int(round(index * height / 8)) for index in range(9)]
    column_boundaries = [int(round(index * width / 8)) for index in range(9)]
    differences: list[float] = []
    for row in range(8):
        for column in range(8):
            top, bottom = row_boundaries[row], row_boundaries[row + 1]
            left, right = column_boundaries[column], column_boundaries[column + 1]
            margin = max(1, int(round(min(bottom - top, right - left) * 0.1)))
            crop = gray[top + margin : bottom - margin, left + margin : right - margin]
            if min(crop.shape) < 2:
                raise ValueError("Board squares are too small to measure print texture")
            descending = float(np.abs(crop[1:, 1:] - crop[:-1, :-1]).mean())
            ascending = float(np.abs(crop[1:, :-1] - crop[:-1, 1:]).mean())
            differences.append(abs(descending - ascending))
    return float(np.mean(differences))


def load_runtime_classifier(
    model_path: Path,
    *,
    version: str,
    metadata: dict[str, Any],
) -> DiagramClassifier | RoutedDiagramClassifier:
    general = DiagramClassifier(model_path, version=version)
    policy = metadata.get("runtime_policy")
    if policy is None:
        return general
    if not isinstance(policy, dict) or policy.get("kind") != "directional-print-router-v1":
        raise ValueError("Model metadata contains an unsupported runtime policy")
    specialist_name = policy.get("specialist_artifact")
    specialist_sha256 = policy.get("specialist_sha256")
    threshold = policy.get("texture_threshold")
    if (
        not isinstance(specialist_name, str)
        or Path(specialist_name).name != specialist_name
        or not is_sha256(specialist_sha256)
        or isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
    ):
        raise ValueError("Model metadata contains an invalid print-specialist policy")
    specialist_path = model_path.with_name(specialist_name)
    if not specialist_path.is_file() or sha256_file(specialist_path) != specialist_sha256:
        raise ValueError(f"Print specialist failed artifact verification: {specialist_path}")
    specialist_version = str(policy.get("specialist_version") or specialist_path.stem)
    specialist = DiagramClassifier(specialist_path, version=specialist_version)
    return RoutedDiagramClassifier(
        general,
        specialist,
        texture_threshold=float(threshold),
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


def preprocess_board(rectified_board_bgr: np.ndarray) -> np.ndarray:
    normalized_board = normalize_board_contrast(rectified_board_bgr)
    return preprocess_square_crops(split_board_squares(normalized_board))


def preprocess_square_crops(crops: list[np.ndarray]) -> np.ndarray:
    resized: list[np.ndarray] = []
    for crop in crops:
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] == (INPUT_SIZE, INPUT_SIZE):
            resized.append(rgb)
            continue
        interpolation = cv2.INTER_AREA if min(rgb.shape[:2]) >= INPUT_SIZE else cv2.INTER_LINEAR
        resized.append(cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=interpolation))
    batch = np.stack(resized).astype(np.float32) / 255.0
    return np.transpose(batch, (0, 3, 1, 2))


def _shift_board(board_bgr: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray:
    if board_bgr.ndim != 3 or board_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR board image")
    height, width = board_bgr.shape[:2]
    transform = np.asarray([[1.0, 0.0, offset_x], [0.0, 1.0, offset_y]], dtype=np.float32)
    return cv2.warpAffine(
        board_bgr,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _label_agreements(
    candidate_labels: list[list[int]],
    consensus_labels: np.ndarray,
) -> list[float]:
    if not candidate_labels:
        return [1.0] * SQUARE_COUNT
    candidates = np.asarray(candidate_labels, dtype=np.int64)
    if candidates.shape != (len(candidate_labels), SQUARE_COUNT):
        raise ValueError("Candidate labels must contain complete boards")
    return np.mean(candidates == consensus_labels[None, :], axis=0).astype(float).tolist()


def _board_prediction(probabilities: np.ndarray) -> BoardPrediction:
    labels = probabilities.argmax(axis=1).astype(np.int64).tolist()
    return BoardPrediction(
        labels=labels,
        probabilities=probabilities.astype(float).tolist(),
        confidences=probabilities.max(axis=1).astype(float).tolist(),
        board_fen=labels_to_board_fen(labels),
    )


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
        self._classifier: DiagramClassifier | RoutedDiagramClassifier | None = None
        self._version: str | None = None
        self._lock = Lock()

    def active(self) -> DiagramClassifier | RoutedDiagramClassifier:
        model = self._database.get_active_model()
        version = str(model["version"])
        if self._classifier is not None and self._version == version:
            return self._classifier

        with self._lock:
            if self._classifier is None or self._version != version:
                artifact_path = Path(model["artifact_path"])
                metadata_json = str(model["metadata_json"])
                verify_model_artifact(artifact_path, metadata_json)
                self._classifier = load_runtime_classifier(
                    artifact_path,
                    version=version,
                    metadata=json.loads(metadata_json),
                )
                self._version = version
        return self._classifier
