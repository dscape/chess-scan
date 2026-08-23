"""Deterministic quality scoring and multi-frame board composition."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from chess_scan.classifier import split_board_squares

_MAX_COLOR_SPREAD = 12
_OVEREXPOSED_LUMINANCE = 250
_LOCAL_GLARE_FRACTION = 0.25
_LOCAL_GLARE_EXCESS = 0.08
_SEVERE_GLARE_FRACTION = 0.55


@dataclass(frozen=True, slots=True)
class BoardCaptureQuality:
    glare_score: float
    glare_squares: tuple[int, ...]
    square_glare: tuple[float, ...]
    sharpness_score: float


def board_capture_quality(board_bgr: np.ndarray) -> BoardCaptureQuality:
    """Measure localized specular clipping without treating normal white squares as glare."""
    crops = split_board_squares(board_bgr)
    square_glare = np.asarray([_specular_fraction(crop) for crop in crops], dtype=np.float32)
    parity = np.indices((8, 8)).sum(axis=0).reshape(-1) % 2
    baselines = {value: float(np.median(square_glare[parity == value])) for value in (0, 1)}
    excess = np.asarray(
        [
            max(0.0, float(value) - baselines[int(parity[index])])
            for index, value in enumerate(square_glare)
        ],
        dtype=np.float32,
    )
    glare_squares = tuple(
        index
        for index, value in enumerate(square_glare)
        if (value >= _LOCAL_GLARE_FRACTION and excess[index] >= _LOCAL_GLARE_EXCESS)
        or value >= _SEVERE_GLARE_FRACTION
    )
    glare_score = max((float(excess[index]) for index in glare_squares), default=0.0)
    gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
    return BoardCaptureQuality(
        glare_score=min(1.0, glare_score),
        glare_squares=glare_squares,
        square_glare=tuple(float(value) for value in square_glare),
        sharpness_score=_sharpness_score(gray),
    )


def compose_best_board(frames: list[np.ndarray]) -> tuple[np.ndarray, tuple[int, ...]]:
    """Build one board from the clearest square in each aligned rectified frame."""
    if not frames:
        raise ValueError("At least one rectified board frame is required")
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise ValueError("Rectified board frames must share one shape")

    frame_crops = [split_board_squares(frame) for frame in frames]
    selected_frames: list[int] = []
    selected_crops: list[np.ndarray] = []
    for square_index in range(64):
        candidates = [crops[square_index] for crops in frame_crops]
        selected_index = max(
            range(len(candidates)),
            key=lambda index: _square_utility(candidates[index]),
        )
        selected_frames.append(selected_index)
        selected_crops.append(candidates[selected_index])

    height, width = shape[:2]
    row_boundaries = [int(round(index * height / 8)) for index in range(9)]
    column_boundaries = [int(round(index * width / 8)) for index in range(9)]
    composite = np.empty_like(frames[0])
    for square_index, crop in enumerate(selected_crops):
        row, column = divmod(square_index, 8)
        composite[
            row_boundaries[row] : row_boundaries[row + 1],
            column_boundaries[column] : column_boundaries[column + 1],
        ] = crop
    return composite, tuple(selected_frames)


def _sharpness_score(gray: np.ndarray) -> float:
    laplacian = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    horizontal = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    vertical = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    gradient_energy = float(np.mean(horizontal * horizontal + vertical * vertical))
    if gradient_energy <= 1e-6:
        return 0.0
    return max(0.0, 1000.0 * laplacian / gradient_energy)


def _square_utility(crop_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    glare = _specular_fraction(crop_bgr)
    low, high = np.percentile(gray, (5.0, 95.0))
    contrast = float(high - low)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    return 0.7 * float(np.log1p(sharpness)) + 0.015 * contrast - 18.0 * glare


def _specular_fraction(crop_bgr: np.ndarray) -> float:
    channels = crop_bgr.astype(np.int16)
    maximum = channels.max(axis=2)
    minimum = channels.min(axis=2)
    specular = (
        (minimum >= _OVEREXPOSED_LUMINANCE) & (maximum - minimum <= _MAX_COLOR_SPREAD)
    ).astype(np.uint8)
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        specular,
        connectivity=8,
    )
    if component_count <= 1:
        return 0.0
    return float(stats[1:, cv2.CC_STAT_AREA].max() / specular.size)
