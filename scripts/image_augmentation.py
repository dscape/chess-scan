"""Shared deterministic image-augmentation primitives for training and QA."""

from __future__ import annotations

import cv2
import numpy as np


def resize_round_trip(board: np.ndarray, size: int, *, contrast: float = 1.0) -> np.ndarray:
    resized = cv2.resize(board, (size, size), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(resized, (512, 512), interpolation=cv2.INTER_CUBIC)
    if contrast == 1.0:
        return resized
    return contrast_brightness(resized, contrast=contrast, brightness=220 * (1 - contrast))


def contrast_brightness(
    image: np.ndarray,
    *,
    contrast: float,
    brightness: float,
) -> np.ndarray:
    return np.clip(image.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)


def hatched_print(
    board: np.ndarray,
    *,
    period: int = 6,
    line_width: int = 2,
    paper_level: int = 242,
    hatch_level: int = 150,
    foreground_contrast: float = 0.8,
    phase: int = 0,
) -> np.ndarray:
    """Render alternating board squares like a low-contrast, diagonally hatched book."""
    if board.ndim != 3 or board.shape[2] != 3:
        raise ValueError("Expected a BGR board image")
    if period < 2 or not 1 <= line_width < period:
        raise ValueError("Hatch width must be positive and smaller than its period")
    if not 0 <= hatch_level <= paper_level <= 255:
        raise ValueError("Hatch and paper levels must be ordered bytes")
    if foreground_contrast <= 0:
        raise ValueError("Foreground contrast must be positive")

    gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY).astype(np.float32)
    height, width = gray.shape
    row_boundaries = [int(round(index * height / 8)) for index in range(9)]
    column_boundaries = [int(round(index * width / 8)) for index in range(9)]
    rendered = np.empty_like(gray)
    y_coordinates, x_coordinates = np.indices(gray.shape)
    hatch = np.where(
        (x_coordinates + y_coordinates + phase) % period < line_width,
        hatch_level,
        paper_level,
    ).astype(np.float32)

    for row in range(8):
        for column in range(8):
            rows = slice(row_boundaries[row], row_boundaries[row + 1])
            columns = slice(column_boundaries[column], column_boundaries[column + 1])
            crop = gray[rows, columns]
            background = float(np.median(crop))
            target = hatch[rows, columns] if (row + column) % 2 else float(paper_level)
            rendered[rows, columns] = np.clip(
                target + (crop - background) * foreground_contrast,
                0,
                255,
            )

    output = rendered.astype(np.uint8)
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)


def jpeg_round_trip(image: np.ndarray, quality: int) -> np.ndarray:
    encoded_ok, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not encoded_ok:
        raise ValueError("Could not encode JPEG augmentation")
    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Could not decode JPEG augmentation")
    return decoded
