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
