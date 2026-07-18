"""Safe image decoding and encoding."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def decode_uploaded_image(file_bytes: bytes, *, max_dimension: int) -> np.ndarray:
    """Decode upload, apply EXIF orientation, resize, and return BGR pixels."""
    try:
        with Image.open(io.BytesIO(file_bytes)) as source:
            if source.format not in _ALLOWED_FORMATS:
                raise ValueError("Use a JPEG, PNG, or WebP image.")
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            rgb = np.asarray(image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_jpeg(path: Path, image_bgr: np.ndarray, *, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Failed to write image to {path}")
