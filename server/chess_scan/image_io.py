"""Safe image decoding and encoding."""

from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

_ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
_MAX_SOURCE_PIXELS = 25_000_000


def decode_uploaded_image(file_bytes: bytes, *, max_dimension: int) -> np.ndarray:
    """Decode upload, apply EXIF orientation, resize, and return BGR pixels."""
    if max_dimension <= 0:
        raise ValueError("Maximum image dimension must be positive")
    try:
        with Image.open(io.BytesIO(file_bytes)) as source:
            if source.format not in _ALLOWED_FORMATS:
                raise ValueError("Use a JPEG, PNG, or WebP image.")
            if source.width * source.height > _MAX_SOURCE_PIXELS:
                raise ValueError(
                    "The image has too many pixels. Use a photo smaller than 25 megapixels."
                )
            if source.format == "JPEG":
                source.draft("RGB", (max_dimension, max_dimension))
            image = ImageOps.exif_transpose(source)
            image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc

    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_jpeg(path: Path, image_bgr: np.ndarray, *, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Failed to write image to {path}")
