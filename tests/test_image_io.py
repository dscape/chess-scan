from __future__ import annotations

import struct
import zlib

import pytest

from chess_scan.image_io import decode_uploaded_image


def test_rejects_source_pixel_budget_before_decoding() -> None:
    oversized_png = _png_header(width=5001, height=5000)

    with pytest.raises(ValueError, match="too many pixels"):
        decode_uploaded_image(oversized_png, max_dimension=1200)


def _png_header(*, width: int, height: int) -> bytes:
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", header) + _png_chunk(b"IEND", b"")


def _png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", checksum)
