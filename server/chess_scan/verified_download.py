"""Atomic checksum-pinned download storage shared by QA source transports."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO

from chess_scan.model_artifact import sha256_file


def install_verified_download(
    *,
    source: str,
    expected_sha256: str,
    destination: Path,
    download: Callable[[BinaryIO], object],
) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        if sha256_file(destination) == expected_sha256:
            return expected_sha256
    except FileNotFoundError:
        pass

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{destination.name}.",
            suffix=".part",
            dir=destination.parent,
            delete=False,
        ) as target:
            temporary = Path(target.name)
            download(target)
            target.flush()
        actual_sha256 = sha256_file(temporary)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Pinned source hash changed for {source}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        temporary.replace(destination)
        return actual_sha256
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
