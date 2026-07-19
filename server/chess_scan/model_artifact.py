"""Integrity helpers for immutable model artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from chess_scan.errors import ArtifactHashMismatchError, MissingArtifactHashError


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_model_artifact(
    path: Path,
    metadata: Mapping[str, Any] | str,
) -> str:
    payload = json.loads(metadata) if isinstance(metadata, str) else metadata
    expected = payload.get("artifact_sha256")
    if not expected:
        raise MissingArtifactHashError("Model metadata is missing artifact_sha256")

    actual = sha256_file(path)
    if actual != expected:
        raise ArtifactHashMismatchError(
            f"Model artifact hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual
