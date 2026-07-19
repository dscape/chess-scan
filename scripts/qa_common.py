"""Shared helpers for reproducible, non-redistributed QA sources."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

from chess_scan.board import CLASS_NAMES
from chess_scan.model_artifact import sha256_file


def download_verified(url: str, expected_sha256: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and sha256_file(destination) == expected_sha256:
        return

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Chess-Scan-QA/1"})
    try:
        with urllib.request.urlopen(request, timeout=60) as source, temporary.open("wb") as target:
            while chunk := source.read(1024 * 1024):
                target.write(chunk)
        actual_sha256 = sha256_file(temporary)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Official source hash changed for {url}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def labels_from_fen(fen: str) -> list[int]:
    board_fen = fen.split()[0]
    labels: list[int] = []
    for rank in board_fen.split("/"):
        for character in rank:
            if character.isdigit():
                labels.extend([0] * int(character))
            else:
                labels.append(CLASS_NAMES.index(character))
    if len(labels) != 64:
        raise ValueError(f"Expected 64 squares in FEN, got {len(labels)}: {fen}")
    return labels


def write_json(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
