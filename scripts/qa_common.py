"""Shared helpers for reproducible, non-redistributed QA sources."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO

from chess_scan.board import CLASS_NAMES
from chess_scan.verified_download import install_verified_download


def download_verified(url: str, expected_sha256: str, destination: Path) -> None:
    install_verified_download(
        source=url,
        expected_sha256=expected_sha256,
        destination=destination,
        download=lambda target: _download_source(url, target),
    )


def _download_source(url: str, target: BinaryIO) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "Chess-Scan-QA/1"})
    with urllib.request.urlopen(request, timeout=60) as source:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)


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
