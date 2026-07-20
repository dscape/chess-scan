"""Verification and evaluation helpers for the external platform corpus."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from chess_scan.metrics import classification_metrics

_MANIFEST_NAME = "platform-training-corpus.json"
_REAL_RECORD_FILES = ("real/records.jsonl", "real/squares.jsonl")
PLATFORM_THEMES = {
    "chess.com": (
        ("green", "#edeed1", "#779952"),
        ("blue", "#ececd7", "#4d6d92"),
        ("brown", "#f0d9b5", "#b58863"),
        ("purple", "#efefef", "#8877b7"),
        ("sky", "#efefef", "#c2d7e2"),
        ("tournament", "#ebece8", "#316549"),
        ("bubblegum", "#fff3f3", "#f9cdd3"),
        ("metal", "#c9c9c9", "#6e6e6e"),
    ),
    "lichess": (
        ("brown", "#f0d9b5", "#b58863"),
        ("blue", "#dee3e6", "#8ca2ad"),
        ("green", "#ffffdd", "#86a666"),
        ("purple", "#e5daf0", "#957ab0"),
        ("grey", "#d3d3d3", "#8a8a8a"),
        ("wood", "#d8b170", "#a06b3b"),
    ),
    "taketaketake": (
        ("purple", "#dad9e7", "#aaa0bd"),
        ("violet", "#e5e0ef", "#9588aa"),
        ("neutral", "#e8e8e8", "#aaa7a4"),
    ),
}


def _default_expected_manifest() -> Path:
    configured = os.getenv("CHESS_SCAN_PLATFORM_MANIFEST")
    if configured:
        return Path(configured).expanduser().resolve()
    source_manifest = Path(__file__).resolve().parents[2] / "benchmarks" / _MANIFEST_NAME
    if source_manifest.is_file():
        return source_manifest
    return (Path.cwd() / "benchmarks" / _MANIFEST_NAME).resolve()


DEFAULT_EXPECTED_MANIFEST = _default_expected_manifest()


def default_data_dir() -> Path:
    configured = os.getenv("CHESS_SCAN_PLATFORM_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "chess-scan-training" / "platforms-v1").resolve()


def load_records(data_dir: Path, *, split: str) -> list[dict[str, Any]]:
    return read_records(data_dir / "records.jsonl", split=split)


def read_records(path: Path, *, split: str | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON record at {path}:{line_number}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"Expected an object record at {path}:{line_number}")
        required_strings = ("path", "split", "sha256")
        if any(not isinstance(record.get(field), str) for field in required_strings):
            raise ValueError(f"Record is missing path, split, or sha256 at {path}:{line_number}")
        if record["split"] not in {"train", "test"}:
            raise ValueError(f"Record has an invalid split at {path}:{line_number}")
        records.append(record)

    selected = (
        records if split is None else [record for record in records if record["split"] == split]
    )
    if split is not None and not selected:
        raise ValueError(f"No {split} records in {path}")
    return selected


def verify_data_manifest(
    data_dir: Path,
    expected_manifest_path: Path = DEFAULT_EXPECTED_MANIFEST,
) -> dict[str, Any]:
    expected = json.loads(expected_manifest_path.read_text())
    actual = json.loads((data_dir / "MANIFEST.json").read_text())
    for field in (
        "version",
        "seed",
        "positions",
        "test_positions",
        "records",
        "platforms",
        "records_sha256",
        "real_records_sha256",
        "real_squares_sha256",
        "source_files",
    ):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"External platform manifest field changed: {field}")

    records_path = data_dir / "records.jsonl"
    if sha256_file(records_path) != expected["records_sha256"]:
        raise ValueError("External platform records failed verification")
    records = read_records(records_path)
    if len(records) != expected["records"]:
        raise ValueError("External platform record count changed")
    verify_record_images(data_dir, records)

    for relative_path, digest_field in zip(
        _REAL_RECORD_FILES,
        ("real_records_sha256", "real_squares_sha256"),
        strict=True,
    ):
        path = data_dir / relative_path
        if sha256_file(path) != expected[digest_field]:
            raise ValueError(f"External platform records failed verification: {relative_path}")
        verify_record_images(data_dir, read_records(path))

    for record in expected["source_files"]:
        path = data_dir / record["path"]
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"External platform source failed verification: {record['path']}")
    return actual


def verify_record_images(data_dir: Path, records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for record in records:
        relative_path = str(record["path"])
        if relative_path in seen:
            raise ValueError(f"Duplicate external platform image record: {relative_path}")
        seen.add(relative_path)
        if sha256_file(data_dir / relative_path) != record["sha256"]:
            raise ValueError(f"External platform image failed verification: {relative_path}")


def summarize_platform_predictions(
    records: list[dict[str, Any]],
    expected: list[np.ndarray],
    predicted: list[np.ndarray],
) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(record["platform"], record["piece_style"])].append(index)
    styles = {
        f"{platform}/{style}": board_metrics(indices, expected, predicted)
        for (platform, style), indices in groups.items()
    }
    platforms = {
        platform: board_metrics(
            [index for index, record in enumerate(records) if record["platform"] == platform],
            expected,
            predicted,
        )
        for platform in sorted({record["platform"] for record in records})
    }
    return {
        "boards": len(records),
        "overall": board_metrics(list(range(len(records))), expected, predicted),
        "platforms": platforms,
        "styles": styles,
    }


def board_metrics(
    indices: list[int],
    expected: list[np.ndarray],
    predicted: list[np.ndarray],
) -> dict[str, Any]:
    wanted = np.concatenate([expected[index] for index in indices])
    actual = np.concatenate([predicted[index] for index in indices])
    metrics = classification_metrics(actual, wanted)
    correct_by_board = [int((predicted[index] == expected[index]).sum()) for index in indices]
    exact_boards = sum(correct == 64 for correct in correct_by_board)
    metrics.update(
        {
            "boards": len(indices),
            "exact_boards": exact_boards,
            "exact_board_accuracy": exact_boards / len(indices),
            "worst_board_correct_squares": min(correct_by_board),
        }
    )
    return metrics


def platform_pair_decision(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    groups = {"overall": (baseline["overall"], candidate["overall"])}
    groups.update(
        {
            f"platform {platform}": (metrics, candidate["platforms"][platform])
            for platform, metrics in baseline["platforms"].items()
        }
    )
    for name, (active, proposed) in groups.items():
        if proposed["exact_boards"] < active["exact_boards"]:
            reasons.append(f"{name} exact boards regressed")
        if proposed["non_empty_correct"] < active["non_empty_correct"]:
            reasons.append(f"{name} occupied squares regressed")
        if proposed["correct"] < active["correct"]:
            reasons.append(f"{name} total squares regressed")
        if name == "overall":
            continue
        for class_name, active_class in active["per_class"].items():
            if proposed["per_class"][class_name]["correct"] < active_class["correct"]:
                reasons.append(f"{name} class {class_name} regressed")
    return not reasons, reasons


def hex_color(value: str) -> tuple[int, int, int]:
    normalized = value.removeprefix("#")
    if len(normalized) != 6:
        raise ValueError(f"Expected a six-digit hex color, got {value!r}")
    try:
        return tuple(int(normalized[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError as exc:
        raise ValueError(f"Invalid hex color: {value!r}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
