"""Verification helpers for the external photographed-print regression corpus."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from chess_scan.board import labels_to_board_fen, validate_labels
from chess_scan.model_artifact import is_sha256, sha256_file

_MANIFEST_NAME = "print-regression-corpus.json"


def _default_expected_manifest() -> Path:
    configured = os.getenv("CHESS_SCAN_PRINT_MANIFEST")
    if configured:
        return Path(configured).expanduser().resolve()
    source_manifest = Path(__file__).resolve().parents[2] / "benchmarks" / _MANIFEST_NAME
    if source_manifest.is_file():
        return source_manifest
    return (Path.cwd() / "benchmarks" / _MANIFEST_NAME).resolve()


DEFAULT_EXPECTED_MANIFEST = _default_expected_manifest()


def default_data_dir() -> Path:
    configured = os.getenv("CHESS_SCAN_PRINT_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "chess-scan-training" / "print-regressions-v1").resolve()


def load_records(data_dir: Path) -> list[dict[str, Any]]:
    return read_records(data_dir / "records.jsonl")


def read_records(path: Path) -> list[dict[str, Any]]:
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
        _validate_record(record, path=path, line_number=line_number)
        records.append(record)
    if not records:
        raise ValueError(f"No photographed-print regression records in {path}")
    return records


def verify_data_manifest(
    data_dir: Path,
    expected_manifest_path: Path = DEFAULT_EXPECTED_MANIFEST,
) -> dict[str, Any]:
    expected = json.loads(expected_manifest_path.read_text())
    actual = json.loads((data_dir / "MANIFEST.json").read_text())
    for field in (
        "version",
        "boards",
        "groups",
        "records_sha256",
        "files",
        "source_images_redistributed",
    ):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"External photographed-print manifest field changed: {field}")

    records_path = data_dir / "records.jsonl"
    if sha256_file(records_path) != expected["records_sha256"]:
        raise ValueError("External photographed-print records failed verification")
    records = read_records(records_path)
    if len(records) != expected["boards"]:
        raise ValueError("External photographed-print board count changed")
    if len({str(record["group"]) for record in records}) != expected["groups"]:
        raise ValueError("External photographed-print group count changed")

    expected_files = {str(record["path"]): record for record in expected["files"]}
    record_files = {str(record["path"]): record for record in records}
    if len(record_files) != len(records):
        raise ValueError("External photographed-print records contain duplicate image paths")
    if set(record_files) != set(expected_files):
        raise ValueError("External photographed-print file inventory changed")
    for relative_path, expected_file in expected_files.items():
        image_path = data_dir / relative_path
        if (
            image_path.stat().st_size != expected_file["bytes"]
            or sha256_file(image_path) != expected_file["sha256"]
            or record_files[relative_path]["sha256"] != expected_file["sha256"]
        ):
            raise ValueError(
                f"External photographed-print image failed verification: {relative_path}"
            )
    return actual


def print_pair_decision(
    active: dict[str, Any] | None,
    candidate: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if candidate["exact_boards"] != candidate["boards"]:
        reasons.append("candidate is not exact on every photographed-print regression board")
    if candidate["exact_robustness_variants"] != candidate["robustness_variants"]:
        reasons.append("candidate is not exact on every photographed-print robustness variant")
    if active is not None:
        if candidate["exact_boards"] < active["exact_boards"]:
            reasons.append("photographed-print exact boards regressed")
        if candidate["correct_squares"] < active["correct_squares"]:
            reasons.append("photographed-print square accuracy regressed")
        if candidate["non_empty_correct"] < active["non_empty_correct"]:
            reasons.append("photographed-print occupied-square accuracy regressed")
        if candidate["exact_robustness_variants"] < active["exact_robustness_variants"]:
            reasons.append("photographed-print robustness variants regressed")
        if candidate["robustness_correct_squares"] < active["robustness_correct_squares"]:
            reasons.append("photographed-print robustness square accuracy regressed")
    return not reasons, reasons


def _validate_record(record: dict[str, Any], *, path: Path, line_number: int) -> None:
    required_strings = (
        "path",
        "split",
        "group",
        "sha256",
        "source_image_sha256",
        "fen",
        "orientation",
        "source",
    )
    if any(not isinstance(record.get(field), str) for field in required_strings):
        raise ValueError(f"Record has missing metadata at {path}:{line_number}")
    relative_path = Path(record["path"])
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(f"Record has an unsafe image path at {path}:{line_number}")
    if record["split"] != "regression":
        raise ValueError(f"Record has an invalid split at {path}:{line_number}")
    if record["orientation"] not in {"white", "black"}:
        raise ValueError(f"Record has an invalid orientation at {path}:{line_number}")
    if not is_sha256(record["sha256"]) or not is_sha256(record["source_image_sha256"]):
        raise ValueError(f"Record has an invalid SHA-256 at {path}:{line_number}")
    labels = record.get("labels")
    if not isinstance(labels, list) or any(not isinstance(label, int) for label in labels):
        raise ValueError(f"Record has invalid labels at {path}:{line_number}")
    validate_labels(labels)
    if labels_to_board_fen(labels, orientation=record["orientation"]) != record["fen"]:
        raise ValueError(f"Record FEN does not match its labels at {path}:{line_number}")
