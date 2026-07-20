"""Verification and evaluation helpers for the external platform corpus."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

_configured_manifest = os.getenv("CHESS_SCAN_PLATFORM_MANIFEST")
DEFAULT_EXPECTED_MANIFEST = (
    Path(_configured_manifest).expanduser().resolve()
    if _configured_manifest
    else Path(__file__).resolve().parents[2] / "benchmarks" / "platform-training-corpus.json"
)
_REAL_RECORD_FILES = ("real/records.jsonl", "real/squares.jsonl")


def default_data_dir() -> Path:
    configured = os.getenv("CHESS_SCAN_PLATFORM_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "chess-scan-training" / "platforms-v1").resolve()


def load_records(data_dir: Path, *, split: str) -> list[dict[str, Any]]:
    records = _read_records(data_dir / "records.jsonl")
    selected = [record for record in records if record["split"] == split]
    if not selected:
        raise ValueError(f"No {split} records in {data_dir / 'records.jsonl'}")
    return selected


def verify_data_manifest(
    data_dir: Path,
    expected_manifest_path: Path = DEFAULT_EXPECTED_MANIFEST,
) -> None:
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
    records = _read_records(records_path)
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
        verify_record_images(data_dir, _read_records(path))

    for record in expected["source_files"]:
        path = data_dir / record["path"]
        if path.stat().st_size != record["bytes"] or sha256_file(path) != record["sha256"]:
            raise ValueError(f"External platform source failed verification: {record['path']}")


def verify_record_images(data_dir: Path, records: list[dict[str, Any]]) -> None:
    seen: set[str] = set()
    for record in records:
        relative_path = str(record["path"])
        if relative_path in seen:
            raise ValueError(f"Duplicate external platform image record: {relative_path}")
        seen.add(relative_path)
        if sha256_file(data_dir / relative_path) != record["sha256"]:
            raise ValueError(f"External platform image failed verification: {relative_path}")


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


def _read_records(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
