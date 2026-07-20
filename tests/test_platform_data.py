from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_scan.platform_data import (
    platform_pair_decision,
    sha256_file,
    verify_data_manifest,
)

CLASS_NAMES = ("empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")


def test_platform_gate_rejects_per_platform_class_regression() -> None:
    active = _evaluation_metrics()
    candidate = _evaluation_metrics()
    candidate["platforms"]["lichess"]["per_class"]["q"]["correct"] = 9

    passed, reasons = platform_pair_decision(active, candidate)

    assert passed is False
    assert "platform lichess class q regressed" in reasons


def test_platform_manifest_verifies_every_training_and_gate_image(tmp_path: Path) -> None:
    generated = _write_image(tmp_path / "boards" / "board.png", b"generated")
    real_board = _write_image(tmp_path / "real" / "board.jpg", b"real-board")
    real_square = _write_image(tmp_path / "real" / "square.png", b"real-square")
    source = _write_image(tmp_path / "source" / "assets.txt", b"source")

    records_path = tmp_path / "records.jsonl"
    _write_records(records_path, [_record(generated, tmp_path, split="test")])
    real_records_path = tmp_path / "real" / "records.jsonl"
    _write_records(real_records_path, [_record(real_board, tmp_path, split="train")])
    real_squares_path = tmp_path / "real" / "squares.jsonl"
    _write_records(real_squares_path, [_record(real_square, tmp_path, split="train")])
    manifest = {
        "version": "test",
        "seed": 1,
        "positions": 1,
        "test_positions": 1,
        "records": 1,
        "platforms": {"lichess": {"piece_styles": 1, "boards": 1}},
        "records_sha256": sha256_file(records_path),
        "real_records_sha256": sha256_file(real_records_path),
        "real_squares_sha256": sha256_file(real_squares_path),
        "source_files": [
            {
                "path": str(source.relative_to(tmp_path)),
                "bytes": source.stat().st_size,
                "sha256": sha256_file(source),
            }
        ],
    }
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))

    verify_data_manifest(tmp_path, manifest_path)
    generated.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="image failed verification"):
        verify_data_manifest(tmp_path, manifest_path)


def _evaluation_metrics() -> dict:
    group = {
        "exact_boards": 10,
        "non_empty_correct": 100,
        "correct": 200,
        "per_class": {name: {"correct": 10, "total": 10, "accuracy": 1.0} for name in CLASS_NAMES},
    }
    return {
        "overall": json.loads(json.dumps(group)),
        "platforms": {"lichess": json.loads(json.dumps(group))},
    }


def _write_image(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _record(path: Path, root: Path, *, split: str) -> dict[str, str]:
    return {
        "path": str(path.relative_to(root)),
        "split": split,
        "sha256": sha256_file(path),
    }


def _write_records(path: Path, records: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
