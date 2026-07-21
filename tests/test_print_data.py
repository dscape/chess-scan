from __future__ import annotations

import json
from pathlib import Path

import pytest

from chess_scan.print_data import (
    print_pair_decision,
    read_records,
    sha256_file,
    verify_data_manifest,
)


def test_print_manifest_verifies_every_rectified_board(tmp_path: Path) -> None:
    image = tmp_path / "boards" / "reference.jpg"
    image.parent.mkdir()
    image.write_bytes(b"rectified-board")
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps(_record(image, tmp_path)) + "\n")
    manifest = {
        "version": "test",
        "boards": 1,
        "groups": 1,
        "records_sha256": sha256_file(records),
        "files": [
            {
                "path": "boards/reference.jpg",
                "bytes": image.stat().st_size,
                "sha256": sha256_file(image),
            }
        ],
        "source_images_redistributed": False,
    }
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))

    verify_data_manifest(tmp_path, manifest_path)
    image.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="image failed verification"):
        verify_data_manifest(tmp_path, manifest_path)


def test_print_record_reader_rejects_unsafe_paths(tmp_path: Path) -> None:
    record = _record(tmp_path / "reference.jpg", tmp_path)
    record["path"] = "../reference.jpg"
    records = tmp_path / "records.jsonl"
    records.write_text(json.dumps(record) + "\n")

    with pytest.raises(ValueError, match="unsafe image path"):
        read_records(records)


def test_print_gate_requires_every_board_to_be_exact_and_non_regressing() -> None:
    active = _metrics(exact=0, correct=62, non_empty=19, exact_variants=2)
    fixed = _metrics(exact=1, correct=64, non_empty=21, exact_variants=8)
    regressed = _metrics(exact=0, correct=63, non_empty=20, exact_variants=7)

    assert print_pair_decision(active, fixed) == (True, [])
    passed, reasons = print_pair_decision(fixed, regressed)

    assert passed is False
    assert any("candidate is not exact" in reason for reason in reasons)
    assert any("exact boards regressed" in reason for reason in reasons)
    assert any("robustness variants regressed" in reason for reason in reasons)


def _record(image: Path, root: Path) -> dict:
    labels = [0] * 64
    labels[4] = 12
    labels[60] = 6
    return {
        "path": str(image.relative_to(root)),
        "split": "regression",
        "group": "reference",
        "sha256": sha256_file(image) if image.exists() else "0" * 64,
        "source_image_sha256": "1" * 64,
        "labels": labels,
        "fen": "4k3/8/8/8/8/8/8/4K3",
        "orientation": "white",
        "source": "consented-confirmed-feedback",
    }


def _metrics(*, exact: int, correct: int, non_empty: int, exact_variants: int) -> dict:
    return {
        "boards": 1,
        "exact_boards": exact,
        "correct_squares": correct,
        "non_empty_correct": non_empty,
        "robustness_variants": 8,
        "exact_robustness_variants": exact_variants,
        "robustness_correct_squares": correct * 8,
    }
