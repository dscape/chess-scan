#!/usr/bin/env python3
"""Evaluate models on consented, non-redistributed photographed-print regressions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chess_scan.board import CLASS_NAMES
from chess_scan.classifier import DiagramClassifier
from chess_scan.model_artifact import model_version
from chess_scan.print_data import (
    default_data_dir,
    load_records,
    print_pair_decision,
    verify_data_manifest,
)
from image_augmentation import jpeg_round_trip, resize_round_trip
from qa_common import write_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "chess-steps-v5.onnx"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    manifest = verify_data_manifest(data_dir)
    records = load_records(data_dir)
    candidate = evaluate_model(args.model, data_dir, records)
    active = evaluate_model(args.baseline, data_dir, records) if args.baseline else None
    passed, reasons = print_pair_decision(active, candidate)
    payload = {
        "passed": passed,
        "reasons": reasons,
        "dataset_version": manifest["version"],
        "active": active,
        "candidate": candidate,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit("Photographed-print regression gate failed")


def evaluate_model(
    model_path: Path,
    data_dir: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    classifier = DiagramClassifier(model_path, version=model_version(model_path))
    exact_boards = 0
    correct_squares = 0
    non_empty_correct = 0
    non_empty_total = 0
    exact_variants = 0
    variant_correct_squares = 0
    failures = []
    variant_failures = []
    for record in records:
        board = cv2.imread(str(data_dir / record["path"]), cv2.IMREAD_COLOR)
        if board is None:
            raise ValueError(f"Cannot read photographed-print board: {record['path']}")
        expected = [int(label) for label in record["labels"]]
        for variant, transformed in _robustness_variants(board).items():
            prediction = classifier.predict(transformed)
            mismatches = _mismatches(expected, prediction.labels)
            variant_correct_squares += 64 - len(mismatches)
            exact_variants += not mismatches
            if variant == "original":
                correct_squares += 64 - len(mismatches)
                non_empty_total += sum(label != 0 for label in expected)
                non_empty_correct += sum(
                    wanted != 0 and wanted == actual
                    for wanted, actual in zip(expected, prediction.labels, strict=True)
                )
                exact_boards += not mismatches
                if mismatches:
                    failures.append({"group": record["group"], "mismatches": mismatches})
            if mismatches:
                variant_failures.append(
                    {
                        "group": record["group"],
                        "variant": variant,
                        "mismatches": mismatches,
                    }
                )
    return {
        "model": classifier.version,
        "boards": len(records),
        "exact_boards": exact_boards,
        "correct_squares": correct_squares,
        "total_squares": len(records) * 64,
        "non_empty_correct": non_empty_correct,
        "non_empty_total": non_empty_total,
        "robustness_variants": len(records) * 8,
        "exact_robustness_variants": exact_variants,
        "robustness_correct_squares": variant_correct_squares,
        "robustness_total_squares": len(records) * 8 * 64,
        "failures": failures,
        "robustness_failures": variant_failures,
    }


def _robustness_variants(board: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "original": board,
        "jpeg-55": jpeg_round_trip(board, 55),
        "jpeg-75": jpeg_round_trip(board, 75),
        "blur-0.8": cv2.GaussianBlur(board, (0, 0), 0.8),
        "blur-1.4": cv2.GaussianBlur(board, (0, 0), 1.4),
        "resize-128": resize_round_trip(board, 128),
        "faded": np.clip(board.astype(np.float32) * 0.55 + 85, 0, 255).astype(np.uint8),
        "dark": np.clip(board.astype(np.float32) * 0.65, 0, 255).astype(np.uint8),
    }


def _mismatches(expected: list[int], predicted: list[int]) -> list[dict[str, Any]]:
    return [
        {
            "index": index,
            "expected": CLASS_NAMES[wanted],
            "predicted": CLASS_NAMES[actual],
        }
        for index, (wanted, actual) in enumerate(zip(expected, predicted, strict=True))
        if wanted != actual
    ]


if __name__ == "__main__":
    main()
