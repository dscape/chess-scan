#!/usr/bin/env python3
"""Evaluate a model on grouped external platform screenshots."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chess_scan.argus_data import classification_metrics
from chess_scan.classifier import DiagramClassifier, preprocess_board
from chess_scan.platform_data import load_records, platform_pair_decision, verify_data_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "chess-steps-v4.onnx"
DEFAULT_DATA_DIR = Path.home() / "chess-scan-training" / "platforms-v1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--variant",
        choices=("clean", "camera", "faded", "low-resolution"),
        default="clean",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    verify_data_manifest(data_dir)
    records = load_records(data_dir, split=args.split)
    classifier = DiagramClassifier(args.model, version=args.model.stem)
    candidate = evaluate_records(classifier, data_dir, records, variant=args.variant)
    candidate["model"] = args.model.stem
    if args.baseline:
        baseline_classifier = DiagramClassifier(args.baseline, version=args.baseline.stem)
        baseline = evaluate_records(
            baseline_classifier,
            data_dir,
            records,
            variant=args.variant,
        )
        baseline["model"] = args.baseline.stem
        passed, reasons = platform_pair_decision(baseline, candidate)
        payload = {
            "passed": passed,
            "reasons": reasons,
            "baseline": baseline,
            "candidate": candidate,
        }
    else:
        passed = True
        payload = candidate
    payload.update(
        {
            "split": args.split,
            "variant": args.variant,
            "dataset_version": json.loads((data_dir / "MANIFEST.json").read_text())["version"],
        }
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


def evaluate_records(
    classifier: DiagramClassifier,
    data_dir: Path,
    records: list[dict[str, Any]],
    *,
    board_batch_size: int = 16,
    variant: str = "clean",
) -> dict[str, Any]:
    expected: list[np.ndarray] = []
    predicted: list[np.ndarray] = []
    for offset in range(0, len(records), board_batch_size):
        batch = records[offset : offset + board_batch_size]
        inputs = []
        for batch_index, record in enumerate(batch):
            board = cv2.imread(str(data_dir / record["path"]), cv2.IMREAD_COLOR)
            if board is None:
                raise ValueError(f"Cannot read platform board: {record['path']}")
            record_index = offset + batch_index
            inputs.append(preprocess_board(transform_board(board, variant, record_index)))
        predictions = classifier.predict_preprocessed(np.concatenate(inputs))
        expected.extend(np.asarray(record["labels"], dtype=np.int64) for record in batch)
        predicted.extend(
            np.asarray(prediction.labels, dtype=np.int64) for prediction in predictions
        )

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


def transform_board(board: np.ndarray, variant: str, index: int) -> np.ndarray:
    if variant == "clean":
        return board
    if variant == "low-resolution":
        reduced = cv2.resize(board, (128, 128), interpolation=cv2.INTER_AREA)
        return cv2.resize(reduced, (512, 512), interpolation=cv2.INTER_CUBIC)
    if variant == "faded":
        hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= 0.3
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.8 + 35, 0, 255)
        faded = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return cv2.GaussianBlur(faded, (3, 3), 0.45)
    if variant != "camera":
        raise ValueError(f"Unknown platform variant: {variant}")

    rng = np.random.RandomState(10_000 + index)
    source = np.float32([[0, 0], [511, 0], [511, 511], [0, 511]])
    margin = 62
    corners = np.float32(
        [
            [margin + rng.uniform(-25, 25), margin + rng.uniform(-25, 25)],
            [578 + rng.uniform(-25, 25), margin + rng.uniform(-25, 25)],
            [578 + rng.uniform(-25, 25), 578 + rng.uniform(-25, 25)],
            [margin + rng.uniform(-25, 25), 578 + rng.uniform(-25, 25)],
        ]
    )
    photographed = cv2.warpPerspective(
        board,
        cv2.getPerspectiveTransform(source, corners),
        (640, 640),
        borderValue=(35, 35, 35),
    )
    x = np.linspace(0, np.pi * rng.uniform(8, 20), 640)
    photographed = np.clip(
        photographed.astype(np.float32) + np.sin(x)[None, :, None] * rng.uniform(2, 7),
        0,
        255,
    ).astype(np.uint8)
    photographed = cv2.GaussianBlur(photographed, (3, 3), rng.uniform(0.2, 0.8))
    _, encoded = cv2.imencode(".jpg", photographed, [cv2.IMWRITE_JPEG_QUALITY, 65])
    photographed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    return cv2.warpPerspective(
        photographed,
        cv2.getPerspectiveTransform(corners, source),
        (512, 512),
    )


def board_metrics(
    indices: list[int],
    expected: list[np.ndarray],
    predicted: list[np.ndarray],
) -> dict[str, Any]:
    wanted = np.concatenate([expected[index] for index in indices])
    actual = np.concatenate([predicted[index] for index in indices])
    metrics = classification_metrics(actual, wanted)
    correct_by_board = [int((predicted[index] == expected[index]).sum()) for index in indices]
    metrics.update(
        {
            "boards": len(indices),
            "exact_boards": sum(correct == 64 for correct in correct_by_board),
            "exact_board_accuracy": sum(correct == 64 for correct in correct_by_board)
            / len(indices),
            "worst_board_correct_squares": min(correct_by_board),
        }
    )
    return metrics


if __name__ == "__main__":
    main()
