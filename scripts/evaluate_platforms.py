#!/usr/bin/env python3
"""Evaluate models on grouped external platform screenshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chess_scan.classifier import DiagramClassifier, preprocess_board
from chess_scan.model_artifact import model_version
from chess_scan.platform_data import (
    default_data_dir,
    load_records,
    platform_pair_decision,
    summarize_platform_predictions,
    verify_data_manifest,
)
from image_augmentation import jpeg_round_trip, resize_round_trip
from qa_common import write_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "models" / "chess-steps-v4.onnx"
_VARIANTS = ("clean", "camera", "faded", "low-resolution")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--data-dir", type=Path, default=default_data_dir())
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        choices=_VARIANTS,
        help="Variant to evaluate; repeat to evaluate multiple variants in one corpus pass",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    manifest = verify_data_manifest(data_dir)
    records = load_records(data_dir, split=args.split)
    variants = tuple(dict.fromkeys(args.variants or ["clean"]))
    candidate = DiagramClassifier(args.model, version=model_version(args.model))
    baseline = (
        DiagramClassifier(args.baseline, version=model_version(args.baseline))
        if args.baseline
        else None
    )

    evaluations = evaluate_variants(
        candidate,
        baseline,
        data_dir,
        records,
        variants=variants,
        split=args.split,
        dataset_version=manifest["version"],
    )
    payload = {
        "passed": all(evaluation["passed"] for evaluation in evaluations.values()),
        "split": args.split,
        "dataset_version": manifest["version"],
        "variants": evaluations,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


def evaluate_variants(
    candidate: DiagramClassifier,
    baseline: DiagramClassifier | None,
    data_dir: Path,
    records: list[dict[str, Any]],
    *,
    variants: tuple[str, ...],
    split: str,
    dataset_version: str,
    board_batch_size: int = 4,
) -> dict[str, dict[str, Any]]:
    expected = [np.asarray(record["labels"], dtype=np.int64) for record in records]
    candidate_predictions: dict[str, list[np.ndarray]] = {variant: [] for variant in variants}
    baseline_predictions: dict[str, list[np.ndarray]] = {variant: [] for variant in variants}
    for offset in range(0, len(records), board_batch_size):
        batch_records = records[offset : offset + board_batch_size]
        boards = []
        for record in batch_records:
            board = cv2.imread(str(data_dir / record["path"]), cv2.IMREAD_COLOR)
            if board is None:
                raise ValueError(f"Cannot read platform board: {record['path']}")
            boards.append(board)
        for variant in variants:
            inputs = np.concatenate(
                [
                    preprocess_board(transform_board(board, variant, offset + index))
                    for index, board in enumerate(boards)
                ]
            )
            candidate_predictions[variant].extend(
                np.asarray(prediction.labels, dtype=np.int64)
                for prediction in candidate.predict_preprocessed(inputs)
            )
            if baseline is not None:
                baseline_predictions[variant].extend(
                    np.asarray(prediction.labels, dtype=np.int64)
                    for prediction in baseline.predict_preprocessed(inputs)
                )

    return {
        variant: build_evaluation(
            candidate,
            baseline,
            records,
            expected,
            candidate_predictions[variant],
            baseline_predictions[variant],
            variant=variant,
            split=split,
            dataset_version=dataset_version,
        )
        for variant in variants
    }


def build_evaluation(
    candidate: DiagramClassifier,
    baseline: DiagramClassifier | None,
    records: list[dict[str, Any]],
    expected: list[np.ndarray],
    candidate_predictions: list[np.ndarray],
    baseline_predictions: list[np.ndarray],
    *,
    variant: str,
    split: str,
    dataset_version: str,
) -> dict[str, Any]:
    candidate_metrics = summarize_platform_predictions(records, expected, candidate_predictions)
    candidate_metrics["model"] = candidate.version
    if baseline is None:
        payload: dict[str, Any] = {"passed": True, "candidate": candidate_metrics}
    else:
        baseline_metrics = summarize_platform_predictions(records, expected, baseline_predictions)
        baseline_metrics["model"] = baseline.version
        passed, reasons = platform_pair_decision(baseline_metrics, candidate_metrics)
        payload = {
            "passed": passed,
            "reasons": reasons,
            "baseline": baseline_metrics,
            "candidate": candidate_metrics,
        }
    payload.update(
        {
            "split": split,
            "variant": variant,
            "dataset_version": dataset_version,
        }
    )
    return payload


def transform_board(board: np.ndarray, variant: str, index: int) -> np.ndarray:
    if variant == "clean":
        return board
    if variant == "low-resolution":
        return resize_round_trip(board, 128)
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
    photographed = jpeg_round_trip(photographed, 65)
    return cv2.warpPerspective(
        photographed,
        cv2.getPerspectiveTransform(corners, source),
        (512, 512),
    )


if __name__ == "__main__":
    main()
