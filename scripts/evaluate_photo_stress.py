#!/usr/bin/env python3
"""Run deterministic photo, perspective, fading, and resolution QA gates."""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chess_scan.classifier import DiagramClassifier
from chess_scan.geometry import detect_board_corners, rectify_board
from qa_common import download_verified, labels_from_fen, write_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK = PROJECT_ROOT / "benchmarks" / "chess-steps-step2.json"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "chess-steps-v4.onnx"
_PRINTED_CORNERS = np.float32([[170, 116], [890, 19], [953, 812], [81, 817]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    benchmark = json.loads(args.benchmark.read_text())
    work_context = (
        nullcontext(args.cache_dir)
        if args.cache_dir is not None
        else tempfile.TemporaryDirectory(prefix="chess-scan-photo-qa-")
    )
    with work_context as temporary:
        work_dir = Path(temporary)
        work_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = work_dir / "reference.pdf"
        download_verified(benchmark["source_url"], benchmark["source_sha256"], pdf_path)
        boards = render_reference_boards(pdf_path, int(benchmark["pdf_page_index"]))
        items = pair_with_labels(boards, benchmark["positions"])
        classifier = DiagramClassifier(args.model, version=args.model.stem)
        payload = evaluate(items, classifier)

    payload["runtime_version"] = args.model.stem
    payload["benchmark"] = str(args.benchmark.relative_to(PROJECT_ROOT))
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))
    enforce_gates(payload)


def render_reference_boards(pdf_path: Path, page_index: int) -> list[np.ndarray]:
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit(
            "PyMuPDF is required for this QA command. Run with "
            "`uv run --with 'pymupdf>=1.25,<2' python scripts/evaluate_photo_stress.py`."
        ) from exc

    with fitz.open(pdf_path) as document:
        page = document[page_index]
        pixmap = page.get_pixmap(matrix=fitz.Matrix(3, 3), colorspace=fitz.csRGB, alpha=False)
        rgb = np.frombuffer(pixmap.samples, np.uint8).reshape(
            pixmap.height,
            pixmap.width,
            pixmap.n,
        )
    page_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    boards = []
    for x, y, width, height in find_board_boxes(page_bgr):
        board = page_bgr[y : y + height, x : x + width]
        boards.append(cv2.resize(board, (512, 512), interpolation=cv2.INTER_AREA))
    if len(boards) != 12:
        raise ValueError(f"Expected 12 audited boards on the reference page, found {len(boards)}")
    return boards


def find_board_boxes(page_bgr: np.ndarray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, 190, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if 300 <= width <= 600 and 300 <= height <= 600 and 0.96 <= width / height <= 1.04:
            candidates.append((x, y, width, height))

    unique: list[tuple[int, int, int, int]] = []
    for box in sorted(candidates, key=lambda item: (item[1], item[0])):
        x, y, width, height = box
        if not any(
            abs(x - prior_x) < 5
            and abs(y - prior_y) < 5
            and abs(width - prior_width) < 5
            and abs(height - prior_height) < 5
            for prior_x, prior_y, prior_width, prior_height in unique
        ):
            unique.append(box)
    return unique


def pair_with_labels(
    boards: list[np.ndarray],
    positions: list[dict[str, Any]],
) -> list[tuple[np.ndarray, list[int]]]:
    by_index = {int(position["board_index"]): position for position in positions}
    if set(by_index) != set(range(len(boards))):
        raise ValueError("Reference benchmark board indices do not match the rendered page")
    return [(board, labels_from_fen(by_index[index]["fen"])) for index, board in enumerate(boards)]


def evaluate(
    items: list[tuple[np.ndarray, list[int]]],
    classifier: DiagramClassifier,
) -> dict[str, Any]:
    halftone_screen = make_halftone_screen(1024)
    classifier_variants: dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "clean": lambda board: board,
        "clean_256px": lambda board: resize_round_trip(board, 256),
        "clean_128px": lambda board: resize_round_trip(board, 128),
        "faded_256px": lambda board: resize_round_trip(board, 256, contrast=0.35),
        "printed_halftone": lambda board: rectify_printed_photo(board, halftone_screen),
    }
    classifier_results = {
        name: evaluate_classifier(items, classifier, transform)
        for name, transform in classifier_variants.items()
    }
    pipeline_results = {
        "moderate_perspective": evaluate_pipeline(
            items,
            classifier,
            lambda board, index: make_photo(board, seed=index, severe=False),
        ),
        "severe_perspective_blur": evaluate_pipeline(
            items,
            classifier,
            lambda board, index: make_photo(board, seed=index, severe=True),
        ),
        "printed_halftone": evaluate_pipeline(
            items,
            classifier,
            lambda board, _index: make_printed_photo(board, halftone_screen),
        ),
    }
    return {
        "boards": len(items),
        "classifier": classifier_results,
        "pipeline": pipeline_results,
    }


def evaluate_classifier(
    items: list[tuple[np.ndarray, list[int]]],
    classifier: DiagramClassifier,
    transform: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    exact_boards = 0
    exact_squares = 0
    failures = []
    for index, (board, expected) in enumerate(items):
        predicted = classifier.predict(transform(board)).labels
        correct_squares = sum(a == b for a, b in zip(expected, predicted, strict=True))
        exact_boards += correct_squares == 64
        exact_squares += correct_squares
        if correct_squares != 64:
            failures.append(failure_details(index, expected, predicted))
    return {
        "exact_boards": exact_boards,
        "exact_squares": exact_squares,
        "total_squares": len(items) * 64,
        "failures": failures,
    }


def evaluate_pipeline(
    items: list[tuple[np.ndarray, list[int]]],
    classifier: DiagramClassifier,
    transform: Callable[[np.ndarray, int], np.ndarray],
) -> dict[str, Any]:
    detected_boards = 0
    exact_boards = 0
    exact_squares = 0
    methods: dict[str, int] = {}
    failures = []
    for index, (board, expected) in enumerate(items):
        photo = transform(board, index)
        detection = detect_board_corners(photo)
        methods[detection.method] = methods.get(detection.method, 0) + 1
        if detection.method == "manual_adjustment_needed":
            continue
        detected_boards += 1
        rectified = rectify_board(photo, detection.corners)
        predicted = classifier.predict(rectified).labels
        correct_squares = sum(a == b for a, b in zip(expected, predicted, strict=True))
        exact_boards += correct_squares == 64
        exact_squares += correct_squares
        if correct_squares != 64:
            failures.append(failure_details(index, expected, predicted))
    return {
        "detected_boards": detected_boards,
        "exact_boards": exact_boards,
        "exact_squares": exact_squares,
        "total_squares": len(items) * 64,
        "detection_methods": methods,
        "failures": failures,
    }


def failure_details(index: int, expected: list[int], predicted: list[int]) -> dict[str, Any]:
    return {
        "board_index": index,
        "mismatched_squares": [
            {
                "index": square_index,
                "expected": expected_label,
                "predicted": predicted_label,
            }
            for square_index, (expected_label, predicted_label) in enumerate(
                zip(expected, predicted, strict=True)
            )
            if expected_label != predicted_label
        ],
    }


def resize_round_trip(board: np.ndarray, size: int, *, contrast: float = 1.0) -> np.ndarray:
    resized = cv2.resize(board, (size, size), interpolation=cv2.INTER_AREA)
    resized = cv2.resize(resized, (512, 512), interpolation=cv2.INTER_CUBIC)
    if contrast == 1.0:
        return resized
    return np.clip(220 + (resized.astype(np.float32) - 220) * contrast, 0, 255).astype(np.uint8)


def make_halftone_screen(size: int) -> np.ndarray:
    positions = np.arange(size, dtype=np.float32)
    wave = np.sin(2 * np.pi * positions / 5.0)
    return (wave[:, None] + wave[None, :] + 2.0) / 4.0


def make_printed_photo(board: np.ndarray, screen: np.ndarray) -> np.ndarray:
    height, width = screen.shape
    high_resolution = cv2.resize(board, (width, height), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(high_resolution, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    halftone = ((gray > screen) * 255).astype(np.uint8)
    halftone = cv2.GaussianBlur(halftone, (0, 0), 1.0)
    printed = cv2.cvtColor(halftone, cv2.COLOR_GRAY2BGR)

    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(source, _PRINTED_CORNERS)
    photo = cv2.warpPerspective(
        printed,
        transform,
        (1095, 900),
        flags=cv2.INTER_CUBIC,
        borderValue=(235, 235, 235),
    )
    return photo


def rectify_printed_photo(board: np.ndarray, screen: np.ndarray) -> np.ndarray:
    return rectify_board(make_printed_photo(board, screen), _PRINTED_CORNERS)


def make_photo(board: np.ndarray, *, seed: int, severe: bool) -> np.ndarray:
    random = np.random.RandomState(seed)
    canvas = np.full((760, 760, 3), random.randint(205, 245), dtype=np.uint8)
    source = np.float32([[0, 0], [511, 0], [511, 511], [0, 511]])
    jitter = 95 if severe else 45
    base = np.float32([[105, 90], [655, 110], [645, 660], [90, 640]])
    destination = base + random.uniform(-jitter, jitter, (4, 2)).astype(np.float32)
    homography = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(board, homography, (760, 760), borderValue=(235, 235, 235))
    mask = cv2.warpPerspective(np.full((512, 512), 255, np.uint8), homography, (760, 760))
    canvas[mask > 0] = warped[mask > 0]
    if severe:
        canvas = cv2.GaussianBlur(canvas, (5, 5), 1)
        canvas = cv2.resize(canvas, (380, 380), interpolation=cv2.INTER_AREA)
        canvas = cv2.resize(canvas, (760, 760), interpolation=cv2.INTER_CUBIC)
    return canvas


def enforce_gates(payload: dict[str, Any]) -> None:
    classifier = payload["classifier"]
    pipeline = payload["pipeline"]
    passed = (
        payload["boards"] == 12
        and classifier["clean"]["exact_boards"] == 12
        and classifier["clean_256px"]["exact_boards"] == 12
        and classifier["faded_256px"]["exact_boards"] == 12
        and classifier["printed_halftone"]["exact_boards"] == 12
        and pipeline["moderate_perspective"]["detected_boards"] == 12
        and pipeline["moderate_perspective"]["exact_boards"] == 12
        and pipeline["severe_perspective_blur"]["detected_boards"] == 12
        and pipeline["severe_perspective_blur"]["exact_boards"] == 12
        and pipeline["printed_halftone"]["detected_boards"] == 12
    )
    if not passed:
        raise SystemExit("Photo stress gate failed")


if __name__ == "__main__":
    main()
