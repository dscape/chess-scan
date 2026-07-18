#!/usr/bin/env python3
"""Evaluate official interactive and German-manual Chess Steps examples."""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from chess_scan.board import CLASS_NAMES
from chess_scan.classifier import DiagramClassifier
from evaluate_photo_stress import find_board_boxes
from qa_common import download_verified, labels_from_fen, write_json

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "benchmarks" / "chess-steps-online-sources.json"
DEFAULT_MANUAL_BENCHMARK = PROJECT_ROOT / "benchmarks" / "chess-steps-german-manuals.json"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "chess-steps-v2.onnx"
PIECE_ASSET_IDS = {
    "K": "c40wk",
    "Q": "c40wq",
    "R": "c40wr",
    "B": "c40wb",
    "N": "c40wn",
    "P": "c40wp",
    "k": "c40bk",
    "q": "c40bq",
    "r": "c40br",
    "b": "c40bb",
    "n": "c40bn",
    "p": "c40bp",
}
FEN_PATTERN = re.compile(r'"([^"_]+ [wb] [^"_]+)_')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--manual-benchmark", type=Path, default=DEFAULT_MANUAL_BENCHMARK)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text())
    position_sources = [
        source
        for source in manifest["evaluated_sources"]
        if source["kind"] == "interactive_position_set"
    ]
    asset_sources = {asset["id"]: asset for asset in manifest["interactive_render_assets"]}
    source_inventory = {source["id"]: source for source in manifest["evaluated_sources"]}
    manual_benchmark = json.loads(args.manual_benchmark.read_text())
    verify_manual_inventory(manual_benchmark, source_inventory)

    work_context = (
        nullcontext(args.cache_dir)
        if args.cache_dir is not None
        else tempfile.TemporaryDirectory(prefix="chess-scan-online-qa-")
    )
    with work_context as temporary:
        work_dir = Path(temporary)
        work_dir.mkdir(parents=True, exist_ok=True)
        classifier = DiagramClassifier(args.model, version=args.model.stem)
        assets = download_assets(asset_sources, work_dir / "assets")
        interactive_results = [
            evaluate_interactive_source(source, work_dir, assets, classifier)
            for source in position_sources
        ]
        manual_results = evaluate_manual_sources(
            manual_benchmark,
            source_inventory,
            work_dir,
            classifier,
        )

    interactive = summarize_results(interactive_results)
    manuals = summarize_results(manual_results)
    combined = summarize_results(interactive_results + manual_results)
    payload: dict[str, Any] = {
        "runtime_version": args.model.stem,
        "interactive": interactive,
        "german_manuals": manuals,
        "combined": combined,
        "source_results": interactive_results + manual_results,
    }
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))

    if not (
        interactive["sources"] == 17
        and interactive["boards"] == interactive["exact_boards"] == 204
        and manuals["sources"] == 6
        and manuals["boards"] == manuals["exact_boards"] == 63
        and combined["boards"] == combined["exact_king_positions"] == 267
        and combined["exact_squares"] == combined["total_squares"] == 267 * 64
    ):
        raise SystemExit("Official online example gate failed")


def download_assets(
    sources: dict[str, dict[str, Any]],
    output_dir: Path,
) -> dict[str, np.ndarray]:
    assets: dict[str, np.ndarray] = {}
    for symbol, asset_id in PIECE_ASSET_IDS.items():
        source = sources[asset_id]
        path = output_dir / f"{asset_id}.png"
        download_verified(source["url"], source["sha256"], path)
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.shape[:2] != (40, 40):
            raise ValueError(f"Invalid official piece asset: {source['url']}")
        assets[symbol] = image
    return assets


def evaluate_interactive_source(
    source: dict[str, Any],
    work_dir: Path,
    assets: dict[str, np.ndarray],
    classifier: DiagramClassifier,
) -> dict[str, Any]:
    path = work_dir / f"{source['id']}.js"
    download_verified(source["url"], source["sha256"], path)
    fens = FEN_PATTERN.findall(path.read_text(encoding="latin-1"))
    expected_positions = int(source["positions"])
    if len(fens) != expected_positions:
        raise ValueError(
            f"Expected {expected_positions} positions in {source['url']}, found {len(fens)}"
        )

    exact_boards = 0
    exact_squares = 0
    exact_kings = 0
    failures: list[dict[str, Any]] = []
    for index, fen in enumerate(fens):
        board, expected = render_official_board(fen, assets)
        predicted = classifier.predict(board).labels
        correct_squares = sum(a == b for a, b in zip(expected, predicted, strict=True))
        board_exact = correct_squares == 64
        king_exact = exact_king_locations(expected, predicted)
        exact_boards += board_exact
        exact_squares += correct_squares
        exact_kings += king_exact
        if not board_exact:
            failures.append(
                {
                    "index": index,
                    "fen": fen,
                    "correct_squares": correct_squares,
                    "predicted": predicted,
                }
            )

    return {
        "source_id": source["id"],
        "boards": len(fens),
        "exact_boards": exact_boards,
        "exact_squares": exact_squares,
        "exact_king_positions": exact_kings,
        "failures": failures,
    }


def verify_manual_inventory(
    benchmark: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
) -> None:
    for benchmark_source in benchmark["sources"]:
        source = inventory.get(benchmark_source["id"])
        if source is None:
            raise ValueError(
                f"Manual source is missing from online inventory: {benchmark_source['id']}"
            )
        for field in ("url", "sha256", "bytes"):
            if source[field] != benchmark_source[field]:
                raise ValueError(
                    f"Manual source inventory mismatch for {benchmark_source['id']} field {field}"
                )


def evaluate_manual_sources(
    benchmark: dict[str, Any],
    inventory: dict[str, dict[str, Any]],
    work_dir: Path,
    classifier: DiagramClassifier,
) -> list[dict[str, Any]]:
    diagrams_by_source: dict[str, list[dict[str, Any]]] = {}
    for diagram in benchmark["diagrams"]:
        diagrams_by_source.setdefault(diagram["source_id"], []).append(diagram)

    results = []
    for source_id, diagrams in diagrams_by_source.items():
        source = inventory[source_id]
        pdf_path = work_dir / f"{source_id}.pdf"
        download_verified(source["url"], source["sha256"], pdf_path)
        boards = render_pdf_diagrams(pdf_path, diagrams)
        exact_boards = 0
        exact_squares = 0
        exact_kings = 0
        failures = []
        for diagram in diagrams:
            locator = (int(diagram["pdf_page_index"]), int(diagram["board_index"]))
            expected = labels_from_fen(diagram["fen"])
            predicted = classifier.predict(boards[locator]).labels
            correct_squares = sum(a == b for a, b in zip(expected, predicted, strict=True))
            exact_boards += correct_squares == 64
            exact_squares += correct_squares
            exact_kings += exact_king_locations(expected, predicted)
            if correct_squares != 64:
                failures.append(
                    {
                        "pdf_page_index": locator[0],
                        "board_index": locator[1],
                        "fen": diagram["fen"],
                        "correct_squares": correct_squares,
                        "predicted": predicted,
                    }
                )
        results.append(
            {
                "source_id": source_id,
                "boards": len(diagrams),
                "exact_boards": exact_boards,
                "exact_squares": exact_squares,
                "exact_king_positions": exact_kings,
                "failures": failures,
            }
        )
    return results


def render_pdf_diagrams(
    pdf_path: Path,
    diagrams: list[dict[str, Any]],
) -> dict[tuple[int, int], np.ndarray]:
    try:
        import fitz
    except ImportError as exc:
        raise SystemExit(
            "PyMuPDF is required for this QA command. Run with "
            "`uv run --with 'pymupdf>=1.25,<2' python scripts/evaluate_online_examples.py`."
        ) from exc

    requested_pages = {int(diagram["pdf_page_index"]) for diagram in diagrams}
    requested_locators = {
        (int(diagram["pdf_page_index"]), int(diagram["board_index"])) for diagram in diagrams
    }
    boards: dict[tuple[int, int], np.ndarray] = {}
    with fitz.open(pdf_path) as document:
        for page_index in requested_pages:
            page = document[page_index]
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(3, 3),
                colorspace=fitz.csRGB,
                alpha=False,
            )
            rgb = np.frombuffer(pixmap.samples, np.uint8).reshape(
                pixmap.height,
                pixmap.width,
                pixmap.n,
            )
            page_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            for board_index, (x, y, width, height) in enumerate(find_board_boxes(page_bgr)):
                locator = (page_index, board_index)
                if locator not in requested_locators:
                    continue
                board = page_bgr[y : y + height, x : x + width]
                boards[locator] = cv2.resize(board, (512, 512), interpolation=cv2.INTER_AREA)
    missing = requested_locators - boards.keys()
    if missing:
        raise ValueError(f"Official manual diagrams were not found at: {sorted(missing)}")
    return boards


def summarize_results(results: list[dict[str, Any]]) -> dict[str, int]:
    boards = sum(result["boards"] for result in results)
    return {
        "sources": len(results),
        "boards": boards,
        "exact_boards": sum(result["exact_boards"] for result in results),
        "exact_squares": sum(result["exact_squares"] for result in results),
        "total_squares": boards * 64,
        "exact_king_positions": sum(result["exact_king_positions"] for result in results),
    }


def render_official_board(
    fen: str,
    assets: dict[str, np.ndarray],
) -> tuple[np.ndarray, list[int]]:
    labels = labels_from_fen(fen)
    board = np.empty((320, 320, 3), dtype=np.uint8)
    for index, label in enumerate(labels):
        row, column = divmod(index, 8)
        square = board[row * 40 : (row + 1) * 40, column * 40 : (column + 1) * 40]
        square[:] = 255 if (row + column) % 2 == 0 else 192
        symbol = CLASS_NAMES[label]
        if symbol != "empty":
            composite_asset(square, assets[symbol])
    return board, labels


def composite_asset(square: np.ndarray, asset: np.ndarray) -> None:
    if asset.ndim != 3 or asset.shape[2] not in (3, 4):
        raise ValueError("Expected a three- or four-channel piece asset")
    if asset.shape[2] == 3:
        square[:] = asset
        return
    alpha = asset[:, :, 3:4].astype(np.float32) / 255.0
    square[:] = np.rint(asset[:, :, :3] * alpha + square * (1.0 - alpha)).astype(np.uint8)


def exact_king_locations(expected: list[int], predicted: list[int]) -> bool:
    if expected.count(6) != 1 or expected.count(12) != 1:
        return False
    expected_kings = {index for index, label in enumerate(expected) if label in (6, 12)}
    predicted_kings = {index for index, label in enumerate(predicted) if label in (6, 12)}
    return expected_kings == predicted_kings and all(
        expected[index] == predicted[index] for index in expected_kings
    )


if __name__ == "__main__":
    main()
