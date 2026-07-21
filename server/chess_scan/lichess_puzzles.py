"""Verification and scoring for the external Lichess puzzle-theme benchmark."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import chess

from chess_scan.model_artifact import sha256_file
from chess_scan.review_detectors import ReviewContext, build_analyzed_line, teaching_subjects
from chess_scan.review_themes import SUPPORTED_LICHESS_THEMES, detect_solution_theme_modes

_MANIFEST_NAME = "lichess-puzzle-corpus.json"
_SPLITS = ("development", "validation")
_PRODUCTION_THEME_BY_HANDLER = {
    "double_attack": "fork",
    "pin": "pin",
    "eliminate_defence": "capturingDefender",
    "discovered_attack": "discoveredAttack",
    "xray": "xRayAttack",
    "intermediate_move": "intermezzo",
    "trapping": "trappedPiece",
    "interference": "interference",
    "luring": "attraction",
    "magnet": "attraction",
    "clearing": "clearance",
}


def _default_expected_manifest() -> Path:
    configured = os.getenv("CHESS_SCAN_LICHESS_PUZZLE_MANIFEST")
    if configured:
        return Path(configured).expanduser().resolve()
    source_manifest = Path(__file__).resolve().parents[2] / "benchmarks" / _MANIFEST_NAME
    if source_manifest.is_file():
        return source_manifest
    return (Path.cwd() / "benchmarks" / _MANIFEST_NAME).resolve()


DEFAULT_EXPECTED_MANIFEST = _default_expected_manifest()


@dataclass(frozen=True, slots=True)
class LichessPuzzle:
    puzzle_id: str
    fen: str
    moves: tuple[str, ...]
    rating: int
    rating_deviation: int
    popularity: int
    plays: int
    themes: frozenset[str]
    game_url: str
    opening_tags: tuple[str, ...]
    benchmark_theme: str

    @property
    def game_id(self) -> str:
        return lichess_game_id(self.game_url)

    def analysis_input(
        self,
    ) -> tuple[chess.Board, tuple[str, ...], tuple[chess.Board, chess.Move]]:
        source = chess.Board(self.fen)
        setup = chess.Move.from_uci(self.moves[0])
        if setup not in source.legal_moves:
            raise ValueError(f"Puzzle {self.puzzle_id} has an illegal setup move")
        presented = source.copy(stack=False)
        presented.push(setup)
        return presented, self.moves[1:], (source, setup)


def default_data_dir() -> Path:
    configured = os.getenv("CHESS_SCAN_LICHESS_PUZZLE_DATA_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / "chess-scan-training" / "lichess-puzzles-2026-07-05").resolve()


def load_puzzles(data_dir: Path, *, split: str) -> list[LichessPuzzle]:
    if split not in _SPLITS:
        raise ValueError(f"Unknown Lichess puzzle split: {split}")
    path = data_dir / f"{split}.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != _expected_fields():
            raise ValueError(f"Unexpected Lichess puzzle columns in {path}")
        puzzles = [
            _parse_puzzle(row, path=path, line_number=line_number)
            for line_number, row in enumerate(reader, start=2)
        ]
    if not puzzles:
        raise ValueError(f"No Lichess puzzles found in {path}")
    if len({puzzle.puzzle_id for puzzle in puzzles}) != len(puzzles):
        raise ValueError(f"Duplicate Lichess puzzle IDs in {path}")
    return puzzles


def verify_data_manifest(
    data_dir: Path,
    expected_manifest_path: Path = DEFAULT_EXPECTED_MANIFEST,
) -> dict[str, Any]:
    expected = json.loads(expected_manifest_path.read_text())
    actual = json.loads((data_dir / "MANIFEST.json").read_text())
    for field in (
        "version",
        "source",
        "selection",
        "themes",
        "splits",
    ):
        if actual.get(field) != expected.get(field):
            raise ValueError(f"External Lichess puzzle manifest field changed: {field}")

    all_ids: set[str] = set()
    all_game_ids: set[str] = set()
    for split in _SPLITS:
        split_manifest = expected["splits"][split]
        path = data_dir / split_manifest["path"]
        if path.stat().st_size != split_manifest["bytes"]:
            raise ValueError(f"External Lichess puzzle byte size changed: {split}")
        if sha256_file(path) != split_manifest["sha256"]:
            raise ValueError(f"External Lichess puzzle records failed verification: {split}")
        puzzles = load_puzzles(data_dir, split=split)
        if len(puzzles) != split_manifest["puzzles"]:
            raise ValueError(f"External Lichess puzzle count changed: {split}")
        counts = Counter(puzzle.benchmark_theme for puzzle in puzzles)
        if dict(sorted(counts.items())) != split_manifest["theme_counts"]:
            raise ValueError(f"External Lichess puzzle theme balance changed: {split}")
        ids = {puzzle.puzzle_id for puzzle in puzzles}
        game_ids = {puzzle.game_id for puzzle in puzzles}
        if len(game_ids) != len(puzzles):
            raise ValueError(f"External Lichess puzzle split repeats a game: {split}")
        if ids & all_ids or game_ids & all_game_ids:
            raise ValueError("External Lichess puzzle splits overlap")
        all_ids.update(ids)
        all_game_ids.update(game_ids)
    return actual


def evaluate_theme_agreement(puzzles: list[LichessPuzzle]) -> dict[str, Any]:
    target_total: Counter[str] = Counter()
    target_matched: Counter[str] = Counter()
    predictions: Counter[str] = Counter()
    true_predictions: Counter[str] = Counter()
    misses: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    false_positives: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    any_overlap = 0
    history_free_target_matched: Counter[str] = Counter()
    history_free_predictions: Counter[str] = Counter()
    history_free_true_predictions: Counter[str] = Counter()
    production_target_matched: Counter[str] = Counter()
    production_predictions: Counter[str] = Counter()
    production_true_predictions: Counter[str] = Counter()
    production_misses: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    production_false_positives: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)

    for puzzle in puzzles:
        board, moves, previous = puzzle.analysis_input()
        contextual_evidence, history_free_evidence = detect_solution_theme_modes(
            board,
            moves,
            previous=previous,
        )
        detected = {item.theme for item in contextual_evidence}
        history_free_detected = {item.theme for item in history_free_evidence}
        context = ReviewContext(board, build_analyzed_line(board, list(moves)))
        production_detected = {
            theme
            for finding in teaching_subjects(
                context,
                theme_evidence=history_free_evidence,
            )[:3]
            if (theme := _PRODUCTION_THEME_BY_HANDLER.get(finding.handler)) is not None
        }
        target = puzzle.benchmark_theme
        target_total[target] += 1
        target_matched[target] += target in detected
        any_overlap += bool(detected & puzzle.themes)
        history_free_target_matched[target] += target in history_free_detected
        production_target_matched[target] += target in production_detected
        for theme in history_free_detected:
            history_free_predictions[theme] += 1
            history_free_true_predictions[theme] += theme in puzzle.themes
        for theme in production_detected:
            production_predictions[theme] += 1
            production_true_predictions[theme] += theme in puzzle.themes
            if theme not in puzzle.themes and len(production_false_positives[theme]) < 20:
                production_false_positives[theme].append(_diagnostic(puzzle, production_detected))
        if target not in production_detected and len(production_misses[target]) < 20:
            production_misses[target].append(_diagnostic(puzzle, production_detected))
        if target not in detected and len(misses[target]) < 20:
            misses[target].append(_diagnostic(puzzle, detected))
        for theme in detected:
            predictions[theme] += 1
            if theme in puzzle.themes:
                true_predictions[theme] += 1
            elif len(false_positives[theme]) < 20:
                false_positives[theme].append(_diagnostic(puzzle, detected))

    per_theme = {
        theme: {
            "matched": target_matched[theme],
            "total": target_total[theme],
            "accuracy": target_matched[theme] / target_total[theme],
            "predictions": predictions[theme],
            "true_predictions": true_predictions[theme],
            "precision": (
                true_predictions[theme] / predictions[theme] if predictions[theme] else 1.0
            ),
        }
        for theme in sorted(target_total)
    }
    total_predictions = sum(predictions.values())
    total_true_predictions = sum(true_predictions.values())
    target_accuracy = sum(target_matched.values()) / sum(target_total.values())
    macro_target_accuracy = sum(item["accuracy"] for item in per_theme.values()) / len(per_theme)
    precision = total_true_predictions / total_predictions if total_predictions else 1.0
    history_free_total = sum(history_free_predictions.values())
    history_free_true = sum(history_free_true_predictions.values())
    production_total = sum(production_predictions.values())
    production_true = sum(production_true_predictions.values())
    nonhistorical_themes = set(target_total) - {"intermezzo"}
    nonhistorical_total = sum(target_total[theme] for theme in nonhistorical_themes)
    nonhistorical_matched = sum(
        history_free_target_matched[theme] for theme in nonhistorical_themes
    )
    return {
        "puzzles": len(puzzles),
        "target_accuracy": target_accuracy,
        "macro_target_accuracy": macro_target_accuracy,
        "mapped_prediction_precision": precision,
        "any_mapped_theme_overlap": any_overlap / len(puzzles),
        "history_free": {
            "target_accuracy": sum(history_free_target_matched.values()) / len(puzzles),
            "nonhistorical_target_accuracy": nonhistorical_matched / nonhistorical_total,
            "intermezzo_target_accuracy": (
                history_free_target_matched["intermezzo"] / target_total["intermezzo"]
            ),
            "mapped_prediction_precision": (
                history_free_true / history_free_total if history_free_total else 1.0
            ),
            "predictions": history_free_total,
            "true_predictions": history_free_true,
        },
        "production": {
            "target_accuracy": sum(production_target_matched.values()) / len(puzzles),
            "nonhistorical_target_accuracy": (
                sum(production_target_matched[theme] for theme in nonhistorical_themes)
                / nonhistorical_total
            ),
            "mapped_prediction_precision": (
                production_true / production_total if production_total else 1.0
            ),
            "predictions": production_total,
            "true_predictions": production_true,
            "misses": dict(production_misses),
            "false_positives": dict(production_false_positives),
        },
        "per_theme": per_theme,
        "misses": dict(misses),
        "false_positives": dict(false_positives),
    }


def theme_gate(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if metrics["target_accuracy"] < 0.995:
        reasons.append("target-theme accuracy is below 99.5%")
    if metrics["macro_target_accuracy"] < 0.995:
        reasons.append("macro target-theme accuracy is below 99.5%")
    if metrics["mapped_prediction_precision"] < 0.995:
        reasons.append("mapped-theme precision is below 99.5%")
    history_free = metrics["history_free"]
    if history_free["nonhistorical_target_accuracy"] < 0.995:
        reasons.append("history-free nonhistorical accuracy is below 99.5%")
    if history_free["mapped_prediction_precision"] < 0.995:
        reasons.append("history-free mapped-theme precision is below 99.5%")
    production = metrics["production"]
    if production["mapped_prediction_precision"] < 0.995:
        reasons.append("production mapped-theme precision is below 99.5%")
    return not reasons, reasons


def _parse_puzzle(
    row: dict[str, str],
    *,
    path: Path,
    line_number: int,
) -> LichessPuzzle:
    try:
        moves = tuple(row["Moves"].split())
        themes = frozenset(row["Themes"].split())
        benchmark_theme = row["BenchmarkTheme"]
        puzzle = LichessPuzzle(
            puzzle_id=row["PuzzleId"],
            fen=row["FEN"],
            moves=moves,
            rating=int(row["Rating"]),
            rating_deviation=int(row["RatingDeviation"]),
            popularity=int(row["Popularity"]),
            plays=int(row["NbPlays"]),
            themes=themes,
            game_url=row["GameUrl"],
            opening_tags=tuple(row["OpeningTags"].split()),
            benchmark_theme=benchmark_theme,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid Lichess puzzle record at {path}:{line_number}") from exc
    if not puzzle.puzzle_id or len(moves) < 2:
        raise ValueError(f"Incomplete Lichess puzzle record at {path}:{line_number}")
    try:
        puzzle.game_id
    except ValueError as exc:
        raise ValueError(f"Invalid Lichess game URL at {path}:{line_number}") from exc
    if benchmark_theme not in SUPPORTED_LICHESS_THEMES or benchmark_theme not in themes:
        raise ValueError(f"Invalid benchmark theme at {path}:{line_number}")
    if puzzle.popularity < 80 or puzzle.plays < 50:
        raise ValueError(f"Puzzle quality threshold changed at {path}:{line_number}")
    try:
        board, solution, _previous = puzzle.analysis_input()
        current = board.copy(stack=False)
        for uci in solution:
            move = chess.Move.from_uci(uci)
            if move not in current.legal_moves:
                raise ValueError
            current.push(move)
    except ValueError as exc:
        raise ValueError(f"Illegal Lichess puzzle line at {path}:{line_number}") from exc
    return puzzle


def lichess_game_id(game_url: str) -> str:
    path = [part for part in urlsplit(game_url).path.split("/") if part]
    if not path:
        raise ValueError(f"Invalid Lichess game URL: {game_url}")
    return path[0]


def _diagnostic(puzzle: LichessPuzzle, detected: set[str]) -> dict[str, Any]:
    return {
        "puzzle_id": puzzle.puzzle_id,
        "benchmark_theme": puzzle.benchmark_theme,
        "expected_themes": sorted(puzzle.themes),
        "detected_themes": sorted(detected),
        "game_url": puzzle.game_url,
    }


def _expected_fields() -> list[str]:
    return [
        "PuzzleId",
        "FEN",
        "Moves",
        "Rating",
        "RatingDeviation",
        "Popularity",
        "NbPlays",
        "Themes",
        "GameUrl",
        "OpeningTags",
        "BenchmarkTheme",
    ]
