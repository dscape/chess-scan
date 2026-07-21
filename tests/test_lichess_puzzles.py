from __future__ import annotations

import chess
import pytest

from chess_scan.lichess_puzzles import theme_gate
from chess_scan.review_themes import (
    detect_solution_theme_modes,
    detect_solution_themes,
    detected_lichess_themes,
)

LICHESS_THEME_FIXTURES = (
    (
        "attraction",
        "4r1k1/ppp2pp1/3b1r2/2p4p/3P3q/P1P1B2P/2P1QPB1/R3R1K1 w - - 2 21",
        "e2b5 e8e3 e1e3 h4f2 g1h1 f2e3",
    ),
    (
        "capturingDefender",
        "r3kb1r/ppp2ppp/2n5/4p2b/5B2/2P2N1P/PPP1BPP1/R2R2K1 w kq - 0 11",
        "f4e5 h5f3 e2f3 c6e5",
    ),
    (
        "clearance",
        "3r1r1k/pp3pqN/8/7Q/3p1p2/P3p2R/2P3P1/4K3 b - - 0 33",
        "h8g8 h7f6 g7f6 h5h7",
    ),
    (
        "discoveredAttack",
        "r4rk1/1b6/1p2pqp1/p3n1NP/3p3Q/3B4/PPP2P1P/R4RK1 w - - 2 21",
        "h5g6 f6g5 h4g5 e5f3 g1g2 f3g5",
    ),
    (
        "fork",
        "7k/p5p1/1bp4p/4Q3/8/3q2Pb/P3N3/2R4K w - - 3 30",
        "c1c6 d3f3 h1h2 f3g2",
    ),
    (
        "interference",
        "r4r1k/7q/2pN1p2/p1p1pR2/P2p2Q1/1P1P3P/2PN2n1/6K1 b - - 1 26",
        "f8g8 d6f7 h7f7 f5h5 f7h7 h5h7 h8h7 g4h5 h7g7 g1g2",
    ),
    (
        "intermezzo",
        "r2q1rk1/pp2ppb1/3p1npp/2pP1b2/2P2n1N/P3P3/1P1NBPPP/R2Q1RK1 w - - 0 13",
        "h4f5 f4e2 d1e2 g6f5",
    ),
    (
        "pin",
        "6rr/pp6/1k5b/4pP2/Qp1n4/3P1bB1/P4P2/R3R1K1 w - - 0 26",
        "a4b4 b6c7 b4c4 c7b8 c4g8 h8g8",
    ),
    (
        "trappedPiece",
        "6k1/4bppp/p3p3/1p1pP3/1P3P2/PNr1B3/1nP1K1PP/2R5 b - - 1 29",
        "b2c4 e3d4 c3c2 c1c2",
    ),
    (
        "xRayAttack",
        "4Rr1k/pp4pp/5r2/5nq1/2P5/8/PP3QB1/4R1K1 b - - 7 40",
        "f5h4 e8f8 f6f8 f2f8",
    ),
)


@pytest.mark.parametrize(("theme", "fen", "moves"), LICHESS_THEME_FIXTURES)
def test_supported_theme_matches_public_lichess_metadata(
    theme: str,
    fen: str,
    moves: str,
) -> None:
    source = chess.Board(fen)
    all_moves = moves.split()
    setup = chess.Move.from_uci(all_moves[0])
    presented = source.copy(stack=False)
    presented.push(setup)

    detected = detected_lichess_themes(
        presented,
        all_moves[1:],
        previous=(source, setup),
    )

    assert theme in detected


def test_discovered_check_records_the_hidden_checker_as_the_actor() -> None:
    _theme, fen, moves = next(
        fixture for fixture in LICHESS_THEME_FIXTURES if fixture[0] == "discoveredAttack"
    )
    source = chess.Board(fen)
    all_moves = moves.split()
    setup = chess.Move.from_uci(all_moves[0])
    presented = source.copy(stack=False)
    presented.push(setup)

    evidence = next(
        item
        for item in detect_solution_themes(
            presented,
            all_moves[1:],
            previous=(source, setup),
        )
        if item.theme == "discoveredAttack"
    )

    assert evidence.actor is not None
    assert evidence.actor.square == "b7"
    assert evidence.to_square == "g5"


def test_theme_modes_share_detection_without_changing_history_free_results() -> None:
    _theme, fen, moves = LICHESS_THEME_FIXTURES[0]
    source = chess.Board(fen)
    all_moves = moves.split()
    setup = chess.Move.from_uci(all_moves[0])
    presented = source.copy(stack=False)
    presented.push(setup)

    contextual, history_free = detect_solution_theme_modes(
        presented,
        all_moves[1:],
        previous=(source, setup),
    )

    assert {item.theme for item in contextual} == set(
        detected_lichess_themes(presented, all_moves[1:], previous=(source, setup))
    )
    assert {item.theme for item in history_free} == set(
        detected_lichess_themes(presented, all_moves[1:])
    )


def test_history_dependent_intermezzo_abstains_without_the_setup_move() -> None:
    theme, fen, moves = next(
        fixture for fixture in LICHESS_THEME_FIXTURES if fixture[0] == "intermezzo"
    )
    source = chess.Board(fen)
    all_moves = moves.split()
    source.push_uci(all_moves[0])

    assert theme not in detected_lichess_themes(source, all_moves[1:])


def test_lichess_theme_gate_requires_near_perfect_recall_and_precision() -> None:
    passing = {
        "target_accuracy": 0.998,
        "macro_target_accuracy": 0.998,
        "mapped_prediction_precision": 0.999,
        "history_free": {
            "nonhistorical_target_accuracy": 0.998,
            "mapped_prediction_precision": 1.0,
        },
        "production": {
            "nonhistorical_target_accuracy": 0.998,
            "mapped_prediction_precision": 1.0,
        },
    }
    failing = {**passing, "mapped_prediction_precision": 0.99}

    assert theme_gate(passing) == (True, [])
    passed, reasons = theme_gate(failing)
    assert passed is False
    assert reasons == ["mapped-theme precision is below 99.5%"]

    history_free_failing = {
        **passing,
        "history_free": {
            **passing["history_free"],
            "nonhistorical_target_accuracy": 0.99,
        },
    }
    passed, reasons = theme_gate(history_free_failing)
    assert passed is False
    assert reasons == ["history-free nonhistorical accuracy is below 99.5%"]

    production_failing = {
        **passing,
        "production": {
            **passing["production"],
            "mapped_prediction_precision": 0.99,
        },
    }
    passed, reasons = theme_gate(production_failing)
    assert passed is False
    assert reasons == ["production mapped-theme precision is below 99.5%"]
