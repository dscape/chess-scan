from __future__ import annotations

import chess
import pytest

from chess_scan.review_detectors import (
    DetectedSubject,
    ReviewContext,
    build_analyzed_line,
    detect_primary_subject,
    detect_subjects,
    teaching_subjects,
)


def _findings(fen: str, moves: list[str]) -> tuple[DetectedSubject, ...]:
    board = chess.Board(fen)
    assert board.is_valid(), fen
    line = build_analyzed_line(board, moves)
    return detect_subjects(ReviewContext(board, line))


def _handlers(fen: str, moves: list[str]) -> set[str]:
    return {finding.handler for finding in _findings(fen, moves)}


@pytest.mark.parametrize(
    ("handler", "fen", "moves"),
    [
        (
            "mate",
            "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
            ["f7f8"],
        ),
        (
            "double_attack",
            "8/7k/8/8/2r5/8/4Q3/4K3 w - - 0 1",
            ["e2e4", "h7g8", "e4c4"],
        ),
        (
            "pin",
            "4k3/4n3/8/8/8/8/R7/4K3 w - - 0 1",
            ["a2e2", "e8f7", "e2e7"],
        ),
        (
            "discovered_attack",
            "4k3/8/8/8/4N3/8/8/4R1K1 w - - 0 1",
            ["e4f6", "e8f7"],
        ),
        (
            "eliminate_defence",
            "2r3k1/2q5/8/4p3/3B4/8/8/2R3K1 w - - 0 1",
            ["d4e5", "c7e5", "c1c8"],
        ),
        (
            "xray",
            "7k/2q5/8/2r5/8/8/8/2R3K1 w - - 0 1",
            ["c1c4", "h8g8", "c4c5"],
        ),
        (
            "intermediate_move",
            "6k1/8/8/1Q6/8/8/p7/R3K3 w - - 0 1",
            ["b5b8", "g8f7", "a1a2"],
        ),
        (
            "promotion",
            "8/k1P5/8/8/8/8/8/7K w - - 0 1",
            ["c7c8n", "a7a6"],
        ),
        (
            "trapping",
            "n6k/2p5/1p6/3B4/8/8/8/6K1 w - - 0 1",
            ["d5b7", "h8g8", "b7a8"],
        ),
        (
            "interference",
            "r6q/8/4k3/6B1/8/8/8/4K2R w - - 0 1",
            ["g5d8", "e6f7", "h1h8"],
        ),
        (
            "luring",
            "6k1/7p/8/7Q/8/8/4K3/R7 w - - 0 1",
            ["h5h7", "g8h7", "a1h1"],
        ),
        (
            "magnet",
            "6k1/7p/8/7Q/8/8/4K3/R7 w - - 0 1",
            ["h5h7", "g8h7", "a1h1"],
        ),
        (
            "chasing_targeting",
            "4k3/8/8/q7/8/8/8/R5K1 w - - 0 1",
            ["a1a4", "a5b5", "a4a8"],
        ),
        (
            "clearing",
            "4k3/8/8/3r4/3N4/8/8/3RK3 w - - 0 1",
            ["d4f5", "e8f7", "d1d5"],
        ),
        (
            "breakthrough",
            "7k/8/3p4/4P3/8/8/8/7K w - - 0 1",
            ["e5d6", "h8g8"],
        ),
        (
            "defence",
            "4r1k1/8/8/8/8/8/3B4/4K3 w - - 0 1",
            ["d2e3", "g8f7"],
        ),
    ],
)
def test_tactical_detectors_emit_grounded_subjects(
    handler: str,
    fen: str,
    moves: list[str],
) -> None:
    assert handler in _handlers(fen, moves)


@pytest.mark.parametrize(
    ("handler", "fen", "moves"),
    [
        (
            "passed_pawn",
            "7k/8/8/4P3/8/8/8/7K w - - 0 1",
            ["e5e6", "h8g8"],
        ),
        (
            "open_file",
            "6k1/8/8/8/8/8/8/R5K1 w - - 0 1",
            ["a1a7", "g8f8"],
        ),
        (
            "seventh_rank",
            "6k1/8/8/8/8/8/8/R5K1 w - - 0 1",
            ["a1a7", "g8f8"],
        ),
        (
            "weak_pawn",
            "7k/8/8/3p4/3p4/8/8/R6K w - - 0 1",
            ["a1d1", "h8g8"],
        ),
        (
            "material_advantage",
            "6k1/8/8/8/8/8/8/Q6K w - - 0 1",
            ["a1a7", "g8f8"],
        ),
        (
            "pawn_endgame",
            "7k/8/8/4P3/8/8/8/7K w - - 0 1",
            ["e5e6", "h8g8"],
        ),
        (
            "rook_endgame",
            "6k1/7r/8/8/8/8/8/R5K1 w - - 0 1",
            ["a1a7", "g8f8"],
        ),
        (
            "wrong_bishop",
            "k7/P7/8/8/8/8/3B4/7K w - - 0 1",
            ["d2e3", "a8b7"],
        ),
    ],
)
def test_position_evaluators_only_emit_when_signature_is_present(
    handler: str,
    fen: str,
    moves: list[str],
) -> None:
    assert handler in _handlers(fen, moves)


@pytest.mark.parametrize(
    ("handler", "fen", "moves"),
    [
        (
            "pin",
            "4k3/4n3/8/8/8/8/R7/4K3 w - - 0 1",
            ["a2e2", "e8f7"],
        ),
        (
            "xray",
            "7k/2q5/8/2r5/8/8/8/2R3K1 w - - 0 1",
            ["c1c4", "h8g8"],
        ),
        (
            "promotion",
            "k7/6P1/5K2/8/8/8/8/8 w - - 0 1",
            ["g7g8n", "a8b7"],
        ),
        (
            "trapping",
            "n6k/2p5/1p6/3B4/8/8/8/6K1 w - - 0 1",
            ["d5b7", "h8g8"],
        ),
        (
            "wrong_bishop",
            "k7/P7/8/8/2B5/8/8/7K w - - 0 1",
            ["c4b3", "a8b7"],
        ),
        (
            "open_file",
            "6k1/8/8/8/8/8/8/R5K1 w - - 0 1",
            ["g1f2", "g8f7"],
        ),
        (
            "weak_pawn",
            "7k/8/8/3p4/3p4/8/8/7K w - - 0 1",
            ["h1g1", "h8g8"],
        ),
    ],
)
def test_detectors_abstain_when_a_feature_is_not_exploited(
    handler: str,
    fen: str,
    moves: list[str],
) -> None:
    assert handler not in _handlers(fen, moves)


@pytest.mark.parametrize(
    ("handler", "fen", "moves"),
    [
        (
            "double_attack",
            "8/7k/8/8/2r5/8/4Q3/4K3 w - - 0 1",
            ["e2e4", "h7g8", "e4c4"],
        ),
        (
            "breakthrough",
            "7k/8/3p4/4P3/8/8/8/7K w - - 0 1",
            ["e5d6", "h8g8"],
        ),
        (
            "seventh_rank",
            "6k1/8/8/8/8/8/8/R5K1 w - - 0 1",
            ["a1a7", "g8f8"],
        ),
    ],
)
def test_color_mirrored_positions_keep_the_same_subject(
    handler: str,
    fen: str,
    moves: list[str],
) -> None:
    mirrored = chess.Board(fen).mirror()
    mirrored_moves = [_mirror_uci(move) for move in moves]

    assert handler in _handlers(mirrored.fen(), mirrored_moves)


@pytest.mark.parametrize(
    ("fen", "moves"),
    [
        (
            "7k/2q5/8/R1r5/8/8/B7/6K1 w - - 0 1",
            ["a2c4", "h8g7", "a5c5"],
        ),
        (
            "7k/2q5/8/2r5/1B6/8/8/2R3K1 w - - 0 1",
            ["c1c4", "h8g8", "b4c5"],
        ),
    ],
)
def test_xray_requires_a_valid_slider_ray_and_exploitation_by_that_slider(
    fen: str,
    moves: list[str],
) -> None:
    assert "xray" not in _handlers(fen, moves)


def test_eliminating_defence_requires_the_first_move_to_induce_the_reply() -> None:
    assert "eliminate_defence" not in _handlers(
        "2r3k1/2q5/8/8/8/8/7P/2R3K1 w - - 0 1",
        ["h2h3", "c7e5", "c1c8"],
    )


def test_threat_requires_the_target_to_exist_after_the_first_move() -> None:
    assert "threat" not in _handlers(
        "4k3/3q4/8/8/8/8/8/R3K3 w - - 0 1",
        ["a1a8", "d7d8", "a8d8"],
    )
    assert "threat" in _handlers(
        "7k/8/8/8/2q5/8/8/R5K1 w - - 0 1",
        ["a1a4", "h8g8", "a4c4"],
    )


def test_magnet_requires_the_king_to_capture_the_first_moved_piece() -> None:
    assert "magnet" not in _handlers(
        "6k1/7R/8/8/8/8/8/R2QK3 w - - 0 1",
        ["a1a2", "g8h7", "d1h5"],
    )


def test_passed_pawn_and_seventh_rank_report_the_moved_piece() -> None:
    passed = next(
        finding
        for finding in _findings(
            "7k/8/P7/8/8/4P3/8/7K w - - 0 1",
            ["e3e4", "h8g8"],
        )
        if finding.handler == "passed_pawn"
    )
    seventh_rank = next(
        finding
        for finding in _findings(
            "6k1/R7/8/8/8/8/8/6KR w - - 0 1",
            ["h1h7", "g8f8"],
        )
        if finding.handler == "seventh_rank"
    )

    assert passed.evidence[0].squares == ("e4",)
    assert seventh_rank.evidence[0].squares == ("h7",)


def test_material_advantage_only_describes_the_moving_side_when_ahead() -> None:
    assert "material_advantage" not in _handlers(
        "6k1/7r/8/8/8/8/8/6K1 w - - 0 1",
        ["g1f2", "g8f8"],
    )


def test_unanswered_horizon_capture_is_not_counted_when_it_can_be_recaptured() -> None:
    assert "material" not in _handlers(
        chess.STARTING_FEN,
        ["e2e4", "e7e5", "f1b5", "b8c6", "b5c6"],
    )


def test_primary_detection_matches_the_first_production_teaching_finding() -> None:
    board = chess.Board("8/7k/8/8/2r5/8/4Q3/4K3 w - - 0 1")
    line = build_analyzed_line(board, ["e2e4", "h7g8", "e4c4"])
    context = ReviewContext(board, line)

    assert detect_primary_subject(context) == teaching_subjects(context)[0]
    assert detect_subjects(context)[0].handler == "double_attack"
    assert teaching_subjects(context)[0].handler == "material"


def test_engine_line_rejects_illegal_moves() -> None:
    board = chess.Board()

    with pytest.raises(ValueError, match="illegal move"):
        build_analyzed_line(board, ["e2e5"])


def _mirror_uci(uci: str) -> str:
    move = chess.Move.from_uci(uci)
    mirrored = chess.Move(
        chess.square_mirror(move.from_square),
        chess.square_mirror(move.to_square),
        promotion=move.promotion,
    )
    return mirrored.uci()
