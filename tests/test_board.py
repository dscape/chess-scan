from __future__ import annotations

import pytest

from chess_scan.board import (
    build_full_fen,
    labels_to_board_fen,
    lichess_analysis_url,
    validate_full_fen,
)


def test_labels_to_board_fen() -> None:
    labels = [0] * 64
    labels[0] = 12
    labels[63] = 6

    assert labels_to_board_fen(labels) == "k7/8/8/8/8/8/8/7K"
    assert labels_to_board_fen(labels, orientation="black") == "K7/8/8/8/8/8/8/7k"


def test_full_fen_and_lichess_url() -> None:
    labels = [0] * 64
    labels[4] = 12
    labels[60] = 6
    fen = build_full_fen(labels, orientation="white", side_to_move="b")

    assert fen == "4k3/8/8/8/8/8/8/4K3 b - - 0 1"
    assert lichess_analysis_url(fen, orientation="white") == (
        "https://lichess.org/analysis/4k3/8/8/8/8/8/8/4K3_b_-_-_0_1?color=white"
    )


def test_validate_full_fen_rejects_an_invalid_position() -> None:
    board = validate_full_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1")
    assert board.fen() == "4k3/8/8/8/8/8/8/4K3 w - - 0 1"

    with pytest.raises(ValueError, match="Expected exactly one white king; found 2"):
        validate_full_fen("2R1K1nr/pp3ppp/q1n1p3/2bpP3/P7/1PP2N2/2Q2PPP/RNB1K2R w - - 0 1")
