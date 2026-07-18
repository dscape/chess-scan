from __future__ import annotations

from chess_scan.board import build_full_fen, labels_to_board_fen, lichess_analysis_url


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
