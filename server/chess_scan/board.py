"""Chess-board labels, FEN construction, and Lichess links."""

from __future__ import annotations

from typing import Literal

import chess

CLASS_NAMES = ("empty", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k")
PIECE_SYMBOLS = CLASS_NAMES
SQUARE_COUNT = 64
Orientation = Literal["white", "black"]
SideToMove = Literal["w", "b"]


def labels_to_board_fen(labels: list[int], *, orientation: Orientation = "white") -> str:
    """Convert image-order labels into canonical piece-placement FEN."""
    validate_labels(labels)
    canonical = labels if orientation == "white" else list(reversed(labels))

    ranks: list[str] = []
    for row in range(8):
        rank: list[str] = []
        empty_count = 0
        for class_id in canonical[row * 8 : (row + 1) * 8]:
            symbol = PIECE_SYMBOLS[class_id]
            if symbol == "empty":
                empty_count += 1
                continue
            if empty_count:
                rank.append(str(empty_count))
                empty_count = 0
            rank.append(symbol)
        if empty_count:
            rank.append(str(empty_count))
        ranks.append("".join(rank) or "8")
    return "/".join(ranks)


def build_full_fen(
    labels: list[int],
    *,
    orientation: Orientation,
    side_to_move: SideToMove,
    castling: str = "-",
    en_passant: str = "-",
) -> str:
    board_fen = labels_to_board_fen(labels, orientation=orientation)
    return f"{board_fen} {side_to_move} {castling or '-'} {en_passant or '-'} 0 1"


def fen_warnings(fen: str) -> list[str]:
    try:
        board = chess.Board(fen)
    except ValueError as exc:
        return [f"Invalid FEN: {exc}"]

    warnings: list[str] = []
    if len(board.pieces(chess.KING, chess.WHITE)) != 1:
        warnings.append("Position does not contain exactly one white king.")
    if len(board.pieces(chess.KING, chess.BLACK)) != 1:
        warnings.append("Position does not contain exactly one black king.")
    if not board.is_valid() and not warnings:
        warnings.append("The position is syntactically valid but not a legal chess position.")
    return warnings


def lichess_analysis_url(fen: str, *, orientation: Orientation) -> str:
    path_fen = fen.replace(" ", "_")
    return f"https://lichess.org/analysis/{path_fen}?color={orientation}"


def validate_full_fen(fen: str) -> None:
    try:
        chess.Board(fen)
    except ValueError as exc:
        raise ValueError(f"Invalid FEN: {exc}") from exc


def validate_labels(labels: list[int]) -> None:
    if len(labels) != SQUARE_COUNT:
        raise ValueError(f"Expected {SQUARE_COUNT} square labels, got {len(labels)}")
    invalid = [label for label in labels if label < 0 or label >= len(CLASS_NAMES)]
    if invalid:
        raise ValueError(
            f"Square labels must be between 0 and {len(CLASS_NAMES) - 1}, got {invalid[0]}"
        )
