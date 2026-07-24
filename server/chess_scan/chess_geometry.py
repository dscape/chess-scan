"""Shared legal-move geometry and material helpers."""

from __future__ import annotations

import chess

MATERIAL_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}


def material_balance(board: chess.Board, color: chess.Color) -> int:
    return sum(
        (1 if piece.color == color else -1) * MATERIAL_VALUES[piece.piece_type]
        for piece in board.piece_map().values()
    )


def captured_piece_and_square(
    board: chess.Board,
    move: chess.Move,
) -> tuple[chess.Piece | None, chess.Square | None]:
    if not board.is_capture(move):
        return None, None
    captured_square = move.to_square
    if board.is_en_passant(move):
        captured_square += -8 if board.turn == chess.WHITE else 8
    return board.piece_at(captured_square), captured_square


def captured_piece(board: chess.Board, move: chess.Move) -> chess.Piece | None:
    return captured_piece_and_square(board, move)[0]
