from __future__ import annotations

import cv2
import numpy as np

from chess_scan.capture_quality import board_capture_quality, compose_best_board


def test_localizes_glare_without_flagging_white_board_squares() -> None:
    board = _textured_board()
    cv2.circle(board, (32, 7 * 64 + 32), 16, (255, 255, 255), -1)
    cv2.circle(board, (32, 7 * 64 + 32), 16, (25, 25, 25), 3)
    glared = board.copy()
    cv2.circle(glared, (4 * 64 + 32, 3 * 64 + 32), 27, (255, 255, 255), -1)

    clear_quality = board_capture_quality(board)
    glare_quality = board_capture_quality(glared)

    assert clear_quality.glare_squares == ()
    assert 3 * 8 + 4 in glare_quality.glare_squares
    assert glare_quality.glare_score > 0.2


def test_reports_when_motion_blur_removes_board_detail() -> None:
    board = _textured_board()
    blurred = cv2.GaussianBlur(board, (0, 0), 4.0)

    clear_quality = board_capture_quality(board)
    blurred_quality = board_capture_quality(blurred)

    assert clear_quality.sharpness_score > blurred_quality.sharpness_score * 5
    assert blurred_quality.sharpness_score >= 0


def test_composes_each_square_from_its_clearest_aligned_frame() -> None:
    clear = _textured_board()
    first = clear.copy()
    target_index = 2 * 8 + 5
    center = (5 * 64 + 32, 2 * 64 + 32)
    cv2.circle(first, center, 29, (255, 255, 255), -1)

    composite, selected = compose_best_board([first, clear])

    assert selected[target_index] == 1
    assert np.array_equal(
        composite[2 * 64 : 3 * 64, 5 * 64 : 6 * 64],
        clear[2 * 64 : 3 * 64, 5 * 64 : 6 * 64],
    )
    assert set(selected) <= {0, 1}


def test_rejects_unaligned_frame_shapes() -> None:
    first = _textured_board()
    second = np.zeros((256, 256, 3), dtype=np.uint8)

    try:
        compose_best_board([first, second])
    except ValueError as error:
        assert "share one shape" in str(error)
    else:
        raise AssertionError("Expected mismatched frames to be rejected")


def _textured_board() -> np.ndarray:
    board = np.empty((512, 512, 3), dtype=np.uint8)
    for row in range(8):
        for column in range(8):
            value = 226 if (row + column) % 2 == 0 else 92
            top, left = row * 64, column * 64
            board[top : top + 64, left : left + 64] = value
            cv2.line(
                board,
                (left + 14, top + 18),
                (left + 48, top + 46),
                (35, 35, 35),
                4,
            )
    return board
