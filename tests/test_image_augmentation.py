from __future__ import annotations

import cv2
import numpy as np
import pytest

from image_augmentation import hatched_print


def test_hatched_print_replaces_checker_background_without_erasing_pieces() -> None:
    board = np.empty((512, 512, 3), dtype=np.uint8)
    for row in range(8):
        for column in range(8):
            value = 230 if (row + column) % 2 == 0 else 150
            board[row * 64 : (row + 1) * 64, column * 64 : (column + 1) * 64] = value
    cv2.circle(board, (96, 32), 14, (40, 40, 40), -1)

    rendered = hatched_print(board, period=6, line_width=2)
    gray = cv2.cvtColor(rendered, cv2.COLOR_BGR2GRAY)

    assert rendered.shape == board.shape
    assert np.unique(gray[:64, :64]).tolist() == [242]
    assert len(np.unique(gray[:64, 64:128])) > 2
    assert int(gray[32, 96]) < int(np.median(gray[:64, 64:128])) - 40


def test_hatched_print_rejects_invalid_parameters() -> None:
    board = np.zeros((512, 512, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="BGR"):
        hatched_print(board[:, :, 0])
    with pytest.raises(ValueError, match="Hatch width"):
        hatched_print(board, period=2, line_width=2)
    with pytest.raises(ValueError, match="ordered bytes"):
        hatched_print(board, paper_level=100, hatch_level=150)
