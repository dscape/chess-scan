from __future__ import annotations

import cv2
import numpy as np

from chess_scan.geometry import (
    _checkerboard_score,
    _extrapolate_outer_corners,
    board_grid_fits,
    detect_board_corners,
    order_corners,
    project_board_grid,
    rectify_board,
)


def test_detects_and_rectifies_perspective_checkerboard() -> None:
    board = _checkerboard(640)
    source = np.float32([[0, 0], [639, 0], [639, 639], [0, 639]])
    target = np.float32([[170, 90], [720, 145], [670, 720], [105, 650]])
    transform = cv2.getPerspectiveTransform(source, target)
    photo = cv2.warpPerspective(board, transform, (820, 800), borderValue=(240, 236, 224))

    detection = detect_board_corners(photo)
    rectified = rectify_board(photo, detection.corners, output_size=256)

    assert detection.method in {"checkerboard", "contour"}
    assert detection.confidence >= 0.35
    assert len(project_board_grid(detection.corners)) == 81
    assert rectified.shape == (256, 256, 3)
    light = rectified[8:24, 8:24].mean()
    dark = rectified[8:24, 40:56].mean()
    assert abs(float(light - dark)) > 80


def test_rectification_antialiases_print_screen_patterns() -> None:
    rows, columns = np.indices((1024, 1024))
    screen = ((rows + columns) % 2 * 255).astype(np.uint8)
    source = np.repeat(screen[:, :, None], 3, axis=2)
    corners = [[0, 0], [1023, 0], [1023, 1023], [0, 1023]]
    expected = cv2.resize(source, (512, 512), interpolation=cv2.INTER_AREA)

    rectified = rectify_board(source, corners, output_size=512)

    difference = np.abs(rectified.astype(np.int16) - expected.astype(np.int16))
    assert difference.mean() < 1.0


def test_detects_low_contrast_blurred_checkerboard() -> None:
    board = _checkerboard(640, light=232, dark=202)
    source = np.float32([[0, 0], [639, 0], [639, 639], [0, 639]])
    target = np.float32([[120, 85], [690, 130], [655, 710], [90, 660]])
    transform = cv2.getPerspectiveTransform(source, target)
    photo = cv2.warpPerspective(board, transform, (780, 780), borderValue=(240, 240, 240))
    photo = cv2.GaussianBlur(photo, (5, 5), 1.2)

    detection = detect_board_corners(photo)

    assert detection.method == "checkerboard"
    assert detection.confidence == 0.95


def test_outer_corner_fit_uses_all_detected_intersections() -> None:
    canonical_outer = np.float32([[0, 0], [8, 0], [8, 8], [0, 8]])
    expected_outer = np.float32([[120, 85], [690, 130], [655, 710], [90, 660]])
    transform = cv2.getPerspectiveTransform(canonical_outer, expected_outer)
    canonical_grid = np.float32([[[column, row] for row in range(1, 8) for column in range(1, 8)]])
    detected_grid = cv2.perspectiveTransform(canonical_grid, transform)[0].reshape(7, 7, 2)
    detected_grid.reshape(49, 2)[[0, 6, 42, 48]] += np.float32([[2, -2], [-2, 2], [2, 2], [-2, -2]])

    for reordered_grid in (detected_grid, np.rot90(detected_grid, 2), np.fliplr(detected_grid)):
        fitted_outer = _extrapolate_outer_corners(reordered_grid)
        assert np.max(np.linalg.norm(fitted_outer - expected_outer, axis=1)) < 1.6


def test_pattern_score_uses_background_around_piece_centers() -> None:
    board = _checkerboard(256, light=210, dark=160)
    for row in range(8):
        for col in range(8):
            cv2.circle(board, (col * 32 + 16, row * 32 + 16), 12, (40, 40, 40), -1)

    assert _checkerboard_score(board) == 50.0


def test_grid_fit_rejects_corners_that_cut_through_squares() -> None:
    board = _checkerboard(256)

    assert board_grid_fits(board, [[0, 0], [255, 0], [255, 255], [0, 255]])
    assert not board_grid_fits(board, [[8, 0], [255, 0], [255, 255], [8, 255]])


def test_orders_strongly_perspective_corners_clockwise() -> None:
    shuffled = np.float32([[80, 300], [300, 40], [40, 70], [350, 260]])

    ordered = order_corners(shuffled)

    assert np.allclose(ordered, [[40, 70], [300, 40], [350, 260], [80, 300]])


def _checkerboard(size: int, *, light: int = 238, dark: int = 90) -> np.ndarray:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    square = size // 8
    for row in range(8):
        for col in range(8):
            value = light if (row + col) % 2 == 0 else dark
            image[row * square : (row + 1) * square, col * square : (col + 1) * square] = value
    return image
