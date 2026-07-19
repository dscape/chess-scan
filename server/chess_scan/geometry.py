"""Detect and rectify one photographed 8x8 chess diagram."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BOARD_SIZE = 512
DETECTION_MAX_DIMENSION = 1400
_PATTERN_SIZE = (7, 7)
_PATTERN_PREVIEW_SIZE = 256
_MIN_PATTERN_CONTRAST = 8.0
_MIN_PATTERN_CONSISTENCY = 0.9
_MAX_GRID_ALIGNMENT_ERROR = 0.1
_RECTIFICATION_SUPERSAMPLING = 2


@dataclass(frozen=True, slots=True)
class CornerDetection:
    corners: tuple[tuple[float, float], ...]
    method: str
    confidence: float


def detect_board_corners(image_bgr: np.ndarray) -> CornerDetection:
    """Find board corners, preferring the diagram's 7x7 internal intersections."""
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a BGR image")

    scaled, scale = _resize_for_detection(image_bgr)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)

    inner_corners = _find_checkerboard_corners(gray)
    if inner_corners is not None:
        scaled_outer = _clamp_corners(
            _extrapolate_outer_corners(inner_corners),
            scaled.shape,
        )
        if board_grid_fits(scaled, scaled_outer):
            outer = _clamp_corners(scaled_outer / scale, image_bgr.shape)
            return CornerDetection(
                corners=_corners_as_tuple(outer),
                method="checkerboard",
                confidence=0.95,
            )

    contour = _find_contour_board(scaled)
    if contour is not None:
        contour_corners, score = contour
        outer = _clamp_corners(contour_corners / scale, image_bgr.shape)
        if _quad_is_usable(outer, image_bgr.shape):
            return CornerDetection(
                corners=_corners_as_tuple(outer),
                method="contour",
                confidence=min(0.8, max(0.35, score / 50.0)),
            )

    fallback = _centered_square_corners(image_bgr.shape)
    return CornerDetection(
        corners=_corners_as_tuple(fallback),
        method="manual_adjustment_needed",
        confidence=0.1,
    )


def project_board_grid(
    corners: list[list[float]] | tuple[tuple[float, float], ...] | np.ndarray,
) -> list[list[float]]:
    """Project the canonical 9x9 board intersections into image coordinates."""
    image_corners = order_corners(np.asarray(corners, dtype=np.float32))
    canonical_corners = np.array(
        [[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]],
        dtype=np.float32,
    )
    canonical_grid = np.array(
        [[[float(col), float(row)] for row in range(9) for col in range(9)]],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(canonical_corners, image_corners)
    projected = cv2.perspectiveTransform(canonical_grid, transform)[0]
    return [[float(point[0]), float(point[1])] for point in projected]


def board_grid_fits(
    image_bgr: np.ndarray,
    corners: list[list[float]] | tuple[tuple[float, float], ...] | np.ndarray,
) -> bool:
    """Return whether the corners tightly frame a complete, aligned 8x8 grid."""
    points = order_corners(np.asarray(corners, dtype=np.float32))
    if not _quad_is_usable(points, image_bgr.shape):
        return False
    preview = rectify_board(image_bgr, points, output_size=_PATTERN_PREVIEW_SIZE)
    return _checkerboard_score(preview) >= _MIN_PATTERN_CONTRAST


def rectify_board(
    image_bgr: np.ndarray,
    corners: list[list[float]] | tuple[tuple[float, float], ...] | np.ndarray,
    *,
    output_size: int = BOARD_SIZE,
) -> np.ndarray:
    points = order_corners(np.asarray(corners, dtype=np.float32))
    if points.shape != (4, 2):
        raise ValueError(f"Expected four corners, got shape {points.shape}")
    if not _quad_is_usable(points, image_bgr.shape):
        raise ValueError("The selected corners do not form a usable board area")

    sampling_size = output_size * _RECTIFICATION_SUPERSAMPLING
    destination = np.array(
        [
            [0.0, 0.0],
            [sampling_size - 1.0, 0.0],
            [sampling_size - 1.0, sampling_size - 1.0],
            [0.0, sampling_size - 1.0],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    sampled = cv2.warpPerspective(
        image_bgr,
        transform,
        (sampling_size, sampling_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return cv2.resize(
        sampled,
        (output_size, output_size),
        interpolation=cv2.INTER_AREA,
    )


def order_corners(points: np.ndarray) -> np.ndarray:
    """Return quadrilateral points as top-left, top-right, bottom-right, bottom-left."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2):
        raise ValueError(f"Expected four 2D points, got shape {points.shape}")

    center = points.mean(axis=0)
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    clockwise = points[np.argsort(angles)]
    top_left_index = int(np.argmin(clockwise[:, 0] + clockwise[:, 1]))
    return np.roll(clockwise, -top_left_index, axis=0).astype(np.float32)


def _resize_for_detection(image_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image_bgr.shape[:2]
    scale = min(1.0, DETECTION_MAX_DIMENSION / max(height, width))
    if scale == 1.0:
        return image_bgr, scale
    resized = cv2.resize(
        image_bgr,
        (int(round(width * scale)), int(round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def _find_checkerboard_corners(gray: np.ndarray) -> np.ndarray | None:
    sb_flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
    enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    for candidate in (gray, enhanced):
        try:
            found, corners = cv2.findChessboardCornersSB(
                candidate,
                _PATTERN_SIZE,
                flags=sb_flags,
            )
        except cv2.error:
            found, corners = False, None
        if found and corners is not None:
            return corners.reshape(-1, 2)

    regular_flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    for candidate in (gray, enhanced):
        found, corners = cv2.findChessboardCorners(
            candidate,
            _PATTERN_SIZE,
            flags=regular_flags,
        )
        if not found or corners is None:
            continue
        refined = cv2.cornerSubPix(
            candidate,
            corners,
            (5, 5),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        return refined.reshape(-1, 2)
    return None


def _extrapolate_outer_corners(inner_corners: np.ndarray) -> np.ndarray:
    detected_grid = _order_detected_grid(inner_corners.reshape(7, 7, 2))
    canonical_grid = np.array(
        [[float(column), float(row)] for row in range(1, 8) for column in range(1, 8)],
        dtype=np.float32,
    )
    canonical_outer = np.array(
        [[[0.0, 0.0], [8.0, 0.0], [8.0, 8.0], [0.0, 8.0]]],
        dtype=np.float32,
    )
    transform, _ = cv2.findHomography(canonical_grid, detected_grid.reshape(49, 2))
    if transform is None:
        canonical_inner = np.array(
            [[1.0, 1.0], [7.0, 1.0], [7.0, 7.0], [1.0, 7.0]],
            dtype=np.float32,
        )
        detected_inner = np.array(
            [
                detected_grid[0, 0],
                detected_grid[0, 6],
                detected_grid[6, 6],
                detected_grid[6, 0],
            ],
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(canonical_inner, detected_inner)
    return cv2.perspectiveTransform(canonical_outer, transform)[0]


def _order_detected_grid(grid: np.ndarray) -> np.ndarray:
    endpoints = np.array([grid[0, 0], grid[0, 6], grid[6, 6], grid[6, 0]], dtype=np.float32)
    ordered_endpoints = order_corners(endpoints)
    rotations = [np.rot90(grid, turns) for turns in range(4)]
    candidates = rotations + [np.fliplr(candidate) for candidate in rotations]

    def endpoint_error(candidate: np.ndarray) -> float:
        candidate_endpoints = np.array(
            [candidate[0, 0], candidate[0, 6], candidate[6, 6], candidate[6, 0]],
            dtype=np.float32,
        )
        return float(np.linalg.norm(candidate_endpoints - ordered_endpoints, axis=1).sum())

    return min(candidates, key=endpoint_error).copy()


def _find_contour_board(image_bgr: np.ndarray) -> tuple[np.ndarray, float] | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    image_area = float(gray.shape[0] * gray.shape[1])

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:80]:
        area = float(cv2.contourArea(contour))
        if area < image_area * 0.06 or area > image_area * 0.98:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        corners = order_corners(approx.reshape(4, 2).astype(np.float32))
        if not _quad_is_usable(corners, image_bgr.shape):
            continue
        preview = rectify_board(image_bgr, corners, output_size=_PATTERN_PREVIEW_SIZE)
        score = _checkerboard_score(preview)
        if score >= _MIN_PATTERN_CONTRAST:
            return corners, score

    return None


def _checkerboard_score(board_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    backgrounds = _cell_backgrounds(gray)
    signed_differences = _alternating_neighbor_differences(backgrounds)
    if np.median(signed_differences) < 0:
        signed_differences *= -1

    contrast = float(np.median(signed_differences))
    minimum_difference = max(2.0, contrast * 0.2)
    consistency = float(np.mean(signed_differences > minimum_difference))
    alignment_error = _grid_alignment_error(gray)
    if (
        contrast < _MIN_PATTERN_CONTRAST
        or consistency < _MIN_PATTERN_CONSISTENCY
        or alignment_error > _MAX_GRID_ALIGNMENT_ERROR
    ):
        return 0.0
    return contrast


def _cell_backgrounds(gray: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    row_boundaries = [int(round(index * height / 8)) for index in range(9)]
    column_boundaries = [int(round(index * width / 8)) for index in range(9)]
    backgrounds = np.empty((8, 8), dtype=np.float32)
    for row in range(8):
        for column in range(8):
            cell = gray[
                row_boundaries[row] : row_boundaries[row + 1],
                column_boundaries[column] : column_boundaries[column + 1],
            ]
            backgrounds[row, column] = float(np.median(_corner_patch_pixels(cell)))
    return backgrounds


def _corner_patch_pixels(cell: np.ndarray) -> np.ndarray:
    height, width = cell.shape
    top, bottom = _patch_ranges(height)
    left, right = _patch_ranges(width)
    return np.concatenate(
        (
            cell[top, left].ravel(),
            cell[top, right].ravel(),
            cell[bottom, left].ravel(),
            cell[bottom, right].ravel(),
        )
    )


def _patch_ranges(length: int) -> tuple[slice, slice]:
    near_start = max(1, int(round(length * 0.14)))
    near_end = max(near_start + 1, int(round(length * 0.32)))
    far_start = min(length - 2, int(round(length * 0.68)))
    far_end = min(length - 1, max(far_start + 1, int(round(length * 0.86))))
    return slice(near_start, near_end), slice(far_start, far_end)


def _alternating_neighbor_differences(backgrounds: np.ndarray) -> np.ndarray:
    parity_sign = np.where(np.indices((8, 8)).sum(axis=0) % 2 == 0, 1.0, -1.0)
    horizontal = parity_sign[:, :-1] * (backgrounds[:, :-1] - backgrounds[:, 1:])
    vertical = parity_sign[:-1, :] * (backgrounds[:-1, :] - backgrounds[1:, :])
    return np.concatenate((horizontal.ravel(), vertical.ravel()))


def _grid_alignment_error(gray: np.ndarray) -> float:
    vertical = _alternating_profile(gray, axis=0)
    horizontal = _alternating_profile(gray, axis=1)
    return max(_profile_alignment_error(vertical), _profile_alignment_error(horizontal))


def _alternating_profile(gray: np.ndarray, *, axis: int) -> np.ndarray:
    length = gray.shape[axis]
    boundaries = [int(round(index * length / 8)) for index in range(9)]
    profiles = []
    for index in range(8):
        start, end = boundaries[index], boundaries[index + 1]
        near, far = _patch_ranges(end - start)
        selected = np.r_[
            np.arange(start + near.start, start + near.stop),
            np.arange(start + far.start, start + far.stop),
        ]
        profiles.append(np.median(np.take(gray, selected, axis=axis), axis=axis))
    signs = np.where(np.arange(8) % 2 == 0, 1.0, -1.0)
    return np.mean(np.stack(profiles) * signs[:, None], axis=0)


def _profile_alignment_error(profile: np.ndarray) -> float:
    length = len(profile)
    square = length / 8.0
    smoothed = cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), 2.0)[0]
    gradient = np.abs(np.gradient(smoothed))
    search_radius = max(2, int(round(square * 0.25)))
    errors = []
    for index in range(1, 8):
        expected = int(round(index * square))
        start = max(0, expected - search_radius)
        end = min(length, expected + search_radius + 1)
        peak = start + int(np.argmax(gradient[start:end]))
        errors.append(abs(peak - expected) / square)
    return max(errors)


def _quad_is_usable(corners: np.ndarray, image_shape: tuple[int, ...]) -> bool:
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return False
    height, width = image_shape[:2]
    area = abs(float(cv2.contourArea(order_corners(corners))))
    if area < height * width * 0.02:
        return False
    edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
    return bool(edges.min() >= 16 and edges.max() / edges.min() <= 6.0)


def _clamp_corners(corners: np.ndarray, image_shape: tuple[int, ...]) -> np.ndarray:
    height, width = image_shape[:2]
    clamped = corners.copy()
    clamped[:, 0] = np.clip(clamped[:, 0], 0, width - 1)
    clamped[:, 1] = np.clip(clamped[:, 1], 0, height - 1)
    return order_corners(clamped)


def _centered_square_corners(image_shape: tuple[int, ...]) -> np.ndarray:
    height, width = image_shape[:2]
    side = min(height, width) * 0.9
    left = (width - side) / 2.0
    top = (height - side) / 2.0
    return np.array(
        [[left, top], [left + side, top], [left + side, top + side], [left, top + side]],
        dtype=np.float32,
    )


def _corners_as_tuple(corners: np.ndarray) -> tuple[tuple[float, float], ...]:
    return tuple((float(point[0]), float(point[1])) for point in corners)
