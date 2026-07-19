"""Detect and rectify one photographed 8x8 chess diagram."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BOARD_SIZE = 512
DETECTION_MAX_DIMENSION = 1400
_PATTERN_SIZE = (7, 7)


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
        outer = _extrapolate_outer_corners(inner_corners) / scale
        outer = _clamp_corners(outer, image_bgr.shape)
        if _quad_is_usable(outer, image_bgr.shape):
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
                confidence=min(0.8, max(0.35, score / 80.0)),
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

    destination = np.array(
        [
            [0.0, 0.0],
            [output_size - 1.0, 0.0],
            [output_size - 1.0, output_size - 1.0],
            [0.0, output_size - 1.0],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(points, destination)
    return cv2.warpPerspective(
        image_bgr,
        transform,
        (output_size, output_size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
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

    candidates: list[tuple[float, np.ndarray]] = []
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
        preview = rectify_board(image_bgr, corners, output_size=256)
        score = _checkerboard_score(preview)
        candidates.append((score, corners))

    if not candidates:
        return None
    score, corners = max(candidates, key=lambda item: item[0])
    if score < 8.0:
        return None
    return corners, score


def _checkerboard_score(board_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    size = min(gray.shape)
    square = size // 8
    trimmed = gray[: square * 8, : square * 8]
    cells = trimmed.reshape(8, square, 8, square).transpose(0, 2, 1, 3)
    margin = max(1, square // 5)
    inner = cells[:, :, margin:-margin, margin:-margin]
    means = inner.mean(axis=(2, 3))
    parity = np.indices((8, 8)).sum(axis=0) % 2 == 0
    group_a = means[parity]
    group_b = means[~parity]
    contrast = abs(float(group_a.mean() - group_b.mean()))
    inconsistency = float(group_a.std() + group_b.std())
    return contrast - 0.25 * inconsistency


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
