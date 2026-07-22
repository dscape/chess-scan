import type { BoardPoint } from "./board";
import type {
  ReviewArrow,
  ReviewDiagramBadge,
  ReviewSquareMarker,
} from "./types";

export type ReviewArrowLine = {
  start: BoardPoint;
  end: BoardPoint;
  keylineEnd: BoardPoint;
};

export type PositionedReviewArrow = {
  arrow: ReviewArrow;
  line: ReviewArrowLine;
};

export type PositionedReviewBadge = {
  badge: ReviewDiagramBadge;
  point: BoardPoint;
  arrow: ReviewArrow;
};

export type ReviewDiagramErrorCode =
  | "invalid_arrow"
  | "invalid_badge_anchor"
  | "missing_badge_arrow";

export class ReviewDiagramError extends Error {
  readonly code: ReviewDiagramErrorCode;

  constructor(code: ReviewDiagramErrorCode, message: string) {
    super(message);
    this.name = "ReviewDiagramError";
    this.code = code;
  }
}

type PointForSquare = (square: string) => BoardPoint | null;

const ARROW_WIDTH = 0.115;
const ARROW_KEYLINE_WIDTH = 0.185;
const ARROW_BORDER_WIDTH = (ARROW_KEYLINE_WIDTH - ARROW_WIDTH) / 2;
const BADGE_RADIUS = 0.245;
const BADGE_FACE_RADIUS = 0.18;
const BADGE_KEYLINE_WIDTH = 0.07;

export const REVIEW_DIAGRAM_GEOMETRY = {
  arrowWidth: ARROW_WIDTH,
  arrowKeylineWidth: ARROW_KEYLINE_WIDTH,
  arrowHeadKeylineWidth: (ARROW_BORDER_WIDTH * 2) / ARROW_WIDTH,
  badgeRadius: BADGE_RADIUS,
  badgeFaceRadius: BADGE_FACE_RADIUS,
  badgeKeylineWidth: BADGE_KEYLINE_WIDTH,
} as const;

const ARROW_START_INSET = 0.16;
const ARROW_END_INSET = 0.3;
const ARROW_KEYLINE_END_INSET = ARROW_END_INSET + 0.12;
const ARROW_HEAD_BACK_LENGTH = 4.05 * ARROW_WIDTH;
const BADGE_OUTER_RADIUS = BADGE_RADIUS + BADGE_KEYLINE_WIDTH / 2;
const BADGE_START_CLEARANCE = BADGE_OUTER_RADIUS + 0.08;
const BADGE_END_CLEARANCE = BADGE_OUTER_RADIUS + ARROW_HEAD_BACK_LENGTH + 0.08;
const BADGE_SIDE_OFFSET = BADGE_OUTER_RADIUS + ARROW_KEYLINE_WIDTH / 2 + 0.05;
const BOARD_EXTENT = 8;
const EDGE_CLEARANCE_EPSILON = 1e-9;
const DANGER_ARROW_ROLES: ReadonlySet<ReviewArrow["role"]> = new Set([
  "reply",
  "threat",
]);

export function positionReviewArrows(
  arrows: ReviewArrow[],
  pointForSquare: PointForSquare,
): PositionedReviewArrow[] {
  return arrows.map((arrow) => {
    const line = arrowLine(arrow, pointForSquare);
    if (!line) {
      throw new ReviewDiagramError(
        "invalid_arrow",
        `Review arrow has invalid endpoints: ${arrow.from_square}-${arrow.to_square}`,
      );
    }
    return { arrow, line };
  });
}

export function positionReviewBadge(
  badge: ReviewDiagramBadge | null,
  arrows: PositionedReviewArrow[],
  pointForSquare: PointForSquare,
): PositionedReviewBadge | null {
  if (!badge) return null;
  if (arrows.length === 0) {
    throw new ReviewDiagramError(
      "missing_badge_arrow",
      "Review diagram badges require a positioned arrow",
    );
  }
  const anchor = pointForSquare(badge.square);
  if (!anchor) {
    throw new ReviewDiagramError(
      "invalid_badge_anchor",
      `Review diagram badge has an invalid anchor: ${badge.square}`,
    );
  }
  const positioned = arrows[badge.arrow_index];
  if (!positioned) {
    throw new ReviewDiagramError(
      "missing_badge_arrow",
      `Review diagram badge references missing arrow ${badge.arrow_index}`,
    );
  }

  return {
    badge,
    arrow: positioned.arrow,
    point: badgePoint(positioned.line, progressAlongLine(anchor, positioned.line)),
  };
}

export function visibleReviewMarkers(
  markers: ReviewSquareMarker[],
  arrows: PositionedReviewArrow[],
): ReviewSquareMarker[] {
  const rolesByTarget = new Map<string, Set<ReviewArrow["role"]>>();
  for (const { arrow } of arrows) {
    const roles = rolesByTarget.get(arrow.to_square) ?? new Set<ReviewArrow["role"]>();
    roles.add(arrow.role);
    rolesByTarget.set(arrow.to_square, roles);
  }

  return markers.filter((marker) => {
    if (marker.role === "vacated" || marker.role === "blocked") return true;
    const arrowRoles = rolesByTarget.get(marker.square);
    if (!arrowRoles) return true;
    if (marker.role !== "danger") return false;
    return ![...arrowRoles].some((role) => DANGER_ARROW_ROLES.has(role));
  });
}

function badgePoint(line: ReviewArrowLine, preferredProgress: number): BoardPoint {
  const length = lineLength(line);
  const minimumProgress = BADGE_START_CLEARANCE / length;
  const maximumProgress = 1 - BADGE_END_CLEARANCE / length;
  if (minimumProgress > maximumProgress) return pointBeside(line);
  return pointAlong(
    line,
    Math.min(maximumProgress, Math.max(minimumProgress, preferredProgress)),
  );
}

function pointBeside(line: ReviewArrowLine): BoardPoint {
  const midpoint = pointAlong(line, 0.5);
  const dx = line.end.x - line.start.x;
  const dy = line.end.y - line.start.y;
  const length = lineLength(line);
  const offset = {
    x: (-dy / length) * BADGE_SIDE_OFFSET,
    y: (dx / length) * BADGE_SIDE_OFFSET,
  };
  const first = { x: midpoint.x + offset.x, y: midpoint.y + offset.y };
  const second = { x: midpoint.x - offset.x, y: midpoint.y - offset.y };
  return edgeClearance(first) + EDGE_CLEARANCE_EPSILON >= edgeClearance(second)
    ? first
    : second;
}

function edgeClearance(point: BoardPoint): number {
  return Math.min(point.x, point.y, BOARD_EXTENT - point.x, BOARD_EXTENT - point.y);
}

function lineLength(line: ReviewArrowLine): number {
  return Math.hypot(line.end.x - line.start.x, line.end.y - line.start.y);
}

function arrowLine(
  arrow: ReviewArrow,
  pointForSquare: PointForSquare,
): ReviewArrowLine | null {
  const from = pointForSquare(arrow.from_square);
  const to = pointForSquare(arrow.to_square);
  if (!from || !to) return null;
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const distance = Math.hypot(dx, dy);
  if (distance === 0) return null;
  const unitX = dx / distance;
  const unitY = dy / distance;
  return {
    start: {
      x: from.x + unitX * ARROW_START_INSET,
      y: from.y + unitY * ARROW_START_INSET,
    },
    end: {
      x: to.x - unitX * ARROW_END_INSET,
      y: to.y - unitY * ARROW_END_INSET,
    },
    keylineEnd: {
      x: to.x - unitX * ARROW_KEYLINE_END_INSET,
      y: to.y - unitY * ARROW_KEYLINE_END_INSET,
    },
  };
}

function progressAlongLine(point: BoardPoint, line: ReviewArrowLine): number {
  const dx = line.end.x - line.start.x;
  const dy = line.end.y - line.start.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) return 0.5;
  const progress =
    ((point.x - line.start.x) * dx + (point.y - line.start.y) * dy) /
    lengthSquared;
  return Math.min(1, Math.max(0, progress));
}

function pointAlong(line: ReviewArrowLine, progress: number): BoardPoint {
  return {
    x: line.start.x + (line.end.x - line.start.x) * progress,
    y: line.start.y + (line.end.y - line.start.y) * progress,
  };
}
