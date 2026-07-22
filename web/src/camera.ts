import type { Point } from "./types";

export const MIN_PREVIEW_BOARD_EDGE = 256;

export function minimumBoardEdge(corners: Point[]): number {
  if (corners.length !== 4) return 0;
  return Math.min(
    ...corners.map(([x, y], index) => {
      const next = corners[(index + 1) % corners.length];
      return next ? Math.hypot(next[0] - x, next[1] - y) : 0;
    }),
  );
}

export function cornersAreStable(previous: Point[] | null, current: Point[]): boolean {
  if (!previous || previous.length !== 4 || current.length !== 4) return false;
  const boardEdge = minimumBoardEdge(current);
  if (boardEdge <= 0 || quadrilateralIoU(previous, current) < 0.9) return false;

  const maximumDrift = Math.max(
    ...current.map(([x, y], index) => {
      const prior = previous[index];
      return prior ? Math.hypot(x - prior[0], y - prior[1]) : boardEdge;
    }),
  );
  return maximumDrift / boardEdge < 0.04;
}

export function scaleCorners(
  corners: Point[],
  sourceWidth: number,
  sourceHeight: number,
  targetWidth: number,
  targetHeight: number,
): Point[] {
  if (sourceWidth <= 0 || sourceHeight <= 0 || targetWidth <= 0 || targetHeight <= 0) {
    throw new RangeError("Image dimensions must be positive");
  }
  const scaleX = targetWidth / sourceWidth;
  const scaleY = targetHeight / sourceHeight;
  return corners.map(([x, y]) => [x * scaleX, y * scaleY]);
}

export function quadrilateralIoU(first: Point[], second: Point[]): number {
  if (first.length !== 4 || second.length !== 4) return 0;
  const firstArea = polygonArea(first);
  const secondArea = polygonArea(second);
  if (firstArea <= 0 || secondArea <= 0) return 0;
  const intersectionArea = polygonArea(intersectConvexPolygons(first, second));
  const unionArea = firstArea + secondArea - intersectionArea;
  return unionArea > 0 ? intersectionArea / unionArea : 0;
}

function intersectConvexPolygons(subject: Point[], clip: Point[]): Point[] {
  let output = subject.slice();
  const orientation = Math.sign(signedPolygonArea(clip)) || 1;
  for (let index = 0; index < clip.length; index += 1) {
    const edgeStart = clip[index];
    const edgeEnd = clip[(index + 1) % clip.length];
    if (!edgeStart || !edgeEnd || output.length === 0) return [];

    const input = output;
    output = [];
    let segmentStart = input[input.length - 1];
    if (!segmentStart) return [];
    for (const segmentEnd of input) {
      const endInside = isInside(segmentEnd, edgeStart, edgeEnd, orientation);
      const startInside = isInside(segmentStart, edgeStart, edgeEnd, orientation);
      if (endInside) {
        if (!startInside) output.push(lineIntersection(segmentStart, segmentEnd, edgeStart, edgeEnd));
        output.push(segmentEnd);
      } else if (startInside) {
        output.push(lineIntersection(segmentStart, segmentEnd, edgeStart, edgeEnd));
      }
      segmentStart = segmentEnd;
    }
  }
  return output;
}

function isInside(point: Point, edgeStart: Point, edgeEnd: Point, orientation: number): boolean {
  return orientation * cross(
    edgeEnd[0] - edgeStart[0],
    edgeEnd[1] - edgeStart[1],
    point[0] - edgeStart[0],
    point[1] - edgeStart[1],
  ) >= -1e-6;
}

function lineIntersection(start: Point, end: Point, edgeStart: Point, edgeEnd: Point): Point {
  const segmentX = end[0] - start[0];
  const segmentY = end[1] - start[1];
  const edgeX = edgeEnd[0] - edgeStart[0];
  const edgeY = edgeEnd[1] - edgeStart[1];
  const denominator = cross(segmentX, segmentY, edgeX, edgeY);
  if (Math.abs(denominator) < 1e-8) return end;
  const offsetX = edgeStart[0] - start[0];
  const offsetY = edgeStart[1] - start[1];
  const distance = cross(offsetX, offsetY, edgeX, edgeY) / denominator;
  return [start[0] + distance * segmentX, start[1] + distance * segmentY];
}

function polygonArea(points: Point[]): number {
  return Math.abs(signedPolygonArea(points));
}

function signedPolygonArea(points: Point[]): number {
  let twiceArea = 0;
  for (let index = 0; index < points.length; index += 1) {
    const point = points[index];
    const next = points[(index + 1) % points.length];
    if (point && next) twiceArea += point[0] * next[1] - next[0] * point[1];
  }
  return twiceArea / 2;
}

function cross(ax: number, ay: number, bx: number, by: number): number {
  return ax * by - ay * bx;
}
