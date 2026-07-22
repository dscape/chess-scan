import assert from "node:assert/strict";
import test from "node:test";

import { boardPoint } from "../src/board.ts";
import {
  positionReviewArrows,
  positionReviewBadge,
  ReviewDiagramError,
  visibleReviewMarkers,
} from "../src/reviewDiagram.ts";

const forkArrows = [
  { from_square: "e2", to_square: "e4", role: "engine" },
  { from_square: "e4", to_square: "c6", role: "attack" },
  { from_square: "e4", to_square: "h7", role: "attack" },
];
const whitePoint = (square) => boardPoint(square, "white");
const blackPoint = (square) => boardPoint(square, "black");

test("attaches a fork badge to its declared branch instead of its square", () => {
  const arrows = positionReviewArrows(forkArrows, whitePoint);
  const badge = positionReviewBadge(
    { kind: "fork", square: "e4", role: "engine", arrow_index: 1 },
    arrows,
    whitePoint,
  );

  assert.equal(arrows.length, 3);
  assert.equal(badge?.arrow.from_square, "e4");
  assert.equal(badge?.arrow.to_square, "c6");
  assert.ok(badge && badge.point.x < 4.5 && badge.point.x > 2.5);
  assert.ok(badge && badge.point.y < 4.5 && badge.point.y > 2.5);
});

test("moves a short-arrow badge beside the shaft to preserve direction", () => {
  const arrows = positionReviewArrows(
    [{ from_square: "a2", to_square: "a3", role: "engine" }],
    whitePoint,
  );
  const badge = positionReviewBadge(
    { kind: "engine", square: "a3", role: "engine", arrow_index: 0 },
    arrows,
    whitePoint,
  );

  assert.ok(badge && arrows[0]);
  assert.ok(Math.abs(badge.point.x - arrows[0].line.start.x) > 0.4);
  assert.equal(
    badge.point.y,
    (arrows[0].line.start.y + arrows[0].line.end.y) / 2,
  );
});

test("mirrors a side-offset diagonal badge with the board orientation", () => {
  const reviewArrow = [{ from_square: "e5", to_square: "d4", role: "engine" }];
  const badge = {
    kind: "engine",
    square: "d4",
    role: "engine",
    arrow_index: 0,
  };
  const white = positionReviewBadge(
    badge,
    positionReviewArrows(reviewArrow, whitePoint),
    whitePoint,
  );
  const black = positionReviewBadge(
    badge,
    positionReviewArrows(reviewArrow, blackPoint),
    blackPoint,
  );

  assert.ok(white && black);
  assert.ok(Math.abs(black.point.x - (8 - white.point.x)) < 1e-9);
  assert.ok(Math.abs(black.point.y - (8 - white.point.y)) < 1e-9);
});

test("rejects a badge without an arrow instead of drawing a blank cue", () => {
  assert.throws(
    () =>
      positionReviewBadge(
        { kind: "engine", square: "a3", role: "engine", arrow_index: 0 },
        [],
        whitePoint,
      ),
    (error) =>
      error instanceof ReviewDiagramError && error.code === "missing_badge_arrow",
  );
});

test("rejects invalid arrows without shifting badge indices", () => {
  assert.throws(
    () =>
      positionReviewArrows(
        [{ from_square: "z9", to_square: "a3", role: "engine" }],
        whitePoint,
      ),
    (error) => error instanceof ReviewDiagramError && error.code === "invalid_arrow",
  );
});

test("keeps a future-capture badge on the dashed threat arrow", () => {
  const arrows = positionReviewArrows(
    [
      { from_square: "a1", to_square: "a4", role: "engine" },
      { from_square: "a4", to_square: "c4", role: "threat" },
    ],
    whitePoint,
  );
  const badge = positionReviewBadge(
    { kind: "capture", square: "c4", role: "threat", arrow_index: 1 },
    arrows,
    whitePoint,
  );

  assert.equal(badge?.arrow.role, "threat");
  assert.equal(badge?.arrow.from_square, "a4");
  assert.equal(badge?.arrow.to_square, "c4");
  assert.ok(badge && badge.point.x > 0.5 && badge.point.x < 2.5);
});

test("removes endpoint halos that duplicate arrows but keeps state markers", () => {
  const arrows = positionReviewArrows(forkArrows, whitePoint);
  const markers = visibleReviewMarkers(
    [
      { square: "c6", role: "target" },
      { square: "h7", role: "danger" },
      { square: "e4", role: "focus" },
      { square: "e2", role: "vacated" },
      { square: "d4", role: "blocked" },
    ],
    arrows,
  );

  assert.deepEqual(markers, [
    { square: "h7", role: "danger" },
    { square: "e2", role: "vacated" },
    { square: "d4", role: "blocked" },
  ]);
});

test("keeps danger halos unless a red arrow already conveys danger", () => {
  const played = positionReviewArrows(
    [{ from_square: "a1", to_square: "a4", role: "played" }],
    whitePoint,
  );
  const reply = positionReviewArrows(
    [{ from_square: "a1", to_square: "a4", role: "reply" }],
    whitePoint,
  );
  const danger = [{ square: "a4", role: "danger" }];

  assert.deepEqual(visibleReviewMarkers(danger, played), danger);
  assert.deepEqual(visibleReviewMarkers(danger, reply), []);
});

test("mirrors arrow geometry with the board orientation", () => {
  const white = positionReviewArrows(
    [{ from_square: "a1", to_square: "a8", role: "engine" }],
    whitePoint,
  )[0];
  const black = positionReviewArrows(
    [{ from_square: "a1", to_square: "a8", role: "engine" }],
    blackPoint,
  )[0];

  assert.ok(white && black);
  assert.equal(white.line.start.x, 0.5);
  assert.equal(black.line.start.x, 7.5);
  assert.equal(white.line.start.y, 7.34);
  assert.equal(black.line.start.y, 0.66);
});
