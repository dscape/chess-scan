import assert from "node:assert/strict";
import test from "node:test";

import {
  cornersAreStable,
  minimumBoardEdge,
  quadrilateralIoU,
  scaleCorners,
} from "../src/camera.ts";

const board = [[100, 100], [500, 100], [500, 500], [100, 500]];

test("measures board-relative stability", () => {
  assert.equal(cornersAreStable(null, board), false);
  assert.equal(
    cornersAreStable(board, [[105, 102], [505, 103], [504, 503], [104, 502]]),
    true,
  );
  assert.equal(
    cornersAreStable(board, [[125, 100], [525, 100], [525, 500], [125, 500]]),
    false,
  );
});

test("computes convex quadrilateral overlap", () => {
  assert.equal(quadrilateralIoU(board, board), 1);
  assert.equal(
    quadrilateralIoU(board, [[500, 100], [900, 100], [900, 500], [500, 500]]),
    0,
  );
  assert.ok(
    Math.abs(
      quadrilateralIoU(board, [[300, 100], [700, 100], [700, 500], [300, 500]]) - 1 / 3,
    ) < 1e-9,
  );
});

test("scales detected corners onto the retained camera frame", () => {
  assert.deepEqual(
    scaleCorners([[10, 20], [90, 20], [90, 80], [10, 80]], 100, 100, 200, 300),
    [[20, 60], [180, 60], [180, 240], [20, 240]],
  );
  assert.throws(() => scaleCorners(board, 0, 100, 200, 200), RangeError);
});

test("uses the shortest projected board edge as the resolution floor", () => {
  assert.equal(minimumBoardEdge(board), 400);
  assert.equal(minimumBoardEdge([[0, 0], [300, 0], [280, 200], [0, 200]]), 200);
  assert.equal(minimumBoardEdge([]), 0);
});
