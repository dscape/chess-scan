import assert from "node:assert/strict";
import test from "node:test";

import {
  cornerStability,
  cornersAreStable,
  minimumBoardEdge,
  polygonCenter,
  projectedBoardCell,
  quadrilateralIoU,
  requiredCaptureBoardEdge,
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
    cornersAreStable(board, [[160, 100], [560, 100], [560, 500], [160, 500]]),
    false,
  );
  assert.ok(cornerStability(board, [[105, 102], [505, 103], [504, 503], [104, 502]]) > 0.8);
  assert.ok(cornerStability(board, [[160, 100], [560, 100], [560, 500], [160, 500]]) < 0.7);
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

test("maps classifier squares onto the projected camera grid", () => {
  const grid = Array.from({ length: 81 }, (_, index) => [index % 9, Math.floor(index / 9)]);

  assert.deepEqual(projectedBoardCell(grid, 9), [[1, 1], [2, 1], [2, 2], [1, 2]]);
  assert.deepEqual(polygonCenter(projectedBoardCell(grid, 9)), [1.5, 1.5]);
  assert.equal(projectedBoardCell(grid, 64), null);
  assert.equal(projectedBoardCell([], 0), null);
});

test("scales the capture resolution floor instead of tying it to preview size", () => {
  assert.equal(requiredCaptureBoardEdge(900, 1600), 384);
  assert.equal(requiredCaptureBoardEdge(640, 1080), 320);
  assert.throws(() => requiredCaptureBoardEdge(0, 1080), RangeError);

  const portraitPreview = [
    [125.5, 277.7], [331.2, 272.6], [338.9, 481.9], [129.6, 483.4],
  ];
  const retained = scaleCorners(portraitPreview, 405, 720, 900, 1600);
  assert.ok(minimumBoardEdge(portraitPreview) < 256);
  assert.ok(minimumBoardEdge(retained) > requiredCaptureBoardEdge(900, 1600));
});

test("uses the shortest projected board edge as the resolution floor", () => {
  assert.equal(minimumBoardEdge(board), 400);
  assert.equal(minimumBoardEdge([[0, 0], [300, 0], [280, 200], [0, 200]]), 200);
  assert.equal(minimumBoardEdge([]), 0);
});
