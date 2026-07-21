import assert from "node:assert/strict";
import test from "node:test";

import {
  contiguousRankedLines,
  isStableLine,
  parseBestMove,
  parseInfoLine,
  parseUciMove,
} from "../src/engine/uci.ts";

test("parses ranked engine lines with WDL evidence", () => {
  const line = parseInfoLine(
    "info depth 24 seldepth 31 multipv 2 score cp -137 wdl 120 330 550 nodes 456789 pv e2e4 e7e5 g1f3",
  );

  assert.deepEqual(line, {
    rank: 2,
    depth: 24,
    score: { kind: "cp", value: -137 },
    wdl: [120, 330, 550],
    pv: ["e2e4", "e7e5", "g1f3"],
  });
});

test("preserves mate scores and bound status", () => {
  assert.deepEqual(
    parseInfoLine("info depth 18 score mate 4 lowerbound pv f7f8 h8h7 f8g7"),
    {
      rank: 1,
      depth: 18,
      score: { kind: "mate", value: 4, bound: "lower" },
      pv: ["f7f8", "h8h7", "f8g7"],
    },
  );
});

test("rejects incomplete info while bounding the principal variation", () => {
  assert.equal(parseInfoLine("info depth 12 nodes 42"), null);
  const moves = Array.from({ length: 30 }, (_, index) => index % 2 === 0 ? "e2e4" : "e7e5");
  const line = parseInfoLine(`info depth 12 score cp 10 pv ${moves.join(" ")}`);

  assert.equal(line?.pv.length, 16);
});

test("rejects malformed moves, ranks, and WDL", () => {
  assert.equal(parseInfoLine("info depth 12 score cp 10 pv e2e4 broken e7e5"), null);
  assert.equal(parseInfoLine("info depth 12 multipv 4 score cp 10 pv e2e4"), null);
  assert.deepEqual(
    parseInfoLine("info depth 12 score cp 10 wdl 1 2 3 pv e2e4"),
    parseInfoLine("info depth 12 score cp 10 pv e2e4"),
  );
});

test("requires repeated matching principal moves before teaching", () => {
  const primary = parseInfoLine("info depth 18 score cp 42 pv e2e4 e7e5");

  assert.equal(isStableLine("e2e4", primary ?? undefined, ["e2e4", "e2e4"]), true);
  assert.equal(isStableLine("d2d4", primary ?? undefined, ["e2e4", "e2e4"]), false);
  assert.equal(isStableLine("e2e4", primary ?? undefined, ["e2e4", "d2d4"]), false);
  assert.equal(
    isStableLine(
      "e2e4",
      parseInfoLine("info depth 19 score cp 50 upperbound pv e2e4 e7e5") ?? undefined,
      ["e2e4", "e2e4"],
    ),
    false,
  );
});

test("drops gapped MultiPV ranks before sending analysis to the API", () => {
  const first = parseInfoLine("info depth 18 multipv 1 score cp 50 pv e2e4 e7e5");
  const third = parseInfoLine("info depth 18 multipv 3 score cp 20 pv g1f3 g8f6");

  assert.deepEqual(contiguousRankedLines([third, first].filter(Boolean)), [first]);
});

test("parses normal and terminal bestmove messages with the shared move grammar", () => {
  assert.equal(parseBestMove("bestmove e2e4 ponder e7e5"), "e2e4");
  assert.equal(parseBestMove("bestmove (none)"), "(none)");
  assert.equal(parseBestMove("bestmove e2e4oops"), null);
  assert.equal(parseBestMove("info string no best move"), null);
  assert.deepEqual(parseUciMove("a7a8n"), {
    from: "a7",
    to: "a8",
    promotion: "n",
  });
  assert.equal(parseUciMove("e2e2"), null);
});
