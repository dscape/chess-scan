import assert from "node:assert/strict";
import test from "node:test";

import {
  isStablePrimary,
  parseBestMove,
  parseInfoLine,
  parseUciMove,
} from "../src/engine/uci.ts";

test("parses a complete MultiPV line without conflating score and WDL", () => {
  const line = parseInfoLine(
    "info depth 24 seldepth 31 multipv 2 score cp -137 nodes 456789 wdl 12 341 647 pv e2e4 e7e5 g1f3",
  );

  assert.deepEqual(line, {
    multipv: 2,
    depth: 24,
    score: { kind: "cp", value: -137 },
    wdl: [12, 341, 647],
    pv: ["e2e4", "e7e5", "g1f3"],
  });
});

test("preserves mate scores and bound status", () => {
  assert.deepEqual(
    parseInfoLine("info depth 18 score mate 4 lowerbound pv f7f8 h8h7 f8g7"),
    {
      multipv: 1,
      depth: 18,
      score: { kind: "mate", value: 4, bound: "lower" },
      pv: ["f7f8", "h8h7", "f8g7"],
    },
  );
});

test("rejects incomplete info and invalid WDL while bounding the PV", () => {
  assert.equal(parseInfoLine("info depth 12 nodes 42"), null);
  const moves = Array.from({ length: 30 }, (_, index) => index % 2 === 0 ? "e2e4" : "e7e5");
  const line = parseInfoLine(`info depth 12 score cp 10 wdl 1 2 3 pv ${moves.join(" ")}`);

  assert.equal(line?.wdl, undefined);
  assert.equal(line?.pv.length, 24);
});

test("rejects malformed PV moves and explicit invalid MultiPV indexes", () => {
  assert.equal(
    parseInfoLine("info depth 12 multipv 1 score cp 10 pv e2e4 broken e7e5"),
    null,
  );
  assert.equal(parseInfoLine("info depth 12 multipv 0 score cp 10 pv e2e4"), null);
  assert.equal(parseInfoLine("info depth 12 multipv 6 score cp 10 pv e2e4"), null);
});

test("requires repeated exact principal moves before teaching from a line", () => {
  const primary = parseInfoLine("info depth 18 score cp 42 pv e2e4 e7e5");

  assert.equal(isStablePrimary("e2e4", primary, ["e2e4", "e2e4", "e2e4"]), true);
  assert.equal(isStablePrimary("e2e4", primary, ["d2d4", "e2e4"]), false);
  assert.equal(isStablePrimary("d2d4", primary, ["d2d4", "d2d4"]), false);
  assert.equal(
    isStablePrimary(
      "e2e4",
      parseInfoLine("info depth 19 score cp 50 upperbound pv e2e4 e7e5"),
      ["e2e4", "e2e4"],
    ),
    false,
  );
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
