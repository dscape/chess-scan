import assert from "node:assert/strict";
import test from "node:test";

import {
  boardPoint,
  fenError,
  parentFensForMoves,
  pieceName,
  pieceOptionForLabel,
  positionAt,
} from "../src/board.ts";

test("accepts a FEN that can be loaded into a review", () => {
  assert.equal(fenError("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), null);
});

test("maps chess.js pieces to the shared board metadata", () => {
  assert.equal(pieceName("w", "n"), "White knight");
  assert.equal(pieceName("b", "q"), "Black queen");
});

test("maps classifier labels to canonical piece metadata", () => {
  assert.deepEqual(pieceOptionForLabel(2), {
    name: "White knight",
    fenSymbol: "N",
    piece: { color: "w", type: "n" },
  });
  assert.deepEqual(pieceOptionForLabel(11), {
    name: "Black queen",
    fenSymbol: "q",
    piece: { color: "b", type: "q" },
  });
  assert.deepEqual(pieceOptionForLabel(0), { name: "Empty", fenSymbol: "", piece: null });
  assert.throws(() => pieceOptionForLabel(13), RangeError);
  assert.throws(() => pieceOptionForLabel(1.5), RangeError);
});

test("replays review moves through one validated board helper", () => {
  const position = positionAt(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ["e2e4", "e7e5"],
  );

  assert.equal(position.fen(), "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2");
  assert.throws(() => positionAt(position.fen(), ["e2e5"]), /Illegal review move/);
});

test("records parent positions while replaying a line once", () => {
  const root = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

  assert.deepEqual(parentFensForMoves(root, ["e2e4", "e7e5"]), [
    root,
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  ]);
  assert.throws(
    () => parentFensForMoves(root, ["e2e5"]),
    /Illegal review move/,
  );
});

test("maps diagram geometry for both board orientations", () => {
  assert.deepEqual(boardPoint("a8", "white"), { x: 0.5, y: 0.5 });
  assert.deepEqual(boardPoint("h1", "white"), { x: 7.5, y: 7.5 });
  assert.deepEqual(boardPoint("a8", "black"), { x: 7.5, y: 7.5 });
  assert.deepEqual(boardPoint("h1", "black"), { x: 0.5, y: 0.5 });
  assert.equal(boardPoint("z9", "white"), null);
});

test("rejects a FEN with invalid king counts", () => {
  assert.equal(
    fenError("2R1K1nr/pp3ppp/q1n1p3/2bpP3/P7/1PP2N2/2Q2PPP/RNB1K2R w - - 0 1"),
    "Invalid FEN: too many white kings",
  );
});
