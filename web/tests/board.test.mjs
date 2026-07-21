import assert from "node:assert/strict";
import test from "node:test";

import { fenError, pieceDisplay } from "../src/board.ts";

test("accepts a FEN that can be loaded into a lesson", () => {
  assert.equal(fenError("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), null);
});

test("maps chess.js pieces to the shared board metadata", () => {
  assert.deepEqual(pieceDisplay("w", "n"), { name: "White knight", symbol: "♘" });
  assert.deepEqual(pieceDisplay("b", "q"), { name: "Black queen", symbol: "♛" });
});

test("rejects a FEN with invalid king counts", () => {
  assert.equal(
    fenError("2R1K1nr/pp3ppp/q1n1p3/2bpP3/P7/1PP2N2/2Q2PPP/RNB1K2R w - - 0 1"),
    "Invalid FEN: too many white kings",
  );
});
