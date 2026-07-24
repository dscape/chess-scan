import test from "node:test";
import assert from "node:assert/strict";
import {
  coachingMoveColor,
  coachingMoveCue,
  coachingMoveDisplay,
  coachingMoveLabel,
  hasCoachingNarrative,
} from "../src/coaching.ts";

test("renders SAN moves as color-aware piece tokens", () => {
  assert.deepEqual(coachingMoveDisplay("Rh4+", "w"), {
    symbol: "♖",
    notation: "h4+",
  });
  assert.deepEqual(coachingMoveDisplay("Nxd4!", "b"), {
    symbol: "♞",
    notation: "xd4!",
  });
  assert.deepEqual(coachingMoveDisplay("e8=Q+", "w"), {
    symbol: "♙",
    notation: "e8=Q+",
  });
  assert.deepEqual(coachingMoveDisplay("O-O-O", "b"), {
    symbol: "♚",
    notation: "O-O-O",
  });
});

test("hides candidate-free and disabled coaching", () => {
  assert.equal(hasCoachingNarrative({ status: "fallback", sections: [] }), false);
  assert.equal(hasCoachingNarrative({ status: "disabled", sections: [] }), false);
  assert.equal(
    hasCoachingNarrative({ status: "fallback", sections: [{ kind: "diagnosis" }] }),
    true,
  );
});

test("names each move token with its checked-line role", () => {
  assert.equal(
    coachingMoveLabel({
      scope: "attempt_refutation",
      ply: 1,
      role: "reply",
      move: { san: "Nxd4" },
    }),
    "Hypothetical reply line, line step 2: show Nxd4 on the board",
  );
  assert.equal(
    coachingMoveLabel({
      scope: "best_line",
      ply: 0,
      role: "better",
      move: { san: "Qd1" },
    }),
    "Engine line, line step 1: show Qd1 on the board",
  );
  assert.equal(
    coachingMoveLabel({
      scope: "attempt_refutation",
      ply: 0,
      role: "attempt",
      move: { san: "Nd4" },
    }),
    "Learner attempt, line step 1: show Nd4 on the board",
  );
  assert.notEqual(
    coachingMoveLabel({
      scope: "best_line",
      ply: 0,
      role: "line",
      move: { san: "Nc3" },
    }),
    coachingMoveLabel({
      scope: "best_line",
      ply: 4,
      role: "line",
      move: { san: "Nc3" },
    }),
  );
});

test("alternates move-token color from the root side to move", () => {
  assert.equal(coachingMoveColor("b", 0), "b");
  assert.equal(coachingMoveColor("b", 1), "w");
  assert.equal(coachingMoveColor("b", 2), "b");
});

test("rejects malformed move references before drawing an arrow", () => {
  assert.throws(
    () =>
      coachingMoveCue({
        scope: "best_line",
        ply: 0,
        role: "line",
        move: { uci: "bad", san: "Qd1" },
      }),
    /invalid UCI move/,
  );
});

test("turns a checked move reference into a board annotation", () => {
  assert.deepEqual(
    coachingMoveCue({
      type: "move",
      scope: "attempt_refutation",
      ply: 1,
      role: "reply",
      move: { uci: "c6d4", san: "Nxd4" },
    }),
    {
      id: "coach-attempt-refutation-1",
      label: "Nxd4",
      text: "Show Nxd4 on the board.",
      scope: "attempt_refutation",
      ply: 1,
      markers: [],
      arrows: [{ from_square: "c6", to_square: "d4", role: "reply" }],
      badge: null,
      evidence_ids: [],
    },
  );
});
