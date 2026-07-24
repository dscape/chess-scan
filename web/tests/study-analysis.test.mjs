import assert from "node:assert/strict";
import test from "node:test";
import { Chess } from "chess.js";

import {
  StudyAnalysisError,
  attemptLineFromChild,
  canUseCachedBestGrade,
  classifyStudyMove,
  prepareStudyRetry,
  rootEvaluationScore,
  sanForEngineMove,
  settledStudyState,
  studyBoardCue,
  studyEngineArrow,
  studyPathKey,
  studyPositionKey,
  studyStateForKey,
  terminalAttemptLine,
  terminalStudyPosition,
} from "../src/studyAnalysis.ts";

function candidate(move, wdl) {
  return {
    role: "best_candidate",
    rank: 1,
    depth: 18,
    score: { kind: "cp", value: 100 },
    wdl,
    moves: [{ uci: move, san: move }],
  };
}

function parentReview({
  bestMove = "e2e4",
  bestWdl = [600, 300, 100],
  alternativeWdl = [500, 300, 200],
  sacrifice = false,
} = {}) {
  return {
    best_move: { uci: bestMove, san: "e4" },
    lines: [
      candidate(bestMove, bestWdl),
      { ...candidate("d2d4", alternativeWdl), role: "alternative_candidate", rank: 2 },
    ],
    findings: sacrifice
      ? [{ topic: { id: "temporary-sacrifice", name: "Temporary sacrifice" }, evidence_ids: ["sacrifice"] }]
      : [],
    evidence: sacrifice
      ? [{
          id: "sacrifice",
          kind: "temporary_sacrifice",
          from_square: bestMove.slice(0, 2),
          to_square: bestMove.slice(2, 4),
        }]
      : [],
    explanation: [],
  };
}

function comparison(verdict, move = "e2e4") {
  return {
    move: { uci: move, san: move === "e2e4" ? "e4" : move },
    headline: `${move} is ${verdict}.`,
    verdict,
  };
}

function attempted(uci = "e2e4", san = "e4") {
  return { uci, san };
}

function apiFailure(message, status = 503, retryable = true) {
  return { kind: "api", message, status, retryable };
}

test("shares position analysis across transpositions but keeps path grades distinct", () => {
  const first = ["g1f3", "g8f6", "b1c3", "b8c6"];
  const second = ["b1c3", "b8c6", "g1f3", "g8f6"];
  const firstPosition = new Chess();
  const secondPosition = new Chess();
  for (const move of first) firstPosition.move(move);
  for (const move of second) secondPosition.move(move);

  assert.equal(studyPositionKey(firstPosition.fen()), studyPositionKey(secondPosition.fen()));
  assert.notEqual(
    studyPathKey(firstPosition.fen(), first),
    studyPathKey(secondPosition.fen(), second),
  );
});

test("canonicalizes a non-capturable root en-passant square", () => {
  const confirmed = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";

  assert.equal(
    studyPositionKey(confirmed),
    "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
  );
});

test("keeps a node locked when its current-position review failed", () => {
  const state = settledStudyState(
    "node",
    {
      lines: [candidate("e2e4", [600, 300, 100])],
      topMoveSan: "e4",
      review: null,
      reviewError: apiFailure("Review failed."),
      terminal: false,
    },
    { grade: null, error: null },
  );

  assert.equal(state.status, "review-error");
  assert.equal(state.failure.message, "Review failed.");
  assert.equal(state.node.review, null);
});

test("retries only failed API data while retaining stable engine analysis", () => {
  const position = {
    lines: [candidate("e2e4", [600, 300, 100])],
    topMoveSan: "e4",
    review: null,
    reviewError: apiFailure("Review failed."),
    terminal: false,
  };
  const retry = prepareStudyRetry(
    position,
    { grade: null, error: apiFailure("Grade failed.") },
  );

  assert.equal(retry.position.lines, position.lines);
  assert.equal(retry.position.topMoveSan, "e4");
  assert.equal(retry.position.reviewError, null);
  assert.equal(retry.gradeResult, null);
});

test("does not retry a permanent API failure", () => {
  const position = {
    lines: [candidate("e2e4", [600, 300, 100])],
    topMoveSan: "e4",
    review: null,
    reviewError: apiFailure("Invalid analysis.", 422, false),
    terminal: false,
  };
  const gradeResult = {
    grade: null,
    error: apiFailure("Invalid comparison.", 422, false),
  };
  const retry = prepareStudyRetry(position, gradeResult);

  assert.equal(retry.position, position);
  assert.equal(retry.gradeResult, gradeResult);
});

test("retains successful API data during a retry", () => {
  const position = {
    lines: [candidate("e2e4", [600, 300, 100])],
    topMoveSan: "e4",
    review: parentReview(),
    reviewError: null,
    terminal: false,
  };
  const gradeResult = { grade: null, error: null };
  const retry = prepareStudyRetry(position, gradeResult);

  assert.equal(retry.position, position);
  assert.equal(retry.gradeResult, gradeResult);
});

test("filters a stale study state before rendering a new path", () => {
  const stale = settledStudyState(
    "old-path",
    terminalStudyPosition(),
    { grade: null, error: null },
  );

  assert.deepEqual(studyStateForKey(stale, "new-path"), { status: "idle" });
  assert.equal(studyStateForKey(stale, "old-path"), stale);
});

test("restores the root score after its LRU entry is gone", () => {
  const score = { kind: "cp", value: 135 };

  assert.equal(rootEvaluationScore([{ score }], null, null), score);
  assert.deepEqual(
    rootEvaluationScore(undefined, null, { kind: "cp", value: 90 }),
    { kind: "cp", value: 90 },
  );
});

test("allows a locally settled terminal node without a current review", () => {
  const state = settledStudyState(
    "terminal",
    terminalStudyPosition(),
    { grade: null, error: null },
  );

  assert.equal(state.status, "ready");
  assert.equal(state.node.terminal, true);
});

test("builds a decisive parent-side line for a terminal study move", () => {
  const terminal = new Chess(
    "r3kbQ1/ppp2p1p/6p1/3P1b2/2q5/N4P2/PP1P2PP/n1BK1BNR b - - 0 1",
  );
  terminal.move("c4f1");

  assert.equal(terminal.isCheckmate(), true);
  assert.deepEqual(terminalAttemptLine("c4f1", terminal, 21), {
    rank: 1,
    depth: 21,
    score: { kind: "mate", value: 1 },
    wdl: [1000, 0, 0],
    pv: ["c4f1"],
    stable: true,
  });
});

test("requires a path-aware comparison for a terminal draw", () => {
  assert.equal(canUseCachedBestGrade("draw"), false);
  assert.equal(canUseCachedBestGrade("white"), true);
  assert.equal(canUseCachedBestGrade(null), true);
});

test("builds a draw line locally for a threefold repetition", () => {
  const position = new Chess();
  const moves = [
    "g1f3", "g8f6", "f3g1", "f6g8",
    "g1f3", "g8f6", "f3g1", "f6g8",
  ];
  for (const move of moves) position.move(move);

  assert.equal(position.isThreefoldRepetition(), true);
  assert.deepEqual(terminalAttemptLine("f6g8", position, 18), {
    rank: 1,
    depth: 18,
    score: { kind: "cp", value: 0 },
    wdl: [0, 1000, 0],
    pv: ["f6g8"],
    stable: true,
  });
});

test("converts a child line back to the parent point of view", () => {
  assert.deepEqual(
    attemptLineFromChild("e2e4", {
      rank: 2,
      depth: 19,
      score: { kind: "cp", value: -85, bound: "lower" },
      wdl: [100, 300, 600],
      pv: ["e7e5", "g1f3"],
      stable: true,
    }),
    {
      rank: 1,
      depth: 19,
      score: { kind: "cp", value: 85, bound: "upper" },
      wdl: [600, 300, 100],
      pv: ["e2e4", "e7e5", "g1f3"],
      stable: true,
    },
  );
});

test("adds the attempted ply to a nonterminal winning mate distance", () => {
  const line = attemptLineFromChild("e2e4", {
    rank: 1,
    depth: 22,
    score: { kind: "mate", value: -1 },
    wdl: [0, 0, 1000],
    pv: ["e7e5"],
    stable: true,
  });

  assert.deepEqual(line.score, { kind: "mate", value: 2 });
  assert.deepEqual(line.wdl, [1000, 0, 0]);
});

test("grades a cached rank-one sacrifice without a comparison response", () => {
  const grade = classifyStudyMove(
    attempted(),
    parentReview({ sacrifice: true }),
    null,
  );

  assert.equal(grade?.kind, "brilliant");
  assert.equal(grade?.square, "e4");
});

test("honors a path-aware verdict for the cached best move", () => {
  assert.equal(
    classifyStudyMove(
      attempted(),
      parentReview(),
      comparison("blunder"),
    )?.kind,
    "blunder",
  );
});

test("does not call an unrelated sacrifice brilliant", () => {
  const parent = parentReview({ sacrifice: true });
  parent.evidence[0].from_square = "a1";
  parent.evidence[0].to_square = "a2";

  assert.equal(classifyStudyMove(attempted(), parent, null)?.kind, "good");
});

test("only marks the best move good when the runner-up loses meaningful expected score", () => {
  assert.equal(classifyStudyMove(attempted(), parentReview(), null)?.kind, "good");
  assert.equal(
    classifyStudyMove(
      attempted(),
      parentReview({ alternativeWdl: [550, 300, 150] }),
      null,
    ),
    null,
  );
  assert.equal(
    classifyStudyMove(
      attempted("d2d4", "d4"),
      parentReview(),
      comparison("excellent", "d2d4"),
    ),
    null,
  );
});

test("maps reviewed mistakes to bad and preserves blunders", () => {
  const parent = parentReview();
  assert.equal(
    classifyStudyMove(attempted("d2d4", "d4"), parent, comparison("inaccuracy", "d2d4"))?.kind,
    "bad",
  );
  assert.equal(
    classifyStudyMove(attempted("d2d4", "d4"), parent, comparison("mistake", "d2d4"))?.kind,
    "bad",
  );
  assert.equal(
    classifyStudyMove(attempted("d2d4", "d4"), parent, comparison("blunder", "d2d4"))?.kind,
    "blunder",
  );
});

test("rejects malformed attempted move coordinates", () => {
  assert.equal(
    classifyStudyMove(attempted("not-a-move", "bad"), parentReview(), null),
    null,
  );
});

test("keeps validated tactical annotations unchanged", () => {
  const annotation = {
    id: "root-tactic",
    label: "Pin",
    text: "A pin.",
    scope: "best_line",
    ply: 0,
    markers: [],
    arrows: [{ from_square: "c1", to_square: "g5", role: "ray" }],
    badge: {
      kind: "pin",
      square: "e3",
      role: "ray",
      arrow_index: 0,
    },
    evidence_ids: ["pin"],
  };

  assert.equal(studyBoardCue({ explanation: [annotation] }), annotation);
  assert.deepEqual(annotation.evidence_ids, ["pin"]);
});

test("provides the top move as a separate overlay when no root cue exists", () => {
  const review = {
    explanation: [{
      id: "later",
      label: "Later tactic",
      text: "A later idea.",
      scope: "best_line",
      ply: 2,
      markers: [],
      arrows: [{ from_square: "g1", to_square: "f3", role: "attack" }],
      badge: null,
      evidence_ids: ["later"],
    }],
  };

  assert.equal(studyBoardCue(review), null);
  assert.deepEqual(studyEngineArrow("e2e4", null), {
    from_square: "e2",
    to_square: "e4",
    role: "engine",
  });
});

test("does not duplicate a validated engine arrow", () => {
  const cue = {
    arrows: [{ from_square: "e2", to_square: "e4", role: "engine" }],
  };

  assert.equal(studyEngineArrow("e2e4", cue), null);
});

test("returns SAN through the shared replay helper and rejects invalid engine moves", () => {
  assert.equal(sanForEngineMove(new Chess().fen(), "e2e4"), "e4");
  assert.throws(
    () => sanForEngineMove(new Chess().fen(), "e2e5"),
    StudyAnalysisError,
  );
});
