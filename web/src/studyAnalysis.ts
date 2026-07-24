import type { Chess } from "chess.js";
import type { ApiFailure } from "./api";
import { positionAt } from "./board.ts";
import type { EngineLine, ParsedUciMove } from "./engine/uci";
import { oppositeScorePov, parseUciMove } from "./engine/uci.ts";
import { hasBoardCue } from "./reviewCue.ts";
import type {
  PositionReview,
  ReviewAnnotation,
  ReviewArrow,
  ReviewAttempt,
} from "./types";

export type StudyMoveGradeKind = "brilliant" | "good" | "bad" | "blunder";

export type StudyMoveGrade = {
  kind: StudyMoveGradeKind;
  label: "Brilliant" | "Only move" | "Bad move" | "Blunder";
  detail: string;
  square: string;
};

export type StudyFailure = ApiFailure;

export type StudyPosition = {
  lines: EngineLine[];
  topMoveSan: string | null;
  review: PositionReview | null;
  reviewError: StudyFailure | null;
  terminal: boolean;
};

export type StudyGradeResult = {
  grade: StudyMoveGrade | null;
  error: StudyFailure | null;
};

export type StudyNode = {
  lines: EngineLine[];
  topMoveSan: string | null;
  review: PositionReview | null;
  terminal: boolean;
  grade: StudyMoveGrade | null;
  gradeError: StudyFailure | null;
};

export type StudyState =
  | { status: "idle" }
  | {
      status: "loading";
      key: string;
      topLine: EngineLine | null;
      topMoveSan: string | null;
    }
  | { status: "ready"; key: string; node: StudyNode }
  | {
      status: "review-error";
      key: string;
      node: StudyNode;
      failure: StudyFailure;
    }
  | { status: "error"; key: string; message: string };

export type StudyAttemptedMove = {
  uci: string;
  san: string;
};

export class StudyAnalysisError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = "StudyAnalysisError";
  }
}

const MOVE_GRADE_SYMBOLS: Record<StudyMoveGradeKind, string> = {
  brilliant: "!!",
  good: "!",
  bad: "?",
  blunder: "??",
};
const ONLY_GOOD_MOVE_EXPECTED_SCORE_GAP = 0.05;

export function studyMoveGradeSymbol(kind: StudyMoveGradeKind): string {
  return MOVE_GRADE_SYMBOLS[kind];
}

export function studyPositionKey(fen: string): string {
  return positionAt(fen, []).fen();
}

export function studyPathKey(fen: string, moves: string[]): string {
  return JSON.stringify([fen, moves]);
}

export function studyStateForKey(state: StudyState, key: string): StudyState {
  return state.status === "idle" || state.key === key
    ? state
    : { status: "idle" };
}

export function prepareStudyRetry(
  position: StudyPosition | undefined,
  gradeResult: StudyGradeResult | undefined,
): { position: StudyPosition | null; gradeResult: StudyGradeResult | null } {
  return {
    position: position?.reviewError?.retryable
      ? { ...position, reviewError: null }
      : (position ?? null),
    gradeResult: gradeResult?.error?.retryable ? null : (gradeResult ?? null),
  };
}

export function rootEvaluationScore(
  lines: EngineLine[] | undefined,
  cached: EngineLine["score"] | null,
  reviewed: EngineLine["score"] | null,
): EngineLine["score"] | null {
  return lines?.[0]?.score ?? cached ?? reviewed;
}

export function terminalStudyPosition(): StudyPosition {
  return {
    lines: [],
    topMoveSan: null,
    review: null,
    reviewError: null,
    terminal: true,
  };
}

export function settledStudyState(
  key: string,
  position: StudyPosition,
  gradeResult: StudyGradeResult,
): Extract<StudyState, { status: "ready" | "review-error" }> {
  const node: StudyNode = {
    lines: position.lines,
    topMoveSan: position.topMoveSan,
    review: position.review,
    terminal: position.terminal,
    grade: gradeResult.grade,
    gradeError: gradeResult.error,
  };
  if (!position.terminal && position.review === null) {
    return {
      status: "review-error",
      key,
      node,
      failure: position.reviewError ?? {
        kind: "api",
        message: "Tactical annotations are unavailable.",
        status: null,
        retryable: true,
      },
    };
  }
  return { status: "ready", key, node };
}

export function terminalAttemptLine(
  attemptedMove: string,
  terminal: Chess,
  depth: number,
): EngineLine {
  const childLine: EngineLine = {
    rank: 1,
    depth: Math.max(8, depth),
    score: terminal.isCheckmate()
      ? { kind: "mate", value: -1 }
      : { kind: "cp", value: 0 },
    wdl: terminal.isCheckmate() ? [0, 0, 1000] : [0, 1000, 0],
    pv: [],
    stable: true,
  };
  const parentLine = attemptLineFromChild(attemptedMove, childLine);
  return terminal.isCheckmate()
    ? { ...parentLine, score: { kind: "mate", value: 1 } }
    : parentLine;
}

export function attemptLineFromChild(
  attemptedMove: string,
  childLine: EngineLine,
): EngineLine {
  const parentScore = oppositeScorePov(childLine.score);
  const score = parentScore.kind === "mate" && parentScore.value > 0
    ? { ...parentScore, value: parentScore.value + 1 }
    : parentScore;
  return {
    ...childLine,
    rank: 1,
    score,
    ...(childLine.wdl
      ? { wdl: [childLine.wdl[2], childLine.wdl[1], childLine.wdl[0]] }
      : {}),
    pv: [attemptedMove, ...childLine.pv].slice(0, 16),
  };
}

export function canUseCachedBestGrade(
  terminal: "white" | "black" | "draw" | null,
): boolean {
  return terminal !== "draw";
}

export function isStudyBestMove(
  attemptedMove: string,
  review: PositionReview,
): boolean {
  return (review.best_move?.uci ?? review.lines[0]?.moves[0]?.uci ?? null)
    === attemptedMove;
}

export function classifyStudyMove(
  attemptedMove: StudyAttemptedMove,
  parentReview: PositionReview | null,
  comparison: ReviewAttempt | null,
): StudyMoveGrade | null {
  const move = parseUciMove(attemptedMove.uci);
  if (!move || !parentReview) return null;
  const attempt = comparison ?? (
    isStudyBestMove(attemptedMove.uci, parentReview)
      ? {
          move: attemptedMove,
          verdict: "best" as const,
          headline: `${attemptedMove.san} is the best move.`,
        }
      : null
  );
  if (!attempt || attempt.move.uci !== attemptedMove.uci) return null;

  if (attempt.verdict === "blunder") {
    return {
      kind: "blunder",
      label: "Blunder",
      detail: attempt.headline,
      square: move.to,
    };
  }
  if (attempt.verdict === "inaccuracy" || attempt.verdict === "mistake") {
    return {
      kind: "bad",
      label: "Bad move",
      detail: attempt.headline,
      square: move.to,
    };
  }
  if (attempt.verdict !== "best") return null;

  if (isCausalTemporarySacrifice(parentReview, move)) {
    return {
      kind: "brilliant",
      label: "Brilliant",
      detail: `${attemptedMove.san} is Stockfish's first choice and starts a sound temporary sacrifice.`,
      square: move.to,
    };
  }
  if (isOnlyGoodMove(parentReview, attemptedMove.uci)) {
    return {
      kind: "good",
      label: "Only move",
      detail: `${attemptedMove.san} is the only checked move that preserves the evaluation.`,
      square: move.to,
    };
  }
  return null;
}

export function studyBoardCue(review: PositionReview | null): ReviewAnnotation | null {
  const rootCues = review?.explanation.filter(
    (cue) => cue.scope === "best_line" && cue.ply === 0,
  ) ?? [];
  return rootCues.find(hasTacticalDiagram) ?? rootCues.find(hasBoardCue) ?? null;
}

export function studyEngineArrow(
  topMove: string | null,
  cue: ReviewAnnotation | null,
): ReviewArrow | null {
  if (!topMove) return null;
  const move = parseUciMove(topMove);
  if (!move) return null;
  const arrow: ReviewArrow = {
    from_square: move.from,
    to_square: move.to,
    role: "engine",
  };
  return cue?.arrows.some((candidate) => arrowsMatch(candidate, arrow))
    ? null
    : arrow;
}

export function sanForEngineMove(fen: string, uci: string | null): string | null {
  if (!uci) return null;
  try {
    const position = positionAt(fen, [uci]);
    const move = position.history({ verbose: true }).at(-1);
    if (!move) throw new Error("The engine line has no first move.");
    return move.san;
  } catch (cause) {
    throw new StudyAnalysisError(
      `Stockfish returned an invalid study move: ${uci}.`,
      { cause },
    );
  }
}

function isOnlyGoodMove(review: PositionReview, attemptedMove: string): boolean {
  const best = review.lines[0];
  const alternative = review.lines[1];
  if (!best || !alternative || best.moves[0]?.uci !== attemptedMove) return false;
  return expectedScore(best.wdl) - expectedScore(alternative.wdl)
    > ONLY_GOOD_MOVE_EXPECTED_SCORE_GAP + Number.EPSILON;
}

function isCausalTemporarySacrifice(
  review: PositionReview,
  move: ParsedUciMove,
): boolean {
  const evidenceById = new Map(review.evidence.map((evidence) => [evidence.id, evidence]));
  return review.findings.some((finding) =>
    finding.topic.id === "temporary-sacrifice"
    && finding.evidence_ids.some((evidenceId) => {
      const evidence = evidenceById.get(evidenceId);
      return evidence?.kind === "temporary_sacrifice"
        && evidence.from_square === move.from
        && evidence.to_square === move.to;
    })
  );
}

function expectedScore(wdl: [number, number, number]): number {
  return (wdl[0] + wdl[1] / 2) / 1000;
}

function hasTacticalDiagram(cue: ReviewAnnotation): boolean {
  return cue.badge?.kind !== undefined && cue.badge.kind !== "engine"
    || cue.arrows.some((arrow) =>
      arrow.role === "attack" || arrow.role === "ray" || arrow.role === "threat"
    );
}

function arrowsMatch(left: ReviewArrow, right: ReviewArrow): boolean {
  return left.from_square === right.from_square
    && left.to_square === right.to_square
    && left.role === right.role;
}
