import { parseUciMove } from "./engine/uci.ts";
import type {
  CoachingMoveSegment,
  PositionCoaching,
  ReviewAnnotation,
  ReviewArrowRole,
} from "./types";

const PIECE_SYMBOLS = {
  w: { K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙" },
  b: { K: "♚", Q: "♛", R: "♜", B: "♝", N: "♞", P: "♟" },
} as const;

export function hasCoachingNarrative(coaching: PositionCoaching): boolean {
  return coaching.status !== "disabled" && coaching.sections.length > 0;
}

export function coachingMoveLabel(move: CoachingMoveSegment): string {
  const context =
    move.role === "attempt"
      ? "Learner attempt"
      : move.scope === "best_line"
        ? "Engine line"
        : move.scope === "attempt_refutation"
          ? "Hypothetical reply line"
          : "Attempt line";
  return `${context}, line step ${move.ply + 1}: show ${move.move.san} on the board`;
}

export function coachingMoveDisplay(
  san: string,
  color: "w" | "b",
): { symbol: string; notation: string } {
  if (san.startsWith("O-O")) {
    return { symbol: PIECE_SYMBOLS[color].K, notation: san };
  }
  const piece = san[0];
  if (piece && piece in PIECE_SYMBOLS[color] && piece !== "P") {
    return {
      symbol: PIECE_SYMBOLS[color][piece as keyof typeof PIECE_SYMBOLS.w],
      notation: san.slice(1),
    };
  }
  return { symbol: PIECE_SYMBOLS[color].P, notation: san };
}

export function coachingMoveColor(
  rootTurn: "w" | "b",
  ply: number,
): "w" | "b" {
  return ply % 2 === 0 ? rootTurn : rootTurn === "w" ? "b" : "w";
}

export function coachingMoveCue(move: CoachingMoveSegment): ReviewAnnotation {
  const parsedMove = parseUciMove(move.move.uci);
  if (!parsedMove) throw new Error("Coaching contains an invalid UCI move.");
  const arrowRole: ReviewArrowRole =
    move.role === "attempt"
      ? "played"
      : move.role === "reply"
        ? "reply"
        : "engine";
  return {
    id: `coach-${move.scope.replaceAll("_", "-")}-${move.ply}`,
    label: move.move.san,
    text: `Show ${move.move.san} on the board.`,
    scope: move.scope,
    ply: move.ply,
    markers: [],
    arrows: [
      {
        from_square: parsedMove.from,
        to_square: parsedMove.to,
        role: arrowRole,
      },
    ],
    badge: null,
    evidence_ids: [],
  };
}
