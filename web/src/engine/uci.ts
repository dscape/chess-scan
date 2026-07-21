import type { PieceSymbol, Square } from "chess.js";

export type EngineScore = {
  kind: "cp" | "mate";
  value: number;
  bound?: "lower" | "upper" | null;
};

export type EngineLine = {
  depth: number;
  score: EngineScore;
  pv: string[];
};

export type ParsedUciMove = {
  from: Square;
  to: Square;
  promotion?: PieceSymbol;
};

const MOVE_PATTERN = /^([a-h][1-8])([a-h][1-8])([qrbn])?$/;

export function parseInfoLine(message: string): EngineLine | null {
  if (!message.startsWith("info ") || !message.includes(" score ") || !message.includes(" pv ")) {
    return null;
  }
  const tokens = message.trim().split(/\s+/);
  const depth = integerAfter(tokens, "depth");
  const scoreIndex = tokens.indexOf("score");
  const pvIndex = tokens.indexOf("pv");
  if (depth === null || depth < 1 || scoreIndex < 0 || pvIndex < 0 || pvIndex <= scoreIndex + 2) {
    return null;
  }

  const scoreKind = tokens[scoreIndex + 1];
  const scoreValue = Number(tokens[scoreIndex + 2]);
  if ((scoreKind !== "cp" && scoreKind !== "mate") || !Number.isInteger(scoreValue)) return null;

  const pvTokens = tokens.slice(pvIndex + 1);
  if (pvTokens.length === 0 || pvTokens.some((move) => !isUciMove(move))) return null;
  const pv = pvTokens.slice(0, 24);

  const boundToken = tokens[scoreIndex + 3];
  const bound = boundToken === "lowerbound"
    ? "lower"
    : boundToken === "upperbound"
      ? "upper"
      : undefined;

  return {
    depth,
    score: { kind: scoreKind, value: scoreValue, ...(bound ? { bound } : {}) },
    pv,
  };
}

export function isMatchingPrimary(
  bestMove: string,
  primary: EngineLine | null | undefined,
): primary is EngineLine {
  return primary != null
    && primary.score.bound == null
    && primary.pv[0] === bestMove;
}

export function parseBestMove(message: string): string | null {
  const tokens = message.trim().split(/\s+/);
  if (tokens[0] !== "bestmove" || !tokens[1]) return null;
  if (tokens[1] === "(none)") return tokens[1];
  return isUciMove(tokens[1]) ? tokens[1] : null;
}

export function parseUciMove(move: string): ParsedUciMove | null {
  const match = MOVE_PATTERN.exec(move);
  if (!match?.[1] || !match[2] || match[1] === match[2]) return null;
  return {
    from: match[1] as Square,
    to: match[2] as Square,
    ...(match[3] ? { promotion: match[3] as PieceSymbol } : {}),
  };
}

export function isUciMove(move: string): boolean {
  return parseUciMove(move) !== null;
}

function integerAfter(tokens: string[], key: string): number | null {
  const index = tokens.indexOf(key);
  if (index < 0) return null;
  const value = Number(tokens[index + 1]);
  return Number.isInteger(value) && value >= 0 ? value : null;
}
