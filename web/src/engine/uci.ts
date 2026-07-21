import type { PieceSymbol, Square } from "chess.js";

export type EngineScore = {
  kind: "cp" | "mate";
  value: number;
  bound?: "lower" | "upper" | null;
};

export type EngineLine = {
  rank: number;
  depth: number;
  score: EngineScore;
  wdl?: [number, number, number];
  pv: string[];
  stable?: boolean;
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
  const pv = pvTokens.slice(0, 16);

  const rankIndex = tokens.indexOf("multipv");
  const parsedRank = integerAfter(tokens, "multipv");
  if (rankIndex >= 0 && (parsedRank === null || parsedRank < 1 || parsedRank > 3)) return null;
  const rank = parsedRank ?? 1;
  const wdlIndex = tokens.indexOf("wdl");
  const wdl = wdlIndex >= 0 ? parseWdl(tokens.slice(wdlIndex + 1, wdlIndex + 4)) : undefined;

  const boundToken = tokens[scoreIndex + 3];
  const bound = boundToken === "lowerbound"
    ? "lower"
    : boundToken === "upperbound"
      ? "upper"
      : undefined;

  return {
    rank,
    depth,
    score: { kind: scoreKind, value: scoreValue, ...(bound ? { bound } : {}) },
    ...(wdl ? { wdl } : {}),
    pv,
  };
}

export function isStableLine(
  expectedMove: string,
  line: EngineLine | undefined,
  recentMoves: string[],
): line is EngineLine {
  return line !== undefined
    && line.score.bound == null
    && line.pv[0] === expectedMove
    && recentMoves.length >= 2
    && recentMoves.every((move) => move === expectedMove);
}

export function contiguousRankedLines(lines: EngineLine[]): EngineLine[] {
  const contiguous: EngineLine[] = [];
  for (const line of [...lines].sort((left, right) => left.rank - right.rank)) {
    if (line.rank !== contiguous.length + 1) break;
    contiguous.push(line);
  }
  return contiguous;
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

function parseWdl(tokens: string[]): [number, number, number] | undefined {
  if (tokens.length !== 3) return undefined;
  const values = tokens.map(Number);
  if (!values.every((value) => Number.isInteger(value) && value >= 0)) return undefined;
  const total = values[0]! + values[1]! + values[2]!;
  if (total !== 1000) return undefined;
  return [values[0]!, values[1]!, values[2]!];
}
