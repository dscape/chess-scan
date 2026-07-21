import { Chess, validateFen, type Color, type PieceSymbol } from "chess.js";
import type { Orientation, SideToMove } from "./types";

export const pieceSymbols = ["·", "♙", "♘", "♗", "♖", "♕", "♔", "♟", "♞", "♝", "♜", "♛", "♚"] as const;
export const pieceNames = [
  "Empty",
  "White pawn",
  "White knight",
  "White bishop",
  "White rook",
  "White queen",
  "White king",
  "Black pawn",
  "Black knight",
  "Black bishop",
  "Black rook",
  "Black queen",
  "Black king",
] as const;
const fenSymbols = ["", "P", "N", "B", "R", "Q", "K", "p", "n", "b", "r", "q", "k"] as const;
const pieceOffsets: Record<PieceSymbol, number> = {
  p: 1,
  n: 2,
  b: 3,
  r: 4,
  q: 5,
  k: 6,
};

export function pieceDisplay(color: Color, type: PieceSymbol): { name: string; symbol: string } {
  const index = pieceOffsets[type] + (color === "w" ? 0 : 6);
  return {
    name: pieceNames[index]!,
    symbol: pieceSymbols[index]!,
  };
}

export function positionAt(fen: string, moves: string[]): Chess {
  const chess = new Chess(fen);
  for (const uci of moves) {
    try {
      chess.move(uci);
    } catch (cause) {
      throw new Error(`Illegal review move: ${uci}`, { cause });
    }
  }
  return chess;
}

export function labelsToBoardFen(labels: number[], orientation: Orientation): string {
  const canonical = orientation === "white" ? labels : [...labels].reverse();
  const ranks: string[] = [];
  for (let row = 0; row < 8; row += 1) {
    let rank = "";
    let empty = 0;
    for (let col = 0; col < 8; col += 1) {
      const id = canonical[row * 8 + col] ?? 0;
      if (id === 0) {
        empty += 1;
      } else {
        if (empty > 0) rank += String(empty);
        rank += fenSymbols[id] ?? "";
        empty = 0;
      }
    }
    if (empty > 0) rank += String(empty);
    ranks.push(rank || "8");
  }
  return ranks.join("/");
}

export function fullFen(
  labels: number[],
  orientation: Orientation,
  side: SideToMove,
  castling: string,
): string {
  return `${labelsToBoardFen(labels, orientation)} ${side} ${castling || "-"} - 0 1`;
}

export function fenError(fen: string): string | null {
  const result = validateFen(fen);
  return result.ok ? null : (result.error ?? "Invalid FEN");
}

export function squareName(index: number, orientation: Orientation): string {
  const row = Math.floor(index / 8);
  const col = index % 8;
  if (orientation === "white") {
    return `${String.fromCharCode(97 + col)}${8 - row}`;
  }
  return `${String.fromCharCode(104 - col)}${row + 1}`;
}

export function predictionNeedsReview(
  label: number,
  confidence: number,
  probabilities: number[],
): boolean {
  if (confidence < 0.72) return true;
  if (label === 5) return (probabilities[11] ?? 0) >= 0.1;
  if (label === 11) return (probabilities[5] ?? 0) >= 0.1;
  return false;
}

export function countKings(labels: number[]): { white: number; black: number } {
  return {
    white: labels.filter((label) => label === 6).length,
    black: labels.filter((label) => label === 12).length,
  };
}
