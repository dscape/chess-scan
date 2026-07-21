import { useEffect, useMemo, useState } from "react";
import { Chess, SQUARES, type PieceSymbol, type Square } from "chess.js";
import { pieceDisplay } from "../board";
import { parseUciMove } from "../engine/uci";
import type { Orientation } from "../types";

export type AttemptedMove = { uci: string; san: string };

type InteractiveBoardProps = {
  fen: string;
  orientation: Orientation;
  moves?: string[];
  ply?: number;
  interactive?: boolean;
  highlights?: string[];
  onMove?: (move: AttemptedMove) => void;
};

type PromotionChoice = {
  from: Square;
  to: Square;
  pieces: PieceSymbol[];
};

const BLACK_ORIENTED_SQUARES = [...SQUARES].reverse();

export default function InteractiveBoard({
  fen,
  orientation,
  moves = [],
  ply = 0,
  interactive = false,
  highlights = [],
  onMove,
}: InteractiveBoardProps) {
  const [selected, setSelected] = useState<Square | null>(null);
  const [promotion, setPromotion] = useState<PromotionChoice | null>(null);
  const position = useMemo(() => positionAt(fen, moves, ply), [fen, moves, ply]);
  const legalTargets = useMemo(() => {
    if (!interactive || !selected) return new Set<Square>();
    return new Set(
      position.moves({ square: selected, verbose: true }).map((move) => move.to),
    );
  }, [interactive, position, selected]);
  const squares = orientation === "white" ? SQUARES : BLACK_ORIENTED_SQUARES;
  const lastMove = position.history({ verbose: true }).at(-1) ?? null;

  useEffect(() => {
    setSelected(null);
    setPromotion(null);
  }, [fen, ply]);

  function selectSquare(square: Square) {
    if (!interactive) return;
    const piece = position.get(square);
    if (!selected) {
      if (piece?.color === position.turn()) setSelected(square);
      return;
    }
    if (piece?.color === position.turn()) {
      setSelected(square);
      return;
    }
    if (!legalTargets.has(square)) {
      setSelected(null);
      return;
    }
    const candidates = position
      .moves({ square: selected, verbose: true })
      .filter((move) => move.to === square);
    const promotions = candidates
      .map((move) => move.promotion)
      .filter((piece): piece is PieceSymbol => piece !== undefined);
    if (promotions.length > 1) {
      setPromotion({ from: selected, to: square, pieces: promotions });
      return;
    }
    play(selected, square, promotions[0]);
  }

  function play(from: Square, to: Square, promoteTo?: PieceSymbol) {
    const next = new Chess(position.fen());
    const move = next.move({ from, to, ...(promoteTo ? { promotion: promoteTo } : {}) });
    setSelected(null);
    setPromotion(null);
    onMove?.({ uci: move.lan, san: move.san });
  }

  return (
    <div className="lesson-board-wrap">
      <div className="lesson-board" role="grid" aria-label="Position lesson chessboard">
        {squares.map((square, index) => {
          const piece = position.get(square);
          const display = piece ? pieceDisplay(piece.color, piece.type) : null;
          const row = Math.floor(index / 8);
          const col = index % 8;
          const isLastMove = square === lastMove?.from || square === lastMove?.to;
          const keyboardTarget = interactive && (
            piece?.color === position.turn() || legalTargets.has(square)
          );
          return (
            <button
              key={square}
              type="button"
              role="gridcell"
              className={[
                "lesson-square",
                position.squareColor(square) === "light" ? "is-light" : "is-dark",
                selected === square ? "is-selected" : "",
                legalTargets.has(square) ? "is-legal" : "",
                highlights.includes(square) ? "is-highlighted" : "",
                isLastMove ? "is-last-move" : "",
              ].join(" ")}
              aria-label={`${square}${display ? `: ${display.name.toLowerCase()}` : ": empty"}`}
              aria-disabled={!interactive}
              tabIndex={keyboardTarget ? 0 : -1}
              onClick={() => selectSquare(square)}
            >
              {coordinateFor(square, row, col)}
              {display && <span className="chess-symbol lesson-square__piece">{display.symbol}</span>}
              {legalTargets.has(square) && <span className="lesson-square__target" aria-hidden="true" />}
            </button>
          );
        })}
      </div>
      {promotion && (
        <div className="promotion-picker" role="dialog" aria-label="Choose promotion piece">
          <strong>Promote to</strong>
          <div>
            {promotion.pieces.map((piece) => {
              const display = pieceDisplay(position.turn(), piece);
              return (
                <button
                  key={piece}
                  type="button"
                  aria-label={`Promote to ${display.name.toLowerCase()}`}
                  onClick={() => play(promotion.from, promotion.to, piece)}
                >
                  <span className="chess-symbol">{display.symbol}</span>
                </button>
              );
            })}
          </div>
          <button type="button" className="text-button" onClick={() => setPromotion(null)}>Cancel</button>
        </div>
      )}
    </div>
  );
}

function positionAt(fen: string, moves: string[], ply: number): Chess {
  const chess = new Chess(fen);
  for (const uci of moves.slice(0, ply)) {
    if (!parseUciMove(uci)) throw new Error(`Invalid variation move: ${uci}`);
    try {
      chess.move(uci);
    } catch (cause) {
      throw new Error(`Illegal variation move: ${uci}`, { cause });
    }
  }
  return chess;
}

function coordinateFor(square: Square, row: number, col: number) {
  const fileEdge = col === 0;
  const rankEdge = row === 7;
  return (
    <>
      {fileEdge && <span className="lesson-square__rank">{square[1]}</span>}
      {rankEdge && <span className="lesson-square__file">{square[0]}</span>}
    </>
  );
}
