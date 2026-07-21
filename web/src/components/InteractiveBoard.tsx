import { useEffect, useId, useMemo, useState } from "react";
import { Chess, SQUARES, type PieceSymbol, type Square } from "chess.js";
import { pieceDisplay } from "../board";
import { parseUciMove } from "../engine/uci";
import type { Orientation, ReviewAnnotation, ReviewArrow } from "../types";

export type AttemptedMove = { uci: string; san: string };

type InteractiveBoardProps = {
  fen: string;
  orientation: Orientation;
  moves?: string[];
  ply?: number;
  interactive?: boolean;
  cue?: ReviewAnnotation | null;
  onMove?: (move: AttemptedMove) => void;
};

type PromotionChoice = {
  from: Square;
  to: Square;
  pieces: PieceSymbol[];
};

type BoardPoint = { x: number; y: number };

const BLACK_ORIENTED_SQUARES = [...SQUARES].reverse();

export default function InteractiveBoard({
  fen,
  orientation,
  moves = [],
  ply = 0,
  interactive = false,
  cue = null,
  onMove,
}: InteractiveBoardProps) {
  const [selected, setSelected] = useState<Square | null>(null);
  const [promotion, setPromotion] = useState<PromotionChoice | null>(null);
  const markerId = useId().replaceAll(":", "");
  const position = useMemo(() => positionAt(fen, moves, ply), [fen, moves, ply]);
  const highlightedSquares = useMemo(() => new Set(cue?.squares ?? []), [cue]);
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
  }, [fen, moves, ply]);

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
    <div className="analysis-board-wrap">
      <div className="analysis-board" role="grid" aria-label="Interactive chess position">
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
                "analysis-square",
                position.squareColor(square) === "light" ? "is-light" : "is-dark",
                selected === square ? "is-selected" : "",
                legalTargets.has(square) ? "is-legal" : "",
                highlightedSquares.has(square) ? "is-highlighted" : "",
                isLastMove ? "is-last-move" : "",
              ].join(" ")}
              aria-label={`${square}${display ? `: ${display.name.toLowerCase()}` : ": empty"}`}
              aria-disabled={!keyboardTarget}
              tabIndex={keyboardTarget ? 0 : -1}
              onClick={() => selectSquare(square)}
            >
              {coordinateFor(square, row, col)}
              {display && <span className="chess-symbol analysis-square__piece">{display.symbol}</span>}
              {legalTargets.has(square) && <span className="analysis-square__target" aria-hidden="true" />}
            </button>
          );
        })}
        {cue && (
          <BoardOverlay
            cue={cue}
            orientation={orientation}
            moveMarkerId={`${markerId}-move`}
            ideaMarkerId={`${markerId}-idea`}
          />
        )}
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

function BoardOverlay({
  cue,
  orientation,
  moveMarkerId,
  ideaMarkerId,
}: {
  cue: ReviewAnnotation;
  orientation: Orientation;
  moveMarkerId: string;
  ideaMarkerId: string;
}) {
  const arrows = cue.arrows.flatMap((arrow) => {
    const line = arrowLine(arrow, orientation);
    return line ? [{ arrow, line }] : [];
  });
  const anchorSquare = cue.arrows[0]?.to_square ?? cue.squares[0];
  const anchor = anchorSquare ? pointForSquare(anchorSquare, orientation) : null;
  const tagWidth = Math.min(1.9, Math.max(0.85, cue.label.length * 0.095 + 0.35));
  const tagX = anchor ? clamp(anchor.x - tagWidth / 2, 0.08, 7.92 - tagWidth) : 0;
  const tagY = anchor ? clamp(anchor.y < 1.1 ? anchor.y + 0.34 : anchor.y - 0.62, 0.08, 7.5) : 0;

  return (
    <svg className="board-annotation" viewBox="0 0 8 8" aria-hidden="true">
      <defs>
        <marker id={moveMarkerId} markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto" markerUnits="strokeWidth">
          <path className="board-annotation__move-head" d="M0,0 L5,2.5 L0,5 Z" />
        </marker>
        <marker id={ideaMarkerId} markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto" markerUnits="strokeWidth">
          <path className="board-annotation__idea-head" d="M0,0 L5,2.5 L0,5 Z" />
        </marker>
      </defs>
      {arrows.map(({ arrow, line }, index) => (
        <line
          key={`${arrow.from_square}-${arrow.to_square}-${index}`}
          className={`board-annotation__arrow is-${arrow.kind}`}
          x1={line.start.x}
          y1={line.start.y}
          x2={line.end.x}
          y2={line.end.y}
          markerEnd={`url(#${arrow.kind === "move" ? moveMarkerId : ideaMarkerId})`}
        />
      ))}
      {anchor && (
        <g className="board-annotation__tag" transform={`translate(${tagX} ${tagY})`}>
          <rect width={tagWidth} height="0.4" rx="0.1" />
          <text x={tagWidth / 2} y="0.27" textAnchor="middle">{cue.label}</text>
        </g>
      )}
    </svg>
  );
}

function arrowLine(arrow: ReviewArrow, orientation: Orientation) {
  const from = pointForSquare(arrow.from_square, orientation);
  const to = pointForSquare(arrow.to_square, orientation);
  if (!from || !to) return null;
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const distance = Math.hypot(dx, dy);
  if (distance === 0) return null;
  const startInset = 0.16;
  const endInset = 0.3;
  return {
    start: {
      x: from.x + (dx / distance) * startInset,
      y: from.y + (dy / distance) * startInset,
    },
    end: {
      x: to.x - (dx / distance) * endInset,
      y: to.y - (dy / distance) * endInset,
    },
  };
}

function pointForSquare(square: string, orientation: Orientation): BoardPoint | null {
  if (!/^[a-h][1-8]$/.test(square)) return null;
  const file = square.charCodeAt(0) - 97;
  const rank = Number(square[1]) - 1;
  return orientation === "white"
    ? { x: file + 0.5, y: 7 - rank + 0.5 }
    : { x: 7 - file + 0.5, y: rank + 0.5 };
}

function positionAt(fen: string, moves: string[], ply: number): Chess {
  const chess = new Chess(fen);
  for (const uci of moves.slice(0, ply)) {
    if (!parseUciMove(uci)) throw new Error(`Invalid review move: ${uci}`);
    try {
      chess.move(uci);
    } catch (cause) {
      throw new Error(`Illegal review move: ${uci}`, { cause });
    }
  }
  return chess;
}

function coordinateFor(square: Square, row: number, col: number) {
  const fileEdge = col === 0;
  const rankEdge = row === 7;
  return (
    <>
      {fileEdge && <span className="analysis-square__rank">{square[1]}</span>}
      {rankEdge && <span className="analysis-square__file">{square[0]}</span>}
    </>
  );
}

function clamp(value: number, minimum: number, maximum: number): number {
  return Math.min(maximum, Math.max(minimum, value));
}
