import { useEffect, useId, useMemo, useState } from "react";
import { Chess, SQUARES, type PieceSymbol, type Square } from "chess.js";
import {
  boardPoint as pointForSquare,
  pieceDisplay,
  positionAt,
  type BoardPoint,
} from "../board";
import type {
  Orientation,
  ReviewAnnotation,
  ReviewArrow,
} from "../types";
import ChessPiece from "./ChessPiece";
import { ReviewGlyphPaths } from "./ReviewGlyph";

export type AttemptedMove = { uci: string; san: string };

type InteractiveBoardProps = {
  fen: string;
  orientation: Orientation;
  moves?: string[];
  interactive?: boolean;
  cue?: ReviewAnnotation | null;
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
  interactive = false,
  cue = null,
  onMove,
}: InteractiveBoardProps) {
  const [selected, setSelected] = useState<Square | null>(null);
  const [promotion, setPromotion] = useState<PromotionChoice | null>(null);
  const markerId = useId().replaceAll(":", "");
  const position = useMemo(() => positionAt(fen, moves), [fen, moves]);
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
  }, [fen, moves]);

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
                isLastMove ? "is-last-move" : "",
              ].join(" ")}
              aria-label={`${square}${display ? `: ${display.name.toLowerCase()}` : ": empty"}`}
              aria-disabled={!keyboardTarget}
              tabIndex={keyboardTarget ? 0 : -1}
              onClick={() => selectSquare(square)}
            >
              {coordinateFor(square, row, col)}
              {piece && (
                <ChessPiece
                  className="analysis-square__piece"
                  color={piece.color}
                  piece={piece.type}
                />
              )}
              {legalTargets.has(square) && <span className="analysis-square__target" aria-hidden="true" />}
            </button>
          );
        })}
        {cue && (
          <BoardOverlay
            key={`${cue.scope}-${cue.ply}-${cue.label}`}
            cue={cue}
            orientation={orientation}
            markerId={markerId}
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
                  <ChessPiece color={position.turn()} piece={piece} />
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
  markerId,
}: {
  cue: ReviewAnnotation;
  orientation: Orientation;
  markerId: string;
}) {
  const arrows = cue.arrows.flatMap((arrow) => {
    const line = arrowLine(arrow, orientation);
    return line ? [{ arrow, line }] : [];
  });
  const arrowRoles = [...new Set(arrows.map(({ arrow }) => arrow.role))];
  const badgeSquare = cue.badge
    ? pointForSquare(cue.badge.square, orientation)
    : null;
  const positionedBadge = cue.badge && badgeSquare
    ? { badge: cue.badge, point: badgePoint(badgeSquare) }
    : null;
  const markers = cue.markers.flatMap((marker) => {
    const point = pointForSquare(marker.square, orientation);
    return point ? [{ marker, point }] : [];
  });

  return (
    <svg className="board-annotation" viewBox="0 0 8 8" aria-hidden="true">
      <defs>
        {arrowRoles.map((role) => (
          <marker
            key={role}
            id={`${markerId}-${role}`}
            markerWidth="5"
            markerHeight="5"
            refX="4.25"
            refY="2.5"
            orient="auto"
            markerUnits="strokeWidth"
            viewBox="0 0 5 5"
          >
            <path
              className={`board-annotation__head is-${role}`}
              d="M0.7 0.55C0.42 0.39 0.1 0.62 0.2 0.93L0.92 2.5 0.2 4.07C0.1 4.38 0.42 4.61 0.7 4.45L4.25 2.86C4.57 2.72 4.57 2.28 4.25 2.14Z"
            />
          </marker>
        ))}
      </defs>
      <g className="board-annotation__scene">
        {markers.map(({ marker, point }, index) => (
          <g
            key={`${marker.square}-${marker.role}-${index}`}
            className={`board-annotation__marker is-${marker.role}`}
          >
            <rect
              className="board-annotation__marker-keyline"
              x={point.x - 0.405}
              y={point.y - 0.405}
              width="0.81"
              height="0.81"
              rx="0.13"
            />
            <rect
              className="board-annotation__marker-face"
              x={point.x - 0.355}
              y={point.y - 0.355}
              width="0.71"
              height="0.71"
              rx="0.09"
            />
          </g>
        ))}
        {arrows.map(({ arrow, line }, index) => (
          <g key={`${arrow.from_square}-${arrow.to_square}-${arrow.role}-${index}`}>
            <line
              className={`board-annotation__arrow-keyline is-${arrow.role}`}
              x1={line.start.x}
              y1={line.start.y}
              x2={line.end.x}
              y2={line.end.y}
            />
            <line
              className={`board-annotation__arrow is-${arrow.role}`}
              x1={line.start.x}
              y1={line.start.y}
              x2={line.end.x}
              y2={line.end.y}
              markerEnd={`url(#${markerId}-${arrow.role})`}
            />
          </g>
        ))}
        {positionedBadge && (
          <g
            className={`board-annotation__badge is-${positionedBadge.badge.role}`}
            transform={`translate(${positionedBadge.point.x} ${positionedBadge.point.y})`}
          >
            <rect className="board-annotation__badge-keyline" x="-0.27" y="-0.27" width="0.54" height="0.54" rx="0.13" />
            <rect className="board-annotation__badge-face" x="-0.205" y="-0.205" width="0.41" height="0.41" rx="0.085" />
            <g transform="translate(-0.105 -0.105) scale(0.00875)" fill="none">
              <g
                className="board-annotation__glyph-keyline"
                strokeWidth="4.8"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <ReviewGlyphPaths badge={positionedBadge.badge.kind} />
              </g>
              <g
                className="board-annotation__glyph-face"
                strokeWidth="2.2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <ReviewGlyphPaths badge={positionedBadge.badge.kind} />
              </g>
            </g>
          </g>
        )}
      </g>
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

function badgePoint(point: BoardPoint): BoardPoint {
  return {
    x: Math.min(7.78, point.x + 0.27),
    y: Math.max(0.22, point.y - 0.27),
  };
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
