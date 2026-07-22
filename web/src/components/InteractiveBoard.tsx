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
              {display && <span className="chess-symbol analysis-square__piece">{display.symbol}</span>}
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
            markerWidth="4"
            markerHeight="4"
            refX="3.4"
            refY="2"
            orient="auto"
            markerUnits="strokeWidth"
          >
            <path className={`board-annotation__head is-${role}`} d="M0,0 L4,2 L0,4 Z" />
          </marker>
        ))}
      </defs>
      <g className="board-annotation__scene">
        {markers.map(({ marker, point }, index) => (
          <g
            key={`${marker.square}-${marker.role}-${index}`}
            className={`board-annotation__marker is-${marker.role}`}
          >
            <rect x={point.x - 0.39} y={point.y - 0.39} width="0.78" height="0.78" rx="0.1" />
          </g>
        ))}
        {arrows.map(({ arrow, line }, index) => (
          <line
            key={`${arrow.from_square}-${arrow.to_square}-${arrow.role}-${index}`}
            className={`board-annotation__arrow is-${arrow.role}`}
            x1={line.start.x}
            y1={line.start.y}
            x2={line.end.x}
            y2={line.end.y}
            markerEnd={`url(#${markerId}-${arrow.role})`}
          />
        ))}
        {positionedBadge && (
          <g
            className={`board-annotation__badge is-${positionedBadge.badge.role}`}
            transform={`translate(${positionedBadge.point.x} ${positionedBadge.point.y})`}
          >
            <circle r="0.2" />
            <g
              transform="translate(-0.1 -0.1) scale(0.008333)"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.8"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <ReviewGlyphPaths badge={positionedBadge.badge.kind} />
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
