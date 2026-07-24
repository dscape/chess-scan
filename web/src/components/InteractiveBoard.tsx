import { useEffect, useId, useMemo, useState, type CSSProperties } from "react";
import { Chess, SQUARES, type PieceSymbol, type Square } from "chess.js";
import {
  boardPoint as pointForSquare,
  pieceName,
  positionAt,
} from "../board";
import {
  positionReviewArrows,
  positionReviewBadge,
  REVIEW_DIAGRAM_GEOMETRY,
  visibleReviewMarkers,
} from "../reviewDiagram";
import {
  studyEngineArrow,
  studyMoveGradeSymbol,
  type StudyMoveGrade,
} from "../studyAnalysis";
import type { Orientation, ReviewAnnotation, ReviewArrow } from "../types";
import ChessPiece from "./ChessPiece";
import { ReviewGlyphLayers } from "./ReviewGlyph";

export type AttemptedMove = {
  uci: string;
  san: string;
  parentFen: string;
};

type InteractiveBoardProps = {
  fen: string;
  orientation: Orientation;
  moves?: string[];
  interactive?: boolean;
  cue?: ReviewAnnotation | null;
  engineMove?: string | null;
  moveGrade?: StudyMoveGrade | null;
  onMove?: (move: AttemptedMove) => void;
};

type PromotionChoice = {
  from: Square;
  to: Square;
  pieces: PieceSymbol[];
};

const BLACK_ORIENTED_SQUARES = [...SQUARES].reverse();
const REVIEW_DIAGRAM_STYLE = {
  "--review-arrow-width": REVIEW_DIAGRAM_GEOMETRY.arrowWidth,
  "--review-arrow-keyline-width": REVIEW_DIAGRAM_GEOMETRY.arrowKeylineWidth,
  "--review-arrow-head-keyline-width": REVIEW_DIAGRAM_GEOMETRY.arrowHeadKeylineWidth,
  "--review-badge-keyline-width": REVIEW_DIAGRAM_GEOMETRY.badgeKeylineWidth,
} as CSSProperties;

export default function InteractiveBoard({
  fen,
  orientation,
  moves = [],
  interactive = false,
  cue = null,
  engineMove = null,
  moveGrade = null,
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
  const engineArrow = studyEngineArrow(engineMove, cue);

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
    onMove?.({
      uci: move.lan,
      san: move.san,
      parentFen: position.fen(),
    });
  }

  return (
    <div className="analysis-board-wrap">
      <div className="analysis-board" role="grid" aria-label="Interactive chess position">
        {squares.map((square, index) => {
          const piece = position.get(square);
          const name = piece ? pieceName(piece.color, piece.type) : null;
          const row = Math.floor(index / 8);
          const col = index % 8;
          const isLastMove = square === lastMove?.from || square === lastMove?.to;
          const isGradedMove = square === lastMove?.to && square === moveGrade?.square;
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
                isGradedMove ? `is-graded-${moveGrade.kind}` : "",
              ].join(" ")}
              aria-label={`${square}${name ? `: ${name.toLowerCase()}` : ": empty"}`}
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
        {(cue || engineArrow) && (
          <BoardOverlay
            key={cue ? `${cue.id}-${cue.scope}-${cue.ply}` : engineMove}
            cue={cue}
            engineArrow={engineArrow}
            orientation={orientation}
            markerId={markerId}
          />
        )}
        {moveGrade && (
          <MoveGradeBadge grade={moveGrade} orientation={orientation} />
        )}
      </div>
      {promotion && (
        <div className="promotion-picker" role="dialog" aria-label="Choose promotion piece">
          <strong>Promote to</strong>
          <div>
            {promotion.pieces.map((piece) => {
              const name = pieceName(position.turn(), piece);
              return (
                <button
                  key={piece}
                  type="button"
                  aria-label={`Promote to ${name.toLowerCase()}`}
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

function MoveGradeBadge({
  grade,
  orientation,
}: {
  grade: StudyMoveGrade;
  orientation: Orientation;
}) {
  const point = pointForSquare(grade.square, orientation);
  if (!point) return null;
  const x = point.x + (point.x > 7 ? -0.29 : 0.29);
  const y = point.y + (point.y < 1 ? 0.29 : -0.29);
  const symbol = studyMoveGradeSymbol(grade.kind);
  return (
    <span
      className={`move-grade-badge is-${grade.kind}`}
      style={{ left: `${(x / 8) * 100}%`, top: `${(y / 8) * 100}%` }}
      role="img"
      aria-label={`${grade.label}. ${grade.detail}`}
    >
      {symbol}
    </span>
  );
}

function BoardOverlay({
  cue,
  engineArrow,
  orientation,
  markerId,
}: {
  cue: ReviewAnnotation | null;
  engineArrow: ReviewArrow | null;
  orientation: Orientation;
  markerId: string;
}) {
  const resolvePoint = (square: string) => pointForSquare(square, orientation);
  const cueArrows = positionReviewArrows(cue?.arrows ?? [], resolvePoint);
  const engineArrows = positionReviewArrows(
    engineArrow ? [engineArrow] : [],
    resolvePoint,
  );
  const arrows = [...engineArrows, ...cueArrows];
  const arrowRoles = [...new Set(arrows.map(({ arrow }) => arrow.role))];
  const positionedBadge = positionReviewBadge(
    cue?.badge ?? null,
    cueArrows,
    resolvePoint,
  );
  const markers = visibleReviewMarkers(
    cue?.markers ?? [],
    cueArrows,
  ).flatMap((marker) => {
    const point = resolvePoint(marker.square);
    return point ? [{ marker, point }] : [];
  });

  return (
    <svg
      className="board-annotation"
      viewBox="0 0 8 8"
      style={REVIEW_DIAGRAM_STYLE}
      aria-hidden="true"
    >
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
            transform={`translate(${point.x} ${point.y})`}
          >
            <circle className="board-annotation__marker-keyline" r="0.31" />
            <circle className="board-annotation__marker-face" r="0.255" />
            {marker.role === "blocked" && (
              <>
                <path
                  className="board-annotation__marker-symbol-keyline"
                  d="M -0.13 -0.13 L 0.13 0.13 M 0.13 -0.13 L -0.13 0.13"
                />
                <path
                  className="board-annotation__marker-symbol-face"
                  d="M -0.13 -0.13 L 0.13 0.13 M 0.13 -0.13 L -0.13 0.13"
                />
              </>
            )}
          </g>
        ))}
        {arrows.map(({ arrow, line }, index) => (
          <g key={`${arrow.from_square}-${arrow.to_square}-${arrow.role}-${index}`}>
            <line
              className={`board-annotation__arrow-keyline is-${arrow.role}`}
              x1={line.start.x}
              y1={line.start.y}
              x2={line.keylineEnd.x}
              y2={line.keylineEnd.y}
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
            <circle
              className="board-annotation__badge-keyline"
              r={REVIEW_DIAGRAM_GEOMETRY.badgeRadius}
            />
            <circle
              className="board-annotation__badge-face"
              r={REVIEW_DIAGRAM_GEOMETRY.badgeFaceRadius}
            />
            <g transform="translate(-0.1 -0.1) scale(0.0083)">
              <ReviewGlyphLayers
                badge={positionedBadge.badge.kind}
                className="board-annotation__glyph"
              />
            </g>
          </g>
        )}
      </g>
    </svg>
  );
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
