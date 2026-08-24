import { useState } from "react";
import { pieceOptionForLabel, predictionNeedsReview, squareName } from "../board";
import type { Orientation } from "../types";
import ChessPiece from "./ChessPiece";
import ChessPieceAttribution from "./ChessPieceAttribution";
import PiecePicker from "./PiecePicker";

interface BoardEditorProps {
  labels: number[];
  predictedLabels: number[];
  confidences: number[];
  probabilities: number[][];
  frameAgreements: number[];
  viewAgreements: number[];
  predictionMethod: "single-view-v1" | "multi-view-mean-v1";
  glareSquares: number[];
  orientation: Orientation;
  onChange: (labels: number[]) => void;
}

export default function BoardEditor({
  labels,
  predictedLabels,
  confidences,
  probabilities,
  frameAgreements,
  viewAgreements,
  predictionMethod,
  glareSquares,
  orientation,
  onChange,
}: BoardEditorProps) {
  const [selectedSquare, setSelectedSquare] = useState<number | null>(null);
  const glareSquareSet = new Set(glareSquares);

  function setPiece(piece: number) {
    if (selectedSquare === null) return;
    const next = [...labels];
    next[selectedSquare] = piece;
    onChange(next);
    setSelectedSquare(null);
  }

  return (
    <div className="board-editor-wrap">
      <div className="board-editor-stage">
        <div className="board-editor" role="grid" aria-label="Editable predicted chess position">
          {labels.map((piece, index) => {
            const row = Math.floor(index / 8);
            const col = index % 8;
            const confidence = confidences[index] ?? 0;
            const corrected = piece !== predictedLabels[index];
            const frameAgreement = frameAgreements[index] ?? 0;
            const viewAgreement = viewAgreements[index] ?? 0;
            const viewsDisagree = Math.min(frameAgreement, viewAgreement) < 2 / 3;
            const needsReview = predictionNeedsReview(
              predictedLabels[index] ?? 0,
              confidence,
              probabilities[index] ?? [],
              frameAgreement,
              viewAgreement,
              predictionMethod === "multi-view-mean-v1",
            );
            const square = squareName(index, orientation);
            const option = pieceOptionForLabel(piece);
            const hasGlare = glareSquareSet.has(index);
            return (
              <button
                key={index}
                type="button"
                role="gridcell"
                className={[
                  "board-square",
                  (row + col) % 2 === 0 ? "is-light" : "is-dark",
                  needsReview ? "is-uncertain" : "",
                  hasGlare ? "is-glare" : "",
                  corrected ? "is-corrected" : "",
                  selectedSquare === index ? "is-selected" : "",
                ].join(" ")}
                aria-label={`${square}: ${option.name}${viewsDisagree ? "; model views disagree" : ""}${hasGlare ? "; glare detected" : ""}`}
                title={`${square} · ${option.name} · ${Math.round(confidence * 100)}% confidence · ${Math.round(frameAgreement * 100)}% frame agreement · ${Math.round(viewAgreement * 100)}% crop agreement${hasGlare ? " · glare detected" : ""}`}
                onClick={() => setSelectedSquare(index)}
              >
                <span className="board-square__coordinate">{square}</span>
                {option.piece && (
                  <ChessPiece
                    className="board-square__piece"
                    color={option.piece.color}
                    piece={option.piece.type}
                  />
                )}
                {corrected && <span className="board-square__correction" aria-label="Corrected" />}
              </button>
            );
          })}
        </div>
        {selectedSquare !== null && (
          <PiecePicker
            square={squareName(selectedSquare, orientation)}
            value={labels[selectedSquare] ?? 0}
            onPick={setPiece}
            onClose={() => setSelectedSquare(null)}
          />
        )}
      </div>
      <p className="piece-attribution">
        <ChessPieceAttribution />
      </p>
    </div>
  );
}
