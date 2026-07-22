import { useState } from "react";
import { pieceOptionForLabel, predictionNeedsReview, squareName } from "../board";
import type { Orientation } from "../types";
import ChessPiece from "./ChessPiece";
import PiecePicker from "./PiecePicker";

interface BoardEditorProps {
  labels: number[];
  predictedLabels: number[];
  confidences: number[];
  probabilities: number[][];
  orientation: Orientation;
  onChange: (labels: number[]) => void;
}

export default function BoardEditor({
  labels,
  predictedLabels,
  confidences,
  probabilities,
  orientation,
  onChange,
}: BoardEditorProps) {
  const [selectedSquare, setSelectedSquare] = useState<number | null>(null);

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
            const needsReview = predictionNeedsReview(
              predictedLabels[index] ?? 0,
              confidence,
              probabilities[index] ?? [],
            );
            const square = squareName(index, orientation);
            const option = pieceOptionForLabel(piece);
            return (
              <button
                key={index}
                type="button"
                role="gridcell"
                className={[
                  "board-square",
                  (row + col) % 2 === 0 ? "is-light" : "is-dark",
                  needsReview ? "is-uncertain" : "",
                  corrected ? "is-corrected" : "",
                  selectedSquare === index ? "is-selected" : "",
                ].join(" ")}
                aria-label={`${square}: ${option.name}`}
                title={`${square} · ${option.name} · ${Math.round(confidence * 100)}% confidence`}
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
        Adapted from{" "}
        <a
          href="https://www.figma.com/community/file/971870797656870866/chess-simple-assets"
          target="_blank"
          rel="noreferrer"
        >
          Chess Simple Assets by Maciej Świerczek
        </a>{" "}
        ·{" "}
        <a
          href="https://creativecommons.org/licenses/by/4.0/"
          target="_blank"
          rel="noreferrer"
        >
          CC BY 4.0
        </a>
      </p>
    </div>
  );
}
