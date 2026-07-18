import { useState } from "react";
import { pieceNames, pieceSymbols, predictionNeedsReview, squareName } from "../board";
import type { Orientation } from "../types";
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
          const name = squareName(index, orientation);
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
              aria-label={`${name}: ${pieceNames[piece] ?? "Unknown"}`}
              title={`${name} · ${pieceNames[piece] ?? "Unknown"} · ${Math.round(confidence * 100)}% confidence`}
              onClick={() => setSelectedSquare(index)}
            >
              <span className="board-square__coordinate">{name}</span>
              {piece !== 0 && <span className="chess-symbol board-square__piece">{pieceSymbols[piece]}</span>}
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
  );
}
