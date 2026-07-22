import { pieceForLabel, pieceNames } from "../board";
import ChessPiece from "./ChessPiece";

interface PiecePickerProps {
  square: string;
  value: number;
  onPick: (piece: number) => void;
  onClose: () => void;
}

export default function PiecePicker({ square, value, onPick, onClose }: PiecePickerProps) {
  return (
    <div className="piece-picker" role="dialog" aria-label={`Choose piece for ${square}`}>
      <div className="piece-picker__heading">
        <span>
          Set <strong>{square}</strong>
        </span>
        <button type="button" className="text-button" onClick={onClose}>
          Done
        </button>
      </div>
      <div className="piece-picker__grid">
        {pieceNames.map((name, id) => {
          const artwork = pieceForLabel(id);
          return (
            <button
              key={id}
              type="button"
              className={`piece-picker__piece ${id === value ? "is-active" : ""}`}
              title={name}
              aria-label={name}
              aria-pressed={id === value}
              onClick={() => onPick(id)}
            >
              {artwork ? (
                <ChessPiece color={artwork.color} piece={artwork.type} />
              ) : (
                <span className="empty-symbol">·</span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
