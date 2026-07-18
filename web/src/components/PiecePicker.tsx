import { pieceNames, pieceSymbols } from "../board";

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
        {pieceSymbols.map((symbol, id) => (
          <button
            key={id}
            type="button"
            className={`piece-picker__piece ${id === value ? "is-active" : ""}`}
            title={pieceNames[id]}
            aria-label={pieceNames[id]}
            aria-pressed={id === value}
            onClick={() => onPick(id)}
          >
            <span className={id === 0 ? "empty-symbol" : "chess-symbol"}>{symbol}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
