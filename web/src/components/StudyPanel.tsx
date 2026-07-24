import {
  studyBoardCue,
  studyMoveGradeSymbol,
  type StudyState,
} from "../studyAnalysis";
import ReviewGlyph from "./ReviewGlyph";

type StudyPanelProps = {
  state: StudyState;
  turn: "w" | "b";
  terminal: "white" | "black" | "draw" | null;
  onLeave: () => void;
};

export default function StudyPanel({
  state,
  turn,
  terminal,
  onLeave,
}: StudyPanelProps) {
  const loading = state.status === "idle" || state.status === "loading";
  const node = state.status === "ready" || state.status === "review-error"
    ? state.node
    : null;
  const topMoveSan = node?.topMoveSan
    ?? (state.status === "loading" ? state.topMoveSan : null);
  const cue = studyBoardCue(node?.review ?? null);
  const grade = node?.grade ?? null;
  const gradeError = node?.gradeError ?? null;
  const headline = terminal
    ? terminal === "draw"
      ? "The line ends in a draw."
      : `${terminal === "white" ? "White" : "Black"} wins.`
    : grade
      ? `${grade.label}.`
      : state.status === "error" || state.status === "review-error"
        ? "Analysis paused."
        : "Play out the position.";

  return (
    <div className="study-panel">
      <p className="commentary-kicker">
        {terminal
          ? "Interactive study · line complete"
          : `Interactive study · ${turn === "w" ? "White" : "Black"} to move`}
      </p>
      <h1>{headline}</h1>
      <p className="study-panel__lead">
        {terminal
          ? "This line has ended. Undo the last move or reset the board to keep studying."
          : "Move for either side. Stockfish checks every position locally before the board unlocks."}
      </p>

      {grade && (
        <aside className={`study-grade is-${grade.kind}`} role="status">
          <span className="study-grade__mark" aria-hidden="true">
            {studyMoveGradeSymbol(grade.kind)}
          </span>
          <span>
            <strong>{grade.label}</strong>
            <small>{grade.detail}</small>
          </span>
        </aside>
      )}

      {terminal === null && (
        <section className={`study-next${loading ? " is-loading" : ""}`}>
          <span className="study-next__mark" aria-hidden="true">
            <ReviewGlyph badge="engine" />
          </span>
          <span>
            <small>Stockfish's next move</small>
            <strong>{topMoveSan ?? (state.status === "error" ? "Unavailable" : "Checking…")}</strong>
          </span>
          <p>
            {cue
              ? `${cue.label} · ${cue.text}`
              : state.status === "error"
                ? state.message
                : state.status === "review-error"
                  ? state.failure.message
                  : "The arrow appears as soon as the principal move settles."}
          </p>
        </section>
      )}

      {state.status === "review-error" && (
        <p className="study-panel__warning" role="status">
          {state.failure.retryable
            ? "The top move is ready, but tactical annotations could not be loaded. Retry analysis before continuing."
            : "The top move is ready, but tactical annotations are unavailable for this position."}
        </p>
      )}

      {gradeError && (
        <p className="study-panel__warning" role="status">
          {gradeError.retryable
            ? "This position is ready, but the previous move could not be graded. Retry analysis to restore its grade."
            : "This position is ready, but the previous move cannot be graded from the available analysis."}
        </p>
      )}

      <div className="study-grade-legend" aria-label="Move grade legend">
        {(["brilliant", "good", "bad", "blunder"] as const).map((kind) => (
          <span key={kind} className={`is-${kind}`}>
            <b>{studyMoveGradeSymbol(kind)}</b>
            {kind === "good" ? "Only move" : `${kind.charAt(0).toUpperCase()}${kind.slice(1)}`}
          </span>
        ))}
      </div>

      <button type="button" className="try-again-button" onClick={onLeave}>
        Back to the first-move review
      </button>
    </div>
  );
}
