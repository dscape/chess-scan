import { useEffect, useMemo, useRef, useState } from "react";
import { Chess } from "chess.js";
import { createPositionReview } from "../api";
import StockfishClient from "../engine/StockfishClient";
import type { EngineScore } from "../engine/uci";
import type {
  Orientation,
  PositionReview as PositionReviewData,
  ReviewAnnotation,
  ReviewedPosition,
} from "../types";
import InteractiveBoard, { type AttemptedMove } from "./InteractiveBoard";

type PositionReviewProps = {
  position: ReviewedPosition;
  onScanAnother: () => void;
};

type ReviewStatus = "loading" | "ready" | "error";
type Evaluations = Record<string, EngineScore>;

const INITIAL_ANALYSIS_MS = 2500;
const LIVE_ANALYSIS_MS = 750;

export default function PositionReview({
  position,
  onScanAnother,
}: PositionReviewProps) {
  const [status, setStatus] = useState<ReviewStatus>("loading");
  const [review, setReview] = useState<PositionReviewData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [retry, setRetry] = useState(0);
  const [playedMoves, setPlayedMoves] = useState<AttemptedMove[]>([]);
  const [firstAttempt, setFirstAttempt] = useState<AttemptedMove | null>(null);
  const [evaluations, setEvaluations] = useState<Evaluations>({});
  const [evaluatingFen, setEvaluatingFen] = useState<string | null>(null);
  const [hoveredCue, setHoveredCue] = useState<ReviewAnnotation | null>(null);
  const [pinnedCue, setPinnedCue] = useState<ReviewAnnotation | null>(null);
  const engine = useRef<StockfishClient | null>(null);
  const playedMovesRef = useRef(playedMoves);
  playedMovesRef.current = playedMoves;

  const root = useMemo(() => new Chess(position.full_fen), [position.full_fen]);
  const moveUcis = useMemo(() => playedMoves.map((move) => move.uci), [playedMoves]);
  const current = useMemo(
    () => positionAfter(position.full_fen, moveUcis),
    [position.full_fen, moveUcis],
  );
  const currentFen = current.fen();
  const rootIsTerminal = root.isGameOver();
  const revealed = rootIsTerminal || firstAttempt !== null;
  const defaultCue = revealed ? review?.explanation[0] ?? null : null;
  const boardCue = hoveredCue ?? pinnedCue ?? defaultCue;
  const currentScore = evaluations[currentFen] ?? null;
  const lastMove = playedMoves.at(-1) ?? null;

  useEffect(() => () => {
    const activeEngine = engine.current;
    engine.current = null;
    activeEngine?.dispose();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    setStatus("loading");
    setReview(null);
    setError(null);
    setPinnedCue(null);
    setHoveredCue(null);

    async function loadReview() {
      try {
        if (rootIsTerminal) {
          const result = await createPositionReview(
            { fen: position.full_fen, line: null },
            controller.signal,
          );
          if (!active) return;
          setReview(result);
          setStatus("ready");
          return;
        }

        engine.current ??= new StockfishClient();
        setEvaluatingFen(position.full_fen);
        const analysis = await engine.current.analyze(position.full_fen, {
          budgetMs: INITIAL_ANALYSIS_MS,
          signal: controller.signal,
          onUpdate: (line) => {
            if (!active || playedMovesRef.current.length > 0) return;
            cacheEvaluation(position.full_fen, line.score, setEvaluations);
          },
        });
        if (!active) return;
        const line = analysis.line;
        if (!line) throw new Error("Stockfish did not return a principal line.");
        cacheEvaluation(position.full_fen, line.score, setEvaluations);
        const result = await createPositionReview(
          { fen: position.full_fen, line },
          controller.signal,
        );
        if (!active) return;
        setReview(result);
        setStatus("ready");
      } catch (cause) {
        if (!active || isAbortError(cause)) return;
        engine.current?.dispose();
        engine.current = null;
        setError(messageFrom(cause));
        setStatus("error");
      } finally {
        if (active) setEvaluatingFen(null);
      }
    }

    void loadReview();
    return () => {
      active = false;
      controller.abort();
    };
  }, [position.full_fen, retry, rootIsTerminal]);

  useEffect(() => {
    if (status !== "ready" || playedMoves.length === 0 || current.isGameOver()) {
      setEvaluatingFen(null);
      return;
    }
    const cached = evaluations[currentFen];
    if (cached) return;

    const controller = new AbortController();
    const timeout = window.setTimeout(() => {
      async function evaluateCurrentPosition() {
        try {
          engine.current ??= new StockfishClient();
          setEvaluatingFen(currentFen);
          const analysis = await engine.current.analyze(currentFen, {
            budgetMs: LIVE_ANALYSIS_MS,
            signal: controller.signal,
            onUpdate: (line) => {
              if (!controller.signal.aborted) {
                cacheEvaluation(currentFen, line.score, setEvaluations);
              }
            },
          });
          const line = analysis.line;
          if (line) cacheEvaluation(currentFen, line.score, setEvaluations);
        } catch (cause) {
          if (!isAbortError(cause)) {
            engine.current?.dispose();
            engine.current = null;
          }
        } finally {
          if (!controller.signal.aborted) {
            setEvaluatingFen((fen) => fen === currentFen ? null : fen);
          }
        }
      }
      void evaluateCurrentPosition();
    }, 140);

    return () => {
      window.clearTimeout(timeout);
      controller.abort();
    };
  }, [currentFen, playedMoves.length, status]);

  function play(move: AttemptedMove) {
    setPlayedMoves((moves) => [...moves, move]);
    setFirstAttempt((attempt) => attempt ?? move);
    setPinnedCue(null);
  }

  function undo() {
    setPlayedMoves((moves) => moves.slice(0, -1));
    setPinnedCue(null);
  }

  function resetBoard() {
    setPlayedMoves([]);
    setFirstAttempt(null);
    setPinnedCue(null);
    setHoveredCue(null);
  }

  function pinCue(cue: ReviewAnnotation) {
    setPinnedCue((currentCue) => currentCue === cue ? null : cue);
  }

  const cueEvents = (cue: ReviewAnnotation, defaultActive = false) => ({
    active: pinnedCue === cue || (defaultActive && pinnedCue === null && hoveredCue === null),
    onEnter: () => setHoveredCue(cue),
    onLeave: () => setHoveredCue(null),
    onToggle: () => pinCue(cue),
  });

  return (
    <main className="position-review">
      <header className="position-review__header">
        <button type="button" className="brand position-review__brand" onClick={onScanAnother}>
          <span className="brand__mark" aria-hidden="true"><i /><i /><i /><i /></span>
          <span><strong>Chess Scan</strong><em>review</em></span>
        </button>
        <div className={`engine-status is-${status}`} role="status">
          <i aria-hidden="true" />
          <span>{engineStatus(status, evaluatingFen === currentFen)}</span>
        </div>
      </header>

      <section className="position-review__layout">
        <div className="position-review__board-column">
          <div className="board-context">
            <div>
              <span>{turnName(current.turn())} to move</span>
              <small>{lastMove ? `After ${lastMove.san}` : "Starting position"}</small>
            </div>
            <span className="board-context__topic">
              {review?.topic.name ?? (status === "loading" ? "Finding the idea…" : "Position")}
            </span>
          </div>

          <div className="position-review__board-stage">
            <EvalBar
              score={currentScore}
              turn={current.turn()}
              orientation={position.orientation}
              loading={evaluatingFen === currentFen || (status === "loading" && !currentScore)}
              terminal={terminalResult(current)}
            />
            <InteractiveBoard
              fen={position.full_fen}
              orientation={position.orientation}
              moves={moveUcis}
              ply={moveUcis.length}
              interactive={!current.isGameOver()}
              cue={boardCue}
              onMove={play}
            />
          </div>

          <div className="board-tools" aria-label="Board controls">
            <div>
              <button type="button" onClick={undo} disabled={playedMoves.length === 0}>Undo</button>
              <button
                type="button"
                onClick={resetBoard}
                disabled={playedMoves.length === 0 && firstAttempt === null}
              >
                Reset
              </button>
            </div>
            <span className="board-tools__eval">
              Local eval <b>{evaluationLabel(currentScore, current.turn(), current)}</b>
            </span>
          </div>
        </div>

        <article className="review-notes" aria-live={status === "loading" ? "off" : "polite"}>
          {status === "loading" && (
            <div className="review-loading">
              <span className="review-loading__mark" aria-hidden="true" />
              <p className="commentary-kicker">Reading the position</p>
              <h1>Finding the useful clue…</h1>
              <p>The board is live while the local engine checks the first move.</p>
            </div>
          )}

          {status === "error" && (
            <>
              <p className="commentary-kicker">Review unavailable</p>
              <h1>Keep the board.</h1>
              <p>{error}</p>
              <div className="review-actions">
                <button type="button" className="primary-button" onClick={() => setRetry((value) => value + 1)}>
                  Try the review again
                </button>
                <a className="secondary-button" href={position.lichess_url} target="_blank" rel="noreferrer">
                  Open in Lichess ↗
                </a>
              </div>
            </>
          )}

          {status === "ready" && review && (
            <>
              <BoardReference
                className="topic-reference"
                cue={review.hint}
                {...cueEvents(review.hint)}
              >
                <span>Position topic</span>
                <strong>{review.topic.name}</strong>
                <small>Show on board</small>
              </BoardReference>

              {!revealed ? (
                <div className="review-prompt">
                  <p className="commentary-kicker">{turnName(root.turn())} to move</p>
                  <h1>What is the first move?</h1>
                  <BoardReference
                    className="hint-reference"
                    cue={review.hint}
                    {...cueEvents(review.hint)}
                  >
                    <span className="annotation-index">Hint</span>
                    <span className="hint-reference__copy">{review.hint.text}</span>
                    <small>Tap to mark the clues on the board</small>
                  </BoardReference>
                  <p className="spoiler-note"><i aria-hidden="true" /> No move names until you try.</p>
                </div>
              ) : (
                <div className="review-explanation">
                  <p className="commentary-kicker">{review.evaluation}</p>
                  <h1>{attemptVerdict(firstAttempt, review)}</h1>
                  <p className="review-explanation__lead">
                    Follow the red move arrow, then use each note to inspect the idea on the board.
                  </p>
                  <div className="annotation-list">
                    {review.explanation.map((cue, index) => (
                      <BoardReference
                        key={`${cue.label}-${index}`}
                        className="annotation-reference"
                        cue={cue}
                        {...cueEvents(cue, index === 0)}
                      >
                        <span className="annotation-index">{String(index + 1).padStart(2, "0")}</span>
                        <span>
                          <strong>{cue.label}</strong>
                          <span className="annotation-reference__copy">{cue.text}</span>
                        </span>
                        <small>Board</small>
                      </BoardReference>
                    ))}
                  </div>
                  <button type="button" className="try-again-button" onClick={resetBoard}>
                    Hide the answer & try again
                  </button>
                </div>
              )}
            </>
          )}
        </article>
      </section>

      <footer className="position-review__footer">
        <span>
          <a href="/stockfish/Copying.txt" target="_blank" rel="noreferrer">Stockfish 18 lite · GPLv3</a>
          {" · "}
          <a href="/stockfish/SOURCE.md" target="_blank" rel="noreferrer">Source</a>
        </span>
        <div>
          <button type="button" onClick={onScanAnother}>Scan another position</button>
          <a href={position.lichess_url} target="_blank" rel="noreferrer">Advanced analysis ↗</a>
        </div>
      </footer>
    </main>
  );
}

function BoardReference({
  children,
  className,
  cue,
  active,
  onEnter,
  onLeave,
  onToggle,
}: {
  children: React.ReactNode;
  className: string;
  cue: ReviewAnnotation;
  active: boolean;
  onEnter: () => void;
  onLeave: () => void;
  onToggle: () => void;
}) {
  const hasBoardCue = cue.squares.length > 0 || cue.arrows.length > 0;
  return (
    <button
      type="button"
      className={`${className}${active ? " is-active" : ""}${hasBoardCue ? "" : " is-static"}`}
      aria-pressed={hasBoardCue ? active : undefined}
      aria-disabled={!hasBoardCue}
      onPointerEnter={hasBoardCue ? onEnter : undefined}
      onPointerLeave={hasBoardCue ? onLeave : undefined}
      onFocus={hasBoardCue ? onEnter : undefined}
      onBlur={hasBoardCue ? onLeave : undefined}
      onClick={hasBoardCue ? onToggle : undefined}
    >
      {children}
    </button>
  );
}

function EvalBar({
  score,
  turn,
  orientation,
  loading,
  terminal,
}: {
  score: EngineScore | null;
  turn: "w" | "b";
  orientation: Orientation;
  loading: boolean;
  terminal: "white" | "black" | "draw" | null;
}) {
  const whiteShare = terminal === "white"
    ? 100
    : terminal === "black"
      ? 0
      : terminal === "draw"
        ? 50
        : evaluationShare(score, turn);
  const whiteAtBottom = orientation === "white";
  const whiteFavored = whiteShare >= 50;
  const scoreAtBottom = whiteFavored === whiteAtBottom;
  const label = terminal === "white"
    ? "1–0"
    : terminal === "black"
      ? "0–1"
      : terminal === "draw"
        ? "½"
        : scoreText(score, turn);

  return (
    <div
      className={`eval-bar${loading ? " is-loading" : ""}`}
      role="meter"
      aria-label="Position evaluation"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(whiteShare)}
      aria-valuetext={label === "—" ? "Evaluation loading" : `Evaluation ${label}`}
    >
      <div
        className={`eval-bar__white is-${whiteAtBottom ? "bottom" : "top"}`}
        style={{ height: `${whiteShare}%` }}
      />
      <span className={`eval-bar__score is-${scoreAtBottom ? "bottom" : "top"} on-${whiteFavored ? "white" : "black"}`}>{label}</span>
      <span className={`eval-bar__side is-top on-${whiteAtBottom ? "black" : "white"}`}>{whiteAtBottom ? "B" : "W"}</span>
      <span className={`eval-bar__side is-bottom on-${whiteAtBottom ? "white" : "black"}`}>{whiteAtBottom ? "W" : "B"}</span>
    </div>
  );
}

function cacheEvaluation(
  fen: string,
  score: EngineScore,
  setEvaluations: React.Dispatch<React.SetStateAction<Evaluations>>,
) {
  setEvaluations((current) => {
    const previous = current[fen];
    if (previous?.kind === score.kind && previous.value === score.value) return current;
    return { ...current, [fen]: score };
  });
}

function positionAfter(fen: string, moves: string[]): Chess {
  const chess = new Chess(fen);
  for (const move of moves) chess.move(move);
  return chess;
}

function attemptVerdict(
  attempt: AttemptedMove | null,
  review: PositionReviewData,
): string {
  if (!review.best_move) return review.evaluation;
  if (attempt?.uci === review.best_move.uci) return `${attempt.san} finds the idea.`;
  if (attempt) return `You tried ${attempt.san}. The key move is ${review.best_move.san}.`;
  return `The key move is ${review.best_move.san}.`;
}

function evaluationShare(score: EngineScore | null, turn: "w" | "b"): number {
  if (!score) return 50;
  const signed = score.value * (turn === "w" ? 1 : -1);
  if (score.kind === "mate") return signed >= 0 ? 98 : 2;
  return Math.min(98, Math.max(2, 50 + 48 * Math.tanh(signed / 400)));
}

function scoreText(score: EngineScore | null, turn: "w" | "b"): string {
  if (!score) return "—";
  const signed = score.value * (turn === "w" ? 1 : -1);
  if (score.kind === "mate") return signed >= 0 ? `M${Math.abs(signed)}` : `−M${Math.abs(signed)}`;
  const pawns = signed / 100;
  return `${pawns >= 0 ? "+" : "−"}${Math.abs(pawns).toFixed(1)}`;
}

function evaluationLabel(score: EngineScore | null, turn: "w" | "b", chess: Chess): string {
  const terminal = terminalResult(chess);
  if (terminal === "white") return "1–0";
  if (terminal === "black") return "0–1";
  if (terminal === "draw") return "½–½";
  return scoreText(score, turn);
}

function terminalResult(chess: Chess): "white" | "black" | "draw" | null {
  if (chess.isCheckmate()) return chess.turn() === "w" ? "black" : "white";
  if (chess.isDraw()) return "draw";
  return null;
}

function turnName(turn: "w" | "b"): string {
  return turn === "w" ? "White" : "Black";
}

function engineStatus(status: ReviewStatus, evaluatingCurrent: boolean): string {
  if (status === "error") return "Local analysis unavailable";
  if (status === "loading" || evaluatingCurrent) return "Stockfish is checking locally";
  return "Local analysis ready";
}

function messageFrom(cause: unknown): string {
  return cause instanceof Error ? cause.message : "The position could not be reviewed.";
}

function isAbortError(cause: unknown): boolean {
  return cause instanceof DOMException && cause.name === "AbortError";
}
