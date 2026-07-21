import { useEffect, useMemo, useRef, useState } from "react";
import { Chess } from "chess.js";
import { createPositionReview, ratePositionReview } from "../api";
import { positionAt } from "../board";
import StockfishClient, { StockfishError } from "../engine/StockfishClient";
import type { EngineLine, EngineScore } from "../engine/uci";
import type {
  Orientation,
  PositionReview as PositionReviewData,
  ReviewAnalysis,
  ReviewAnnotation,
  ReviewedPosition,
  ReviewLine,
} from "../types";
import InteractiveBoard, { type AttemptedMove } from "./InteractiveBoard";

type PositionReviewProps = {
  position: ReviewedPosition;
  onScanAnother: () => void;
};

type ReviewState =
  | { status: "loading" }
  | { status: "ready"; review: PositionReviewData }
  | { status: "error"; message: string };

type CachedEvaluation = {
  score: EngineScore;
  complete: boolean;
};

type EvaluationPublisher = {
  queue: (score: EngineScore) => void;
  complete: (score: EngineScore) => void;
  cancel: () => void;
};

type RatingState = {
  reviewId: string | null;
  status: "idle" | "sending" | "done" | "error";
};

type AnalysisError = {
  source: "engine" | "api";
  message: string;
};

const INITIAL_ANALYSIS_MS = 2500;
const ATTEMPT_ANALYSIS_MS = 1600;
const LIVE_ANALYSIS_MS = 750;
const EVALUATION_UPDATE_MS = 150;
const EVALUATION_CACHE_LIMIT = 48;

export default function PositionReview({
  position,
  onScanAnother,
}: PositionReviewProps) {
  const [reviewState, setReviewState] = useState<ReviewState>({ status: "loading" });
  const [retry, setRetry] = useState(0);
  const [liveRetry, setLiveRetry] = useState(0);
  const [liveError, setLiveError] = useState<AnalysisError | null>(null);
  const [ratingState, setRatingState] = useState<RatingState>({
    reviewId: null,
    status: "idle",
  });
  const [ratingReason, setRatingReason] = useState<
    "incorrect_chess" | "irrelevant_topic" | "unclear" | "equivalent_move_rejected"
      | "too_verbose" | "missing_detail" | "other"
  >("incorrect_chess");
  const [playedMoves, setPlayedMoves] = useState<AttemptedMove[]>([]);
  const [linePreview, setLinePreview] = useState<"best" | "attempt" | null>(null);
  const [firstAttempt, setFirstAttempt] = useState<AttemptedMove | null>(null);
  const [pendingAttempt, setPendingAttempt] = useState<AttemptedMove | null>(null);
  const checkingAttempt = pendingAttempt !== null;
  const [currentScore, setCurrentScore] = useState<EngineScore | null>(null);
  const [evaluatingFen, setEvaluatingFen] = useState<string | null>(null);
  const [hoveredCue, setHoveredCue] = useState<ReviewAnnotation | null>(null);
  const [pinnedCue, setPinnedCue] = useState<ReviewAnnotation | null>(null);
  const engine = useRef<StockfishClient | null>(null);
  const reviewLines = useRef(new Map<string, EngineLine[]>());
  const initialReview = useRef<PositionReviewData | null>(null);
  const evaluationCache = useRef(new Map<string, CachedEvaluation>());
  const analysisGeneration = useRef(0);
  const ratingGeneration = useRef(0);

  const root = useMemo(() => new Chess(position.full_fen), [position.full_fen]);
  const moveUcis = useMemo(() => playedMoves.map((move) => move.uci), [playedMoves]);
  const current = useMemo(
    () => positionAt(position.full_fen, moveUcis),
    [position.full_fen, moveUcis],
  );
  const currentFen = current.fen();
  const currentFenRef = useRef(currentFen);
  currentFenRef.current = currentFen;
  const rootIsTerminal = root.isGameOver();
  const currentTerminal = terminalResult(current);
  const revealed = rootIsTerminal || firstAttempt !== null;
  const review = reviewState.status === "ready" ? reviewState.review : null;
  const ratingStatus = ratingState.reviewId === review?.review_id
    ? ratingState.status
    : "idle";
  const defaultCue = revealed ? review?.explanation[0] ?? null : null;
  const rootHoveredCue = hoveredCue?.ply === 0 ? hoveredCue : null;
  const rootDefaultCue = defaultCue?.ply === 0 ? defaultCue : null;
  const boardCue = pinnedCue
    ?? (playedMoves.length === 0 ? rootHoveredCue ?? rootDefaultCue : null);
  const lastMove = playedMoves.at(-1) ?? null;

  useEffect(() => () => {
    const activeEngine = engine.current;
    engine.current = null;
    activeEngine?.dispose();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const generation = ++analysisGeneration.current;
    let active = true;
    let updates: EvaluationPublisher | null = null;

    setReviewState({ status: "loading" });
    setPinnedCue(null);
    setHoveredCue(null);
    setLiveError(null);
    ratingGeneration.current += 1;
    setRatingState({ reviewId: null, status: "idle" });
    setLinePreview(null);
    setFirstAttempt(null);
    setPendingAttempt(null);

    async function loadReview() {
      if (rootIsTerminal) {
        setEvaluatingFen(null);
        try {
          const result = await createPositionReview(
            {
              fen: position.full_fen,
              feedback_id: position.feedback_id,
              analysis: null,
            },
            controller.signal,
          );
          if (active) {
            initialReview.current = result;
            setReviewState({ status: "ready", review: result });
          }
        } catch (cause) {
          if (active && !isAbortError(cause)) {
            setReviewState({ status: "error", message: messageFrom(cause) });
          }
        }
        return;
      }

      let lines = reviewLines.current.get(position.full_fen) ?? null;
      if (!lines) {
        try {
          engine.current ??= new StockfishClient();
          setEvaluatingFen(position.full_fen);
          updates = createEvaluationPublisher(
            position.full_fen,
            evaluationCache.current,
            currentFenRef,
            setCurrentScore,
          );
          const analysis = await engine.current.analyze(position.full_fen, {
            budgetMs: INITIAL_ANALYSIS_MS,
            multiPv: 3,
            newGame: true,
            requireStable: true,
            signal: controller.signal,
            onUpdate: (candidates) => {
              const primary = candidates.find((candidate) => candidate.rank === 1);
              if (active && primary) updates?.queue(primary.score);
            },
          });
          if (!active) return;
          lines = analysis.lines;
          const primary = lines.find((line) => line.rank === 1);
          if (!primary) throw new Error("Stockfish did not return a principal line.");
          updates.complete(primary.score);
          reviewLines.current.set(position.full_fen, lines);
        } catch (cause) {
          if (!active || isAbortError(cause)) return;
          engine.current?.dispose();
          engine.current = null;
          setReviewState({ status: "error", message: messageFrom(cause) });
          return;
        } finally {
          updates?.cancel();
          clearEvaluatingFen(generation, position.full_fen);
        }
      } else {
        const primary = lines.find((line) => line.rank === 1);
        if (!primary) throw new Error("Cached analysis has no principal line.");
        storeEvaluation(
          evaluationCache.current,
          position.full_fen,
          { score: primary.score, complete: true },
        );
        if (currentFenRef.current === position.full_fen) setCurrentScore(primary.score);
        clearEvaluatingFen(generation, position.full_fen);
      }

      try {
        const result = await createPositionReview(
          {
            fen: position.full_fen,
            feedback_id: position.feedback_id,
            analysis: reviewAnalysis(lines),
          },
          controller.signal,
        );
        if (active) {
          initialReview.current = result;
          setReviewState({ status: "ready", review: result });
        }
      } catch (cause) {
        if (active && !isAbortError(cause)) {
          setReviewState({ status: "error", message: messageFrom(cause) });
        }
      }
    }

    function clearEvaluatingFen(requestGeneration: number, fen: string) {
      if (analysisGeneration.current !== requestGeneration) return;
      setEvaluatingFen((currentValue) => currentValue === fen ? null : currentValue);
    }

    void loadReview();
    return () => {
      active = false;
      controller.abort();
      updates?.cancel();
      clearEvaluatingFen(generation, position.full_fen);
    };
  }, [position.feedback_id, position.full_fen, retry, rootIsTerminal]);

  useEffect(() => {
    if (!pendingAttempt) return;
    const attempt = pendingAttempt;
    const cachedRootLines = reviewLines.current.get(position.full_fen);
    if (!cachedRootLines) return;
    const rootLines = cachedRootLines;

    const controller = new AbortController();
    let active = true;
    setLiveError(null);

    async function checkAttempt() {
      try {
        let attemptLine: EngineLine;
        try {
          engine.current ??= new StockfishClient();
          const analysis = await engine.current.analyze(position.full_fen, {
            budgetMs: ATTEMPT_ANALYSIS_MS,
            multiPv: 1,
            requireStable: true,
            searchMoves: [attempt.uci],
            signal: controller.signal,
          });
          const line = analysis.lines[0];
          if (!line) throw new StockfishError("Stockfish could not evaluate your move.");
          attemptLine = line;
        } catch (cause) {
          if (active && !isAbortError(cause)) {
            if (cause instanceof StockfishError) {
              engine.current?.dispose();
              engine.current = null;
            }
            setLiveError({ source: "engine", message: messageFrom(cause) });
            if (initialReview.current) {
              setReviewState({ status: "ready", review: initialReview.current });
            }
          }
          return;
        }

        try {
          const result = await createPositionReview(
            {
              fen: position.full_fen,
              feedback_id: position.feedback_id,
              analysis: reviewAnalysis(rootLines, attempt, attemptLine),
            },
            controller.signal,
          );
          if (active) setReviewState({ status: "ready", review: result });
        } catch (cause) {
          if (active && !isAbortError(cause)) {
            setLiveError({ source: "api", message: messageFrom(cause) });
            if (initialReview.current) {
              setReviewState({ status: "ready", review: initialReview.current });
            }
          }
        }
      } finally {
        if (active) setPendingAttempt(null);
      }
    }

    void checkAttempt();
    return () => {
      active = false;
      controller.abort();
    };
  }, [pendingAttempt, position.feedback_id, position.full_fen]);

  useEffect(() => {
    if (reviewState.status !== "ready") return;

    const generation = ++analysisGeneration.current;
    const cached = readEvaluation(evaluationCache.current, currentFen);
    setCurrentScore(cached?.score ?? null);
    setEvaluatingFen(null);
    if (playedMoves.length === 0 || currentTerminal !== null || cached?.complete) return;

    const controller = new AbortController();
    const updates = createEvaluationPublisher(
      currentFen,
      evaluationCache.current,
      currentFenRef,
      setCurrentScore,
    );
    const timeout = window.setTimeout(() => {
      async function evaluateCurrentPosition() {
        try {
          engine.current ??= new StockfishClient();
          setEvaluatingFen(currentFen);
          const analysis = await engine.current.analyze(currentFen, {
            budgetMs: LIVE_ANALYSIS_MS,
            multiPv: 1,
            requireStable: false,
            signal: controller.signal,
            onUpdate: (lines) => {
              const primary = lines[0];
              if (!controller.signal.aborted && primary) updates.queue(primary.score);
            },
          });
          if (controller.signal.aborted) return;
          const line = analysis.lines[0];
          if (!line) throw new StockfishError("Stockfish did not return a live evaluation.");
          updates.complete(line.score);
          setLiveError(null);
        } catch (cause) {
          if (!isAbortError(cause)) {
            engine.current?.dispose();
            engine.current = null;
            setLiveError({ source: "engine", message: messageFrom(cause) });
          }
        } finally {
          updates.cancel();
          if (analysisGeneration.current === generation) {
            setEvaluatingFen((fen) => fen === currentFen ? null : fen);
          }
        }
      }
      void evaluateCurrentPosition();
    }, 140);

    return () => {
      window.clearTimeout(timeout);
      controller.abort();
      updates.cancel();
      if (analysisGeneration.current === generation) {
        setEvaluatingFen((fen) => fen === currentFen ? null : fen);
      }
    };
  }, [currentFen, currentTerminal, liveRetry, playedMoves.length, reviewState.status]);

  function play(move: AttemptedMove) {
    setLinePreview(null);
    if (firstAttempt === null) {
      setFirstAttempt(move);
      setPendingAttempt(move);
      setPlayedMoves([]);
    } else {
      setPlayedMoves((moves) => [...moves, move]);
    }
    setPinnedCue(null);
  }

  function undo() {
    setPlayedMoves((moves) => moves.slice(0, -1));
    setPinnedCue(null);
  }

  function resetBoard() {
    setPlayedMoves([]);
    setLinePreview(null);
    setFirstAttempt(null);
    setPendingAttempt(null);
    setLiveError(null);
    ratingGeneration.current += 1;
    setRatingState({ reviewId: null, status: "idle" });
    setPinnedCue(null);
    setHoveredCue(null);
    if (initialReview.current) {
      setReviewState({ status: "ready", review: initialReview.current });
    }
    const rootEvaluation = evaluationCache.current.get(position.full_fen);
    evaluationCache.current.clear();
    if (rootEvaluation) {
      evaluationCache.current.set(position.full_fen, rootEvaluation);
    }
    setCurrentScore(rootEvaluation?.score ?? null);
  }

  function showLine(line: ReviewLine) {
    setPlayedMoves(line.moves.map((move) => ({ uci: move.uci, san: move.san })));
    setLinePreview(
      line.role === "best_candidate" || line.role === "alternative_candidate"
        ? "best"
        : "attempt",
    );
    setPinnedCue(null);
    setHoveredCue(null);
  }

  async function rateReview(rating: "helpful" | "unhelpful") {
    if (!review?.review_id || ratingStatus === "sending") return;
    const reviewId = review.review_id;
    const generation = ++ratingGeneration.current;
    setRatingState({ reviewId, status: "sending" });
    try {
      await ratePositionReview(reviewId, {
        rating,
        reason: rating === "helpful" ? "correct" : ratingReason,
      });
      if (ratingGeneration.current === generation) {
        setRatingState({ reviewId, status: "done" });
      }
    } catch {
      if (ratingGeneration.current === generation) {
        setRatingState({ reviewId, status: "error" });
      }
    }
  }

  function pinCue(cue: ReviewAnnotation) {
    if (hasBoardCue(cue) && review) {
      setPlayedMoves(movesForCue(cue, review));
      setLinePreview(
        cue.scope === "best_line"
          ? "best"
          : cue.scope === "attempt_line" || cue.scope === "attempt_refutation"
            ? "attempt"
            : null,
      );
    }
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
        <div
          className={`engine-status is-${liveError ? "error" : reviewState.status}`}
          role="status"
          title={liveError?.message}
        >
          <i aria-hidden="true" />
          <span>{engineStatus(
            reviewState.status,
            evaluatingFen === currentFen,
            liveError,
            rootIsTerminal,
            checkingAttempt,
          )}</span>
        </div>
      </header>

      <section className="position-review__layout">
        <div className="position-review__board-column">
          <div className="board-context">
            <div>
              <span>{turnName(current.turn())} to move</span>
              <small>
                {lastMove
                  ? `${linePreview ? "Hypothetical · after" : "After"} ${lastMove.san}`
                  : "Starting position"}
              </small>
            </div>
            <span className="board-context__topic">
              {revealed && review
                ? review.topic.name
                : reviewState.status === "loading"
                  ? "Finding the idea…"
                  : "Your turn"}
            </span>
          </div>

          <div className="position-review__board-stage">
            <EvalBar
              score={currentScore}
              turn={current.turn()}
              orientation={position.orientation}
              loading={currentTerminal === null && (
                evaluatingFen === currentFen
                || (reviewState.status === "loading" && !rootIsTerminal && !currentScore)
              )}
              terminal={currentTerminal}
            />
            <InteractiveBoard
              fen={position.full_fen}
              orientation={position.orientation}
              moves={moveUcis}
              interactive={
                currentTerminal === null
                && reviewState.status === "ready"
                && !checkingAttempt
                && linePreview === null
              }
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
              {liveError?.source === "engine" && playedMoves.length > 0 && (
                <button
                  type="button"
                  onClick={() => {
                    setLiveError(null);
                    setLiveRetry((value) => value + 1);
                  }}
                >
                  Retry analysis
                </button>
              )}
            </div>
            <span className="board-tools__eval">
              Local eval <b>{evaluationLabel(currentScore, current.turn(), current)}</b>
            </span>
          </div>
        </div>

        <article className="review-notes" aria-live={reviewState.status === "loading" ? "off" : "polite"}>
          {reviewState.status === "loading" && (
            <div className="review-loading">
              <span className="review-loading__mark" aria-hidden="true" />
              <p className="commentary-kicker">{rootIsTerminal ? "Rules check" : "Reading the position"}</p>
              <h1>{rootIsTerminal ? "Confirming the result…" : "Finding the useful clue…"}</h1>
              <p>
                {rootIsTerminal
                  ? "The board is checking why the game has ended."
                  : "The board is live while the local engine checks the first move."}
              </p>
            </div>
          )}

          {reviewState.status === "error" && (
            <>
              <p className="commentary-kicker">Review unavailable</p>
              <h1>Keep the board.</h1>
              <p>{reviewState.message}</p>
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

          {reviewState.status === "ready" && (
            <>
              {revealed && (
                <BoardReference
                  className="topic-reference"
                  cue={reviewState.review.hint}
                  {...cueEvents(reviewState.review.hint)}
                >
                  <span>Position topic</span>
                  <strong>{reviewState.review.topic.name}</strong>
                  <small>{hasBoardCue(reviewState.review.hint) ? "Show on board" : "Position guide"}</small>
                </BoardReference>
              )}

              {reviewState.review.best_move === null ? (
                <div className="review-explanation">
                  <p className="commentary-kicker">{reviewState.review.evaluation}</p>
                  <h1>{reviewState.review.hint.text}</h1>
                  <div className="annotation-list">
                    {reviewState.review.explanation.map((cue, index) => (
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
                        <small>{hasBoardCue(cue) ? "Board" : "Note"}</small>
                      </BoardReference>
                    ))}
                  </div>
                </div>
              ) : !revealed ? (
                <div className="review-prompt">
                  <p className="commentary-kicker">{turnName(root.turn())} to move</p>
                  <h1>What is the first move?</h1>
                  <BoardReference
                    className="hint-reference"
                    cue={reviewState.review.hint}
                    {...cueEvents(reviewState.review.hint)}
                  >
                    <span className="annotation-index">Hint</span>
                    <span className="hint-reference__copy">{reviewState.review.hint.text}</span>
                    {hasBoardCue(reviewState.review.hint) && (
                      <small>Tap to mark the clues on the board</small>
                    )}
                  </BoardReference>
                  <p className="spoiler-note"><i aria-hidden="true" /> No move names until you try.</p>
                </div>
              ) : checkingAttempt ? (
                <div className="review-loading">
                  <span className="review-loading__mark" aria-hidden="true" />
                  <p className="commentary-kicker">Your move</p>
                  <h1>Checking the strongest reply…</h1>
                  <p>The line is hypothetical until the local engine finishes comparing it.</p>
                </div>
              ) : (
                <div className="review-explanation">
                  <p className="commentary-kicker">
                    {reviewState.review.attempt
                      ? `Your line · ${scoreText(reviewState.review.attempt.line.score, root.turn())}`
                      : reviewState.review.evaluation}
                  </p>
                  <h1>{attemptVerdict(firstAttempt, reviewState.review)}</h1>
                  <p className="review-explanation__lead">
                    {reviewState.review.explanation.some((cue) => cue.arrows.length > 0)
                      ? "Follow the red move arrow, then use each note to inspect the idea on the board."
                      : "Use each note to compare your move with the checked engine lines."}
                  </p>
                  <div className="annotation-list">
                    {reviewState.review.explanation.map((cue, index) => (
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
                        <small>{hasBoardCue(cue) ? "Board" : "Note"}</small>
                      </BoardReference>
                    ))}
                  </div>
                  <div className="review-actions" aria-label="Hypothetical engine lines">
                    {reviewState.review.attempt && (
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={() => showLine(reviewState.review.attempt!.line)}
                      >
                        Your line: {reviewState.review.attempt.move.san}
                      </button>
                    )}
                    {reviewState.review.lines[0] && (
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={() => showLine(reviewState.review.lines[0]!)}
                      >
                        Best line: {reviewState.review.lines[0].moves[0]?.san}
                      </button>
                    )}
                  </div>
                  <p className="spoiler-note">
                    <i aria-hidden="true" /> These are engine continuations, not moves already played.
                  </p>
                  {reviewState.review.review_id && (
                    <div className="review-feedback" aria-label="Rate this analysis">
                      {ratingStatus === "done" ? (
                        <span>Feedback saved for review.</span>
                      ) : (
                        <>
                          <button
                            type="button"
                            disabled={ratingStatus === "sending"}
                            onClick={() => void rateReview("helpful")}
                          >
                            Helpful
                          </button>
                          <select
                            aria-label="Problem with this analysis"
                            value={ratingReason}
                            onChange={(event) => setRatingReason(event.target.value as typeof ratingReason)}
                          >
                            <option value="incorrect_chess">Incorrect chess</option>
                            <option value="irrelevant_topic">Irrelevant topic</option>
                            <option value="unclear">Unclear explanation</option>
                            <option value="equivalent_move_rejected">Equivalent move rejected</option>
                            <option value="too_verbose">Too verbose</option>
                            <option value="missing_detail">Missing useful detail</option>
                            <option value="other">Other</option>
                          </select>
                          <button
                            type="button"
                            disabled={ratingStatus === "sending"}
                            onClick={() => void rateReview("unhelpful")}
                          >
                            Report issue
                          </button>
                          {ratingStatus === "error" && <span>Feedback could not be saved.</span>}
                        </>
                      )}
                    </div>
                  )}
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
  if (!hasBoardCue(cue)) {
    return <div className={`${className} is-static`}>{children}</div>;
  }
  return (
    <button
      type="button"
      className={`${className}${active ? " is-active" : ""}`}
      aria-pressed={active}
      onPointerEnter={onEnter}
      onPointerLeave={onLeave}
      onFocus={onEnter}
      onBlur={onLeave}
      onClick={onToggle}
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

function createEvaluationPublisher(
  fen: string,
  cache: Map<string, CachedEvaluation>,
  currentFen: React.RefObject<string>,
  setCurrentScore: React.Dispatch<React.SetStateAction<EngineScore | null>>,
): EvaluationPublisher {
  let pending: EngineScore | null = null;
  let timer: number | null = null;

  function publish(score: EngineScore, complete: boolean) {
    storeEvaluation(cache, fen, { score, complete });
    if (currentFen.current === fen) setCurrentScore(score);
  }

  function flush() {
    timer = null;
    if (!pending) return;
    const score = pending;
    pending = null;
    publish(score, false);
  }

  return {
    queue(score) {
      pending = score;
      timer ??= window.setTimeout(flush, EVALUATION_UPDATE_MS);
    },
    complete(score) {
      if (timer !== null) window.clearTimeout(timer);
      timer = null;
      pending = null;
      publish(score, true);
    },
    cancel() {
      if (timer !== null) window.clearTimeout(timer);
      timer = null;
      pending = null;
    },
  };
}

function readEvaluation(
  cache: Map<string, CachedEvaluation>,
  fen: string,
): CachedEvaluation | undefined {
  const value = cache.get(fen);
  if (!value) return undefined;
  cache.delete(fen);
  cache.set(fen, value);
  return value;
}

function storeEvaluation(
  cache: Map<string, CachedEvaluation>,
  fen: string,
  value: CachedEvaluation,
) {
  cache.delete(fen);
  cache.set(fen, value);
  while (cache.size > EVALUATION_CACHE_LIMIT) {
    const oldest = cache.keys().next().value;
    if (oldest === undefined) break;
    cache.delete(oldest);
  }
}

function reviewAnalysis(
  lines: EngineLine[],
  attempt?: AttemptedMove,
  attemptLine?: EngineLine,
): ReviewAnalysis {
  const candidates = lines.map(reviewEngineLine);
  return {
    score_pov: "side_to_move",
    lines: candidates,
    ...(attempt && attemptLine
      ? {
          attempt: {
            move: attempt.uci,
            line: reviewEngineLine({ ...attemptLine, rank: 1 }),
          },
        }
      : {}),
  };
}

function reviewEngineLine(line: EngineLine) {
  if (!line.wdl) throw new Error("Stockfish did not return WDL evidence.");
  return {
    rank: line.rank,
    depth: line.depth,
    score: line.score,
    wdl: line.wdl,
    pv: line.pv.slice(0, 16),
    stable: line.stable === true,
  };
}

function hasBoardCue(cue: ReviewAnnotation): boolean {
  return cue.squares.length > 0 || cue.arrows.length > 0;
}

function movesForCue(
  cue: ReviewAnnotation,
  review: PositionReviewData,
): AttemptedMove[] {
  let line: ReviewLine | undefined;
  if (cue.scope === "best_line") {
    line = review.lines.find((candidate) => candidate.rank === 1);
  } else if (cue.scope === "attempt_line" || cue.scope === "attempt_refutation") {
    line = review.attempt?.line;
  }
  if (!line) return [];
  return line.moves
    .slice(0, cue.ply)
    .map((move) => ({ uci: move.uci, san: move.san }));
}

function attemptVerdict(
  attempt: AttemptedMove | null,
  review: PositionReviewData,
): string {
  if (!review.best_move) return review.evaluation;
  if (review.attempt?.verdict === "best") return `${review.attempt.move.san} is the best move.`;
  if (review.attempt?.equivalent) {
    return `${review.attempt.move.san} is effectively as strong as the first engine choice.`;
  }
  if (review.attempt?.verdict === "blunder") {
    return `${review.attempt.move.san} is a blunder; compare the strongest reply with ${review.best_move.san}.`;
  }
  if (review.attempt?.lost_forced_mate) {
    return `${review.attempt.move.san} stays favorable but gives up the forced mate.`;
  }
  if (review.attempt) {
    const comparison = review.attempt.verdict === "mistake"
      ? "strongest reply"
      : "checked continuation";
    return `${review.attempt.move.san} is a ${review.attempt.verdict}; compare its ${comparison} with ${review.best_move.san}.`;
  }
  if (attempt) return `The comparison for ${attempt.san} was unavailable.`;
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

function engineStatus(
  status: ReviewState["status"],
  evaluatingCurrent: boolean,
  liveError: AnalysisError | null,
  terminal: boolean,
  checkingAttempt: boolean,
): string {
  if (terminal) {
    if (status === "error") return "Rules review unavailable";
    if (status === "loading") return "Checking the result from chess rules";
    return "Rules result ready";
  }
  if (status === "error" || liveError?.source === "engine") {
    return "Local analysis unavailable";
  }
  if (liveError?.source === "api") return "Move review unavailable";
  if (checkingAttempt) return "Comparing your move locally";
  if (status === "loading" || evaluatingCurrent) return "Stockfish is checking locally";
  return "Local analysis ready";
}

function messageFrom(cause: unknown): string {
  return cause instanceof Error ? cause.message : "The position could not be reviewed.";
}

function isAbortError(cause: unknown): boolean {
  return cause instanceof DOMException && cause.name === "AbortError";
}
