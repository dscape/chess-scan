import { useEffect, useMemo, useRef, useState } from "react";
import { Chess } from "chess.js";
import { createPositionReview } from "../api";
import StockfishClient, { type AnalysisResult } from "../engine/StockfishClient";
import type { EngineLine, EngineScore } from "../engine/uci";
import type {
  PositionReview,
  ReviewMode,
  ReviewedPosition,
} from "../types";
import InteractiveBoard, { type AttemptedMove } from "./InteractiveBoard";

type PositionLessonProps = {
  position: ReviewedPosition;
  onScanAnother: () => void;
};

type LessonPhase = "attempt" | "analyzing" | "review" | "error";

export default function PositionLesson({
  position,
  onScanAnother,
}: PositionLessonProps) {
  const [studyLevel, setStudyLevel] = useState(() => storedLevel());
  const [mode, setMode] = useState<ReviewMode>("general");
  const [phase, setPhase] = useState<LessonPhase>("attempt");
  const [attempt, setAttempt] = useState<AttemptedMove | null>(null);
  const [progress, setProgress] = useState<EngineLine[]>([]);
  const [review, setReview] = useState<PositionReview | null>(null);
  const [activeLine, setActiveLine] = useState(0);
  const [ply, setPly] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const engine = useRef<StockfishClient | null>(null);
  const request = useRef<AbortController | null>(null);
  const analysisCache = useRef(new Map<string, AnalysisResult>());
  const reviewCache = useRef(new Map<string, PositionReview>());
  const root = useMemo(() => new Chess(position.full_fen), [position.full_fen]);
  const sideName = root.turn() === "w" ? "White" : "Black";
  const line = review?.lines[activeLine] ?? null;
  const lineMoves = line?.moves.map((move) => move.uci) ?? [];
  const boardMoves = phase === "analyzing" && attempt ? [attempt.uci] : lineMoves;
  const boardPly = phase === "analyzing" && attempt ? 1 : phase === "review" ? ply : 0;
  const highlights = review?.primary_finding?.evidence.flatMap((item) => item.squares) ?? [];

  useEffect(() => {
    try {
      window.localStorage.setItem("chess-scan:study-level", String(studyLevel));
    } catch {
      // The lesson remains usable when browser storage is unavailable.
    }
  }, [studyLevel]);

  useEffect(() => () => {
    request.current?.abort();
    engine.current?.dispose();
  }, []);

  async function analyze(move?: AttemptedMove) {
    const chosenMove = move ?? attempt;
    setAttempt(chosenMove ?? null);
    setPhase("analyzing");
    setError(null);
    setProgress([]);
    setReview(null);
    setPly(0);

    const controller = new AbortController();
    request.current?.abort();
    request.current = controller;
    const reviewKey = lessonReviewKey(position.full_fen, studyLevel, mode);
    const cachedReview = reviewCache.current.get(reviewKey);
    if (cachedReview) {
      finishReview(cachedReview);
      return;
    }

    if (root.isGameOver()) {
      try {
        const result = await createPositionReview({
          fen: position.full_fen,
          study_level: studyLevel,
          mode,
          lines: [],
        }, controller.signal);
        if (controller.signal.aborted) return;
        reviewCache.current.set(reviewKey, result);
        finishReview(result);
      } catch (cause) {
        if (!isAbortError(cause)) showReviewError(cause);
      }
      return;
    }

    const budgetMs = analysisBudget(studyLevel);
    const analysisKey = engineAnalysisKey(position.full_fen, budgetMs);
    let analysis: AnalysisResult;
    try {
      const cachedAnalysis = analysisCache.current.get(analysisKey);
      if (cachedAnalysis) {
        analysis = cachedAnalysis;
        setProgress(analysis.lines);
      } else {
        engine.current ??= new StockfishClient();
        let lastProgressUpdate = 0;
        analysis = await engine.current.analyze(position.full_fen, {
          budgetMs,
          multiPv: 3,
          signal: controller.signal,
          onUpdate: (lines) => {
            const now = performance.now();
            if (now - lastProgressUpdate >= 200) {
              lastProgressUpdate = now;
              setProgress(lines);
            }
          },
        });
        if (controller.signal.aborted) return;
        analysisCache.current.set(analysisKey, analysis);
      }
    } catch (cause) {
      if (isAbortError(cause)) return;
      engine.current?.dispose();
      engine.current = null;
      showReviewError(cause);
      return;
    }

    try {
      const result = await createPositionReview({
        fen: position.full_fen,
        study_level: studyLevel,
        mode,
        lines: analysis.lines,
      }, controller.signal);
      if (controller.signal.aborted) return;
      reviewCache.current.set(reviewKey, result);
      finishReview(result);
    } catch (cause) {
      if (!isAbortError(cause)) showReviewError(cause);
    }
  }

  function finishReview(result: PositionReview) {
    setReview(result);
    setActiveLine(0);
    setPly(0);
    setPhase("review");
  }

  function showReviewError(cause: unknown) {
    setError(cause instanceof Error ? cause.message : "The position could not be reviewed.");
    setPhase("error");
  }

  function resetAttempt() {
    request.current?.abort();
    void engine.current?.stop();
    setPhase("attempt");
    setAttempt(null);
    setProgress([]);
    setReview(null);
    setPly(0);
    setError(null);
  }

  function chooseLine(index: number) {
    setActiveLine(index);
    setPly(0);
  }

  const attemptVerdict = review?.best_move
    ? attempt?.uci === review.best_move.uci
      ? "You found Stockfish’s first choice."
      : attempt
        ? `You chose ${attempt.san}; Stockfish prefers ${review.best_move.san}.`
        : `Stockfish prefers ${review.best_move.san}.`
    : root.moves().length === 0
      ? "There is no legal move in this position."
      : review?.evaluation ?? "The game has ended.";

  return (
    <main className="lesson-shell">
      <header className="lesson-header">
        <div>
          <p className="eyebrow">Position lesson</p>
          <h1>{phase === "review" ? "Review the idea." : "What would you play?"}</h1>
        </div>
        <span className="local-engine-badge"><i /> Stockfish runs on this device</span>
      </header>

      <section className="lesson-layout">
        <div className="lesson-board-column">
          <InteractiveBoard
            fen={position.full_fen}
            orientation={position.orientation}
            moves={boardMoves}
            ply={boardPly}
            interactive={phase === "attempt" && !root.isGameOver()}
            highlights={phase === "review" ? highlights : []}
            onMove={(move) => void analyze(move)}
          />
          {phase === "review" && line && (
            <div className="variation-controls">
              <button type="button" aria-label="First move" onClick={() => setPly(0)} disabled={ply === 0}>↤</button>
              <button type="button" aria-label="Previous move" onClick={() => setPly((value) => Math.max(0, value - 1))} disabled={ply === 0}>←</button>
              <span>{ply === 0 ? "Start" : line.moves[ply - 1]?.san} · {ply}/{line.moves.length}</span>
              <button type="button" aria-label="Next move" onClick={() => setPly((value) => Math.min(line.moves.length, value + 1))} disabled={ply === line.moves.length}>→</button>
              <button type="button" aria-label="Last move" onClick={() => setPly(line.moves.length)} disabled={ply === line.moves.length}>↦</button>
            </div>
          )}
        </div>

        <article
          className="lesson-commentary"
          aria-live={phase === "analyzing" ? "off" : "polite"}
        >
          {phase === "attempt" && (
            <>
              <span className="commentary-kicker">Orientation</span>
              <h2>{sideName} to move</h2>
              <p>{orientationPrompt(mode, studyLevel)}</p>
              <div className="lesson-settings">
                <label>
                  Study level
                  <select value={studyLevel} onChange={(event) => setStudyLevel(Number(event.target.value))}>
                    {[1, 2, 3, 4, 5, 6].map((level) => <option key={level} value={level}>Level {level}</option>)}
                  </select>
                </label>
                <label>
                  Practice mode
                  <select value={mode} onChange={(event) => setMode(event.target.value as ReviewMode)}>
                    <option value="general">Guided</option>
                    <option value="mix">Mix — topic hidden</option>
                    <option value="thinking_ahead">Thinking Ahead</option>
                  </select>
                </label>
              </div>
              <p className="lesson-hint">{searchHint(studyLevel)}</p>
              <button type="button" className="primary-button" onClick={() => void analyze()}>
                {root.isGameOver() ? "Explain the result" : "Show the lesson"}
              </button>
            </>
          )}

          {phase === "analyzing" && (
            <div className="analysis-progress">
              <span className="engine-spinner" aria-hidden="true" />
              <span className="commentary-kicker">
                {root.isGameOver() ? "Rules check" : "Local Stockfish"}
              </span>
              <h2>{root.isGameOver() ? "Confirming the result…" : "Checking the position…"}</h2>
              {attempt && <p>You chose {attempt.san}. Now checking the best reply.</p>}
              <p>{progressSummary(progress)}</p>
              {progress.length > 0 && (
                <div className="analysis-lines-preview">
                  {progress.slice(0, 3).map((candidate) => (
                    <span key={candidate.multipv}>#{candidate.multipv} · {scoreLabel(candidate)}</span>
                  ))}
                </div>
              )}
            </div>
          )}

          {phase === "error" && (
            <>
              <span className="commentary-kicker">Analysis unavailable</span>
              <h2>Keep the position.</h2>
              <p>{error}</p>
              <button type="button" className="primary-button" onClick={() => void analyze()}>Try again</button>
              <a className="secondary-button" href={position.lichess_url} target="_blank" rel="noreferrer">Open in Lichess ↗</a>
            </>
          )}

          {phase === "review" && review && (
            <>
              <span className="commentary-kicker">{review.evaluation}</span>
              <h2>{attemptVerdict}</h2>
              <p className="lesson-explanation">{review.explanation}</p>
              {line?.wdl && <WdlMeter values={line.wdl} />}
              {review.primary_finding && (
                <div className="subject-card">
                  <span>Study subject · Level {review.primary_finding.level}</span>
                  <strong>{review.primary_finding.topic}</strong>
                  {review.primary_finding.evidence.map((evidence) => (
                    <p key={`${evidence.kind}:${evidence.summary}`}>{evidence.summary}</p>
                  ))}
                </div>
              )}
              {review.lines.length > 1 && (
                <div className="candidate-lines" aria-label="Candidate moves">
                  {review.lines.map((candidate, index) => (
                    <button key={candidate.multipv} type="button" className={index === activeLine ? "is-active" : ""} onClick={() => chooseLine(index)}>
                      <b>{candidate.moves[0]?.san ?? "—"}</b>
                      <span>{scoreLabel(candidate)}</span>
                    </button>
                  ))}
                </div>
              )}
              <div className="lesson-next-actions">
                <button type="button" className="primary-button" onClick={resetAttempt}>
                  {review.best_move ? "Try this position again" : "Review the result again"}
                </button>
                <button type="button" className="secondary-button" onClick={onScanAnother}>Scan another position</button>
              </div>
            </>
          )}
        </article>
      </section>

      <footer className="lesson-footer">
        <span>
          <a href="/stockfish/Copying.txt" target="_blank" rel="noreferrer">Stockfish 18 lite · GPLv3</a>
          {" · "}
          <a href="/stockfish/SOURCE.md" target="_blank" rel="noreferrer">Source</a>
        </span>
        <a href={position.lichess_url} target="_blank" rel="noreferrer">Open advanced analysis in Lichess ↗</a>
      </footer>
    </main>
  );
}

function storedLevel(): number {
  try {
    const level = Number(window.localStorage.getItem("chess-scan:study-level") ?? 2);
    return Number.isInteger(level) && level >= 1 && level <= 6 ? level : 2;
  } catch {
    return 2;
  }
}

function analysisBudget(level: number): number {
  return 1100 + level * 200;
}

function engineAnalysisKey(fen: string, budgetMs: number): string {
  return JSON.stringify([fen, budgetMs, 3]);
}

function lessonReviewKey(fen: string, level: number, mode: ReviewMode): string {
  return JSON.stringify([fen, level, mode]);
}

function isAbortError(cause: unknown): boolean {
  return cause instanceof DOMException && cause.name === "AbortError";
}

function orientationPrompt(mode: ReviewMode, level: number): string {
  if (mode === "mix") return "The subject stays hidden until you commit to a move. Find the most important feature yourself.";
  if (mode === "thinking_ahead") return "Do not move the pieces yet. Calculate your move, the best reply, and your continuation.";
  if (level === 1) return "Check whether you can deliver mate, win material, or save a piece in danger.";
  if (level === 2) return "Find the possible targets: the king, loose material, and important squares.";
  if (level === 3) return "Identify what is characteristic about the position and what the opponent threatens.";
  return "Work out what is going on, which targets are vulnerable, and whether a preparatory move is needed.";
}

function searchHint(level: number): string {
  if (level === 1) return "Checks, captures, danger—then verify one legal reply.";
  if (level === 2) return "Checks, captures, threats—then your opponent’s best reply.";
  if (level === 3) return "Candidate moves, the best reply, and your continuation.";
  return "Compare forcing moves, positional targets, and preparatory moves before calculating.";
}

function progressSummary(lines: EngineLine[]): string {
  const depth = Math.max(0, ...lines.map((line) => line.depth));
  return depth > 0 ? `Depth ${depth}. Comparing ${lines.length || 1} candidate move${lines.length === 1 ? "" : "s"}.` : "Loading the engine and building candidate moves.";
}

function WdlMeter({ values }: { values: [number, number, number] }) {
  const [win, draw, loss] = values;
  return (
    <div className="wdl-meter" aria-label={`Win ${win / 10}%, draw ${draw / 10}%, loss ${loss / 10}%`}>
      <div aria-hidden="true">
        <i style={{ width: `${win / 10}%` }} />
        <i style={{ width: `${draw / 10}%` }} />
        <i style={{ width: `${loss / 10}%` }} />
      </div>
      <span>Win {Math.round(win / 10)}%</span>
      <span>Draw {Math.round(draw / 10)}%</span>
      <span>Loss {Math.round(loss / 10)}%</span>
    </div>
  );
}

function scoreLabel(line: { score: EngineScore }): string {
  if (line.score.kind === "mate") return line.score.value > 0 ? `Mate in ${line.score.value}` : `Mated in ${Math.abs(line.score.value)}`;
  const pawns = line.score.value / 100;
  return `${pawns >= 0 ? "+" : ""}${pawns.toFixed(1)}`;
}
