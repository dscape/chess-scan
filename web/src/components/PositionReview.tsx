import { useEffect, useMemo, useRef, useState } from "react";
import { Chess } from "chess.js";
import {
  apiFailureFrom,
  comparePositionAttempt,
  createPositionCoaching,
  createPositionReview,
  ratePositionReview,
} from "../api";
import { parentFensForMoves, positionAt } from "../board";
import {
  coachingMoveColor,
  coachingMoveCue,
  coachingMoveDisplay,
  coachingMoveLabel,
  hasCoachingNarrative,
} from "../coaching";
import { readLru, storeLru } from "../lru";
import {
  cueAccessibleLabel,
  cueRoleMark,
  displayedBoardCue,
  displayCueLabel,
  hasBoardCue,
} from "../reviewCue";
import StockfishClient, { StockfishError } from "../engine/StockfishClient";
import {
  oppositeScorePov,
  type EngineLine,
  type EngineScore,
} from "../engine/uci";
import {
  attemptLineFromChild,
  canUseCachedBestGrade,
  classifyStudyMove,
  isStudyBestMove,
  prepareStudyRetry,
  rootEvaluationScore,
  sanForEngineMove,
  settledStudyState,
  studyBoardCue as buildStudyBoardCue,
  studyPathKey,
  studyPositionKey,
  studyStateForKey,
  terminalAttemptLine,
  terminalStudyPosition,
  StudyAnalysisError,
  type StudyGradeResult,
  type StudyPosition,
  type StudyState,
} from "../studyAnalysis";
import type {
  CoachingMoveSegment,
  CoachingPresentationStatus,
  Orientation,
  PositionAttemptRequest,
  PositionCoaching,
  PositionReview as PositionReviewData,
  ReviewAnalysis,
  ReviewAnnotation,
  ReviewBadge,
  ReviewedPosition,
  ReviewLine,
  ReviewMove,
} from "../types";
import ChessPieceAttribution from "./ChessPieceAttribution";
import InteractiveBoard, { type AttemptedMove } from "./InteractiveBoard";
import ReviewGlyph from "./ReviewGlyph";
import StudyPanel from "./StudyPanel";

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

type CoachingState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "unavailable" }
  | { status: "ready"; coaching: PositionCoaching };

type RatingState = {
  reviewId: string | null;
  status: "idle" | "sending" | "done" | "error";
};

type AnalysisError = {
  source: "engine" | "api";
  message: string;
  retryable?: boolean;
};

type LinePreview = "best" | "attempt" | null;

type CoachingPreviewSource = "pointer" | "focus";

type ActiveCoachingPreview = {
  move: CoachingMoveSegment;
  line: ReviewLine;
  order: number;
};

type CoachingPreview = {
  playedMoves: AttemptedMove[];
  linePreview: LinePreview;
  pinnedCue: ReviewAnnotation | null;
  hoveredCue: ReviewAnnotation | null;
  liveError: AnalysisError | null;
  active: Map<string, ActiveCoachingPreview>;
  nextOrder: number;
  restoreFrame: number | null;
  suspended: boolean;
};

const INITIAL_ANALYSIS_MS = 2500;
const ATTEMPT_ANALYSIS_MS = 1600;
const LIVE_ANALYSIS_MS = 750;
const STUDY_ANALYSIS_MS = 1400;
const EVALUATION_UPDATE_MS = 150;
const COACHING_PREVIEW_ANALYSIS_DELAY_MS = 450;
const EVALUATION_CACHE_LIMIT = 48;
const NO_STUDY_GRADE: StudyGradeResult = { grade: null, error: null };

export default function PositionReview({
  position,
  onScanAnother,
}: PositionReviewProps) {
  const [reviewState, setReviewState] = useState<ReviewState>({
    status: "loading",
  });
  const [coachingState, setCoachingState] = useState<CoachingState>({
    status: "idle",
  });
  const [retry, setRetry] = useState(0);
  const [liveRetry, setLiveRetry] = useState(0);
  const [liveError, setLiveError] = useState<AnalysisError | null>(null);
  const [ratingState, setRatingState] = useState<RatingState>({
    reviewId: null,
    status: "idle",
  });
  const [ratingReason, setRatingReason] = useState<
    | "incorrect_chess"
    | "irrelevant_topic"
    | "unclear"
    | "equivalent_move_rejected"
    | "too_verbose"
    | "missing_detail"
    | "other"
  >("incorrect_chess");
  const [playedMoves, setPlayedMoves] = useState<AttemptedMove[]>([]);
  const [linePreview, setLinePreview] = useState<LinePreview>(null);
  const [firstAttempt, setFirstAttempt] = useState<AttemptedMove | null>(null);
  const [hintRevealed, setHintRevealed] = useState(false);
  const [studyMode, setStudyMode] = useState(false);
  const [studyState, setStudyState] = useState<StudyState>({ status: "idle" });
  const [studyRetry, setStudyRetry] = useState(0);
  const [pendingAttempt, setPendingAttempt] = useState<AttemptedMove | null>(
    null,
  );
  const checkingAttempt = pendingAttempt !== null;
  const [currentScore, setCurrentScore] = useState<EngineScore | null>(null);
  const [evaluatingFen, setEvaluatingFen] = useState<string | null>(null);
  const [hoveredCue, setHoveredCue] = useState<ReviewAnnotation | null>(null);
  const [automaticCue, setAutomaticCue] = useState<ReviewAnnotation | null>(null);
  const [pinnedCue, setPinnedCue] = useState<ReviewAnnotation | null>(null);
  const engine = useRef<StockfishClient | null>(null);
  const boardColumn = useRef<HTMLDivElement | null>(null);
  const coachingPreview = useRef<CoachingPreview | null>(null);
  const coachingMoveCache = useRef(new Map<string, AttemptedMove[]>());
  const reviewLines = useRef(new Map<string, EngineLine[]>());
  const initialReview = useRef<PositionReviewData | null>(null);
  const evaluationCache = useRef(new Map<string, CachedEvaluation>());
  const studyPositionCache = useRef(new Map<string, StudyPosition>());
  const studyGradeCache = useRef(new Map<string, StudyGradeResult>());
  const analysisGeneration = useRef(0);
  const ratingGeneration = useRef(0);
  const boardInteractionGeneration = useRef(0);
  const coachingPresentationGeneration = useRef<number | null>(null);

  const root = useMemo(() => new Chess(position.full_fen), [position.full_fen]);
  const rootFen = root.fen();
  const moveUcis = useMemo(
    () => playedMoves.map((move) => move.uci),
    [playedMoves],
  );
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
  const coaching = coachingState.status === "ready" ? coachingState.coaching : null;
  const explanation = useMemo(
    () => coachingExplanationOrder(review?.explanation ?? [], coaching),
    [coaching, review?.explanation],
  );
  const ratingStatus =
    ratingState.reviewId === review?.review_id ? ratingState.status : "idle";
  const defaultCue = revealed
    ? (explanation.find(hasBoardCue) ?? null)
    : null;
  const reviewBoardCue = displayedBoardCue(
    pinnedCue,
    hoveredCue,
    automaticCue,
    defaultCue,
    playedMoves.length,
  );
  const currentStudyPositionKey = useMemo(
    () => studyPositionKey(currentFen),
    [currentFen],
  );
  const currentStudyKey = useMemo(
    () => studyPathKey(currentStudyPositionKey, moveUcis),
    [currentStudyPositionKey, moveUcis],
  );
  const currentStudyState = studyStateForKey(studyState, currentStudyKey);
  const currentStudyNode = currentStudyState.status === "ready"
    || currentStudyState.status === "review-error"
    ? currentStudyState.node
    : null;
  const studyTopLine = currentStudyNode?.lines[0]
    ?? (currentStudyState.status === "loading"
      ? currentStudyState.topLine
      : null);
  const studyTopMove = studyTopLine?.pv[0] ?? null;
  const studyReview = currentStudyNode?.review ?? null;
  const studyCue = useMemo(
    () => buildStudyBoardCue(studyReview),
    [studyReview],
  );
  const studyGrade = currentStudyNode?.grade ?? null;
  const boardCue = studyMode ? studyCue : reviewBoardCue;
  const lastMove = playedMoves.at(-1) ?? null;

  useEffect(
    () => () => {
      const activeEngine = engine.current;
      engine.current = null;
      activeEngine?.dispose();
      discardCoachingMovePreview();
    },
    [],
  );

  useEffect(() => {
    const controller = new AbortController();
    const generation = ++analysisGeneration.current;
    let active = true;
    let updates: EvaluationPublisher | null = null;

    setReviewState({ status: "loading" });
    setCoachingState({ status: "idle" });
    coachingPresentationGeneration.current = null;
    discardCoachingMovePreview();
    coachingMoveCache.current.clear();
    setPinnedCue(null);
    setAutomaticCue(null);
    setHoveredCue(null);
    setLiveError(null);
    ratingGeneration.current += 1;
    setRatingState({ reviewId: null, status: "idle" });
    setLinePreview(null);
    setFirstAttempt(null);
    setPendingAttempt(null);
    setHintRevealed(false);
    setStudyMode(false);
    setStudyState({ status: "idle" });
    studyPositionCache.current.clear();
    studyGradeCache.current.clear();

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
          setEvaluatingFen(rootFen);
          updates = createEvaluationPublisher(
            rootFen,
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
              const primary = candidates.find(
                (candidate) => candidate.rank === 1,
              );
              if (active && primary) updates?.queue(primary.score);
            },
          });
          if (!active) return;
          lines = analysis.lines;
          const primary = lines.find((line) => line.rank === 1);
          if (!primary)
            throw new Error("Stockfish did not return a principal line.");
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
          clearEvaluatingFen(generation, rootFen);
        }
      } else {
        const primary = lines.find((line) => line.rank === 1);
        if (!primary) throw new Error("Cached analysis has no principal line.");
        storeLru(evaluationCache.current, rootFen, {
          score: primary.score,
          complete: true,
        }, EVALUATION_CACHE_LIMIT);
        if (currentFenRef.current === rootFen) setCurrentScore(primary.score);
        clearEvaluatingFen(generation, rootFen);
      }

      cacheCandidateEvaluations(
        evaluationCache.current,
        position.full_fen,
        lines,
      );

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
      setEvaluatingFen((currentValue) =>
        currentValue === fen ? null : currentValue,
      );
    }

    void loadReview();
    return () => {
      active = false;
      controller.abort();
      updates?.cancel();
      clearEvaluatingFen(generation, rootFen);
    };
  }, [position.feedback_id, position.full_fen, retry, rootFen, rootIsTerminal]);

  useEffect(() => {
    if (!pendingAttempt) return;
    const attempt = pendingAttempt;
    const interactionGeneration = boardInteractionGeneration.current;
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
          if (!line)
            throw new StockfishError("Stockfish could not evaluate your move.");
          attemptLine = line;
          cacheAttemptEvaluation(
            evaluationCache.current,
            position.full_fen,
            attempt.uci,
            attemptLine.score,
          );
        } catch (cause) {
          if (active && !isAbortError(cause)) {
            if (cause instanceof StockfishError) {
              engine.current?.dispose();
              engine.current = null;
            }
            setLiveError({ source: "engine", message: messageFrom(cause) });
            if (initialReview.current) {
              setReviewState({
                status: "ready",
                review: initialReview.current,
              });
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
          if (active) {
            setReviewState({ status: "ready", review: result });
            const teachingDiagram = initialTeachingDiagram(result.explanation);
            if (boardInteractionGeneration.current !== interactionGeneration) {
              setPinnedCue(null);
              setAutomaticCue(null);
            } else if (teachingDiagram) {
              selectAutomaticCue(teachingDiagram, result);
            } else {
              setPlayedMoves([]);
              setPinnedCue(null);
              setAutomaticCue(null);
              setLinePreview(null);
            }
          }
        } catch (cause) {
          if (active && !isAbortError(cause)) {
            setLiveError({ source: "api", message: messageFrom(cause) });
            if (initialReview.current) {
              setReviewState({
                status: "ready",
                review: initialReview.current,
              });
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
    const reviewId = review?.review_id;
    if (
      !position.coaching_available ||
      !firstAttempt ||
      !review?.attempt ||
      !reviewId
    ) {
      setCoachingState({ status: "idle" });
      coachingPresentationGeneration.current = null;
      return;
    }

    const controller = new AbortController();
    coachingPresentationGeneration.current = boardInteractionGeneration.current;
    let active = true;
    setCoachingState({ status: "loading" });

    async function loadCoaching() {
      try {
        const result = await createPositionCoaching(reviewId!, controller.signal);
        if (active) setCoachingState({ status: "ready", coaching: result });
      } catch (cause) {
        if (active && !isAbortError(cause)) {
          setCoachingState({ status: "unavailable" });
        }
      }
    }

    void loadCoaching();
    return () => {
      active = false;
      controller.abort();
    };
  }, [
    firstAttempt,
    position.coaching_available,
    review?.attempt,
    review?.review_id,
  ]);

  useEffect(() => {
    if (
      coachingState.status !== "ready" ||
      coachingState.coaching.status !== "accepted" ||
      !review ||
      pinnedCue !== null ||
      hoveredCue !== null ||
      coachingPresentationGeneration.current !== boardInteractionGeneration.current
    )
      return;
    const teachingDiagram = initialTeachingDiagram(explanation);
    if (teachingDiagram) selectAutomaticCue(teachingDiagram, review);
  }, [coachingState, explanation, review, pinnedCue, hoveredCue]);

  useEffect(() => {
    if (reviewState.status !== "ready" || studyMode) return;

    const generation = ++analysisGeneration.current;
    const cached = readLru(evaluationCache.current, currentFen);
    setCurrentScore(cached?.score ?? null);
    setEvaluatingFen(null);
    if (
      playedMoves.length === 0 ||
      currentTerminal !== null ||
      cached?.complete
    )
      return;

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
              if (!controller.signal.aborted && primary)
                updates.queue(primary.score);
            },
          });
          if (controller.signal.aborted) return;
          const line = analysis.lines[0];
          if (!line)
            throw new StockfishError(
              "Stockfish did not return a live evaluation.",
            );
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
            setEvaluatingFen((fen) => (fen === currentFen ? null : fen));
          }
        }
      }
      void evaluateCurrentPosition();
    }, coachingPreview.current ? COACHING_PREVIEW_ANALYSIS_DELAY_MS : 140);

    return () => {
      window.clearTimeout(timeout);
      controller.abort();
      updates.cancel();
      if (analysisGeneration.current === generation) {
        setEvaluatingFen((fen) => (fen === currentFen ? null : fen));
      }
    };
  }, [
    currentFen,
    currentTerminal,
    liveRetry,
    playedMoves.length,
    reviewState.status,
    studyMode,
  ]);

  useEffect(() => {
    if (!studyMode || reviewState.status !== "ready") return;

    const attemptedMove = playedMoves.at(-1) ?? null;
    const parentFen = attemptedMove?.parentFen ?? null;
    const parentPosition = parentFen
      ? readLru(studyPositionCache.current, studyPositionKey(parentFen)) ?? null
      : null;
    let studyPosition = currentTerminal === null
      ? readLru(studyPositionCache.current, currentStudyPositionKey) ?? null
      : terminalStudyPosition();
    let gradeResult = attemptedMove
      ? readLru(studyGradeCache.current, currentStudyKey) ?? null
      : NO_STUDY_GRADE;

    if (
      attemptedMove
      && gradeResult === null
      && canUseCachedBestGrade(currentTerminal)
      && parentPosition?.review
      && isStudyBestMove(attemptedMove.uci, parentPosition.review)
    ) {
      gradeResult = {
        grade: classifyStudyMove(attemptedMove, parentPosition.review, null),
        error: null,
      };
      storeLru(
        studyGradeCache.current,
        currentStudyKey,
        gradeResult,
        EVALUATION_CACHE_LIMIT,
      );
    }

    if (
      studyPosition
      && gradeResult
      && (
        studyPosition.terminal
        || studyPosition.review !== null
        || studyPosition.reviewError !== null
      )
    ) {
      publishStudyState(studyPosition, gradeResult);
      return;
    }

    const generation = ++analysisGeneration.current;
    const controller = new AbortController();
    let active = true;
    let updates: EvaluationPublisher | null = null;
    let lastPrimaryUpdate: string | null = null;
    let publishedTopMove = studyPosition?.lines[0]?.pv[0] ?? null;
    let publishedTopMoveSan = studyPosition?.topMoveSan ?? null;

    setLiveError(null);
    setStudyState({
      status: "loading",
      key: currentStudyKey,
      topLine: studyPosition?.lines[0] ?? null,
      topMoveSan: studyPosition?.topMoveSan ?? null,
    });

    function publishStudyState(
      settledPosition: StudyPosition,
      settledGrade: StudyGradeResult,
    ) {
      const state = settledStudyState(
        currentStudyKey,
        settledPosition,
        settledGrade,
      );
      setStudyState(state);
      const failure = state.status === "review-error"
        ? state.failure
        : state.node.gradeError;
      setLiveError(failure
        ? {
            source: "api",
            message: failure.message,
            retryable: failure.retryable,
          }
        : null);
      const primary = settledPosition.lines[0];
      setCurrentScore(settledPosition.terminal ? null : (primary?.score ?? null));
      setEvaluatingFen(null);
    }

    async function ensureStudyPosition() {
      if (studyPosition) {
        const primary = studyPosition.lines[0];
        setCurrentScore(studyPosition.terminal ? null : (primary?.score ?? null));
        setEvaluatingFen(null);
        return;
      }

      const cachedEvaluation = readLru(
        evaluationCache.current,
        currentStudyPositionKey,
      );
      setCurrentScore(cachedEvaluation?.score ?? null);
      setEvaluatingFen(currentStudyPositionKey);
      updates = createEvaluationPublisher(
        currentStudyPositionKey,
        evaluationCache.current,
        currentFenRef,
        setCurrentScore,
      );
      engine.current ??= new StockfishClient();
      const analysis = await engine.current.analyze(currentStudyPositionKey, {
        budgetMs: STUDY_ANALYSIS_MS,
        multiPv: 2,
        requireStable: true,
        signal: controller.signal,
        onUpdate: (lines) => {
          const primary = lines.find((line) => line.rank === 1);
          if (!active || !primary) return;
          const topMove = primary.pv[0] ?? null;
          const updateKey = [
            primary.depth,
            primary.score.kind,
            primary.score.value,
            primary.score.bound ?? "",
            topMove ?? "",
          ].join(":");
          if (updateKey === lastPrimaryUpdate) return;
          lastPrimaryUpdate = updateKey;
          updates?.queue(primary.score);
          if (topMove === publishedTopMove) return;
          let topMoveSan: string | null;
          try {
            topMoveSan = sanForEngineMove(currentStudyPositionKey, topMove);
          } catch {
            return;
          }
          publishedTopMove = topMove;
          publishedTopMoveSan = topMoveSan;
          setStudyState({
            status: "loading",
            key: currentStudyKey,
            topLine: primary,
            topMoveSan,
          });
        },
      });
      if (!active) return;
      const primary = analysis.lines[0];
      if (!primary) {
        throw new StockfishError("Stockfish did not return a study line.");
      }
      const topMove = primary.pv[0] ?? null;
      const topMoveSan = topMove === publishedTopMove
        ? publishedTopMoveSan
        : sanForEngineMove(currentStudyPositionKey, topMove);
      if (!topMoveSan) {
        throw new StudyAnalysisError("Stockfish returned an empty study line.");
      }
      updates.complete(primary.score);
      studyPosition = {
        lines: analysis.lines,
        topMoveSan,
        review: null,
        reviewError: null,
        terminal: false,
      };
      storeLru(
        studyPositionCache.current,
        currentStudyPositionKey,
        studyPosition,
        EVALUATION_CACHE_LIMIT,
      );
    }

    async function ensureCurrentReview() {
      if (
        !studyPosition
        || studyPosition.terminal
        || studyPosition.review
        || studyPosition.reviewError
      ) return;
      const analysis = reviewAnalysis(studyPosition.lines);
      try {
        const currentReview = await createPositionReview(
          {
            fen: currentStudyPositionKey,
            feedback_id: null,
            analysis,
          },
          controller.signal,
        );
        if (!active) return;
        studyPosition = {
          ...studyPosition,
          review: currentReview,
          reviewError: null,
        };
      } catch (cause) {
        if (!active || isAbortError(cause)) throw cause;
        const failure = apiFailureFrom(cause);
        if (!failure) throw cause;
        studyPosition = {
          ...studyPosition,
          review: null,
          reviewError: failure,
        };
      }
      storeLru(
        studyPositionCache.current,
        currentStudyPositionKey,
        studyPosition,
        EVALUATION_CACHE_LIMIT,
      );
    }

    async function ensureMoveGrade() {
      if (gradeResult || !attemptedMove) {
        gradeResult ??= NO_STUDY_GRADE;
        return;
      }
      if (!parentFen || !parentPosition?.review || !studyPosition) {
        gradeResult = NO_STUDY_GRADE;
      } else if (
        canUseCachedBestGrade(currentTerminal)
        && isStudyBestMove(attemptedMove.uci, parentPosition.review)
      ) {
        gradeResult = {
          grade: classifyStudyMove(attemptedMove, parentPosition.review, null),
          error: null,
        };
      } else {
        let attemptLine: EngineLine;
        if (studyPosition.terminal) {
          attemptLine = terminalAttemptLine(
            attemptedMove.uci,
            current,
            parentPosition.lines[0]?.depth ?? 8,
          );
        } else {
          const childLine = studyPosition.lines[0];
          if (!childLine) {
            throw new Error("The current position has no principal line.");
          }
          attemptLine = attemptLineFromChild(attemptedMove.uci, childLine);
        }
        const bestLine = parentPosition.lines[0];
        if (!bestLine) {
          throw new Error("The parent position has no principal line.");
        }
        const analysis = positionAttemptAnalysis(
          bestLine,
          attemptedMove,
          attemptLine,
        );
        let comparison = null;
        try {
          comparison = await comparePositionAttempt(
            {
              fen: parentFen,
              path_dependent: currentTerminal === "draw",
              analysis,
            },
            controller.signal,
          );
        } catch (cause) {
          if (!active || isAbortError(cause)) throw cause;
          const failure = apiFailureFrom(cause);
          if (!failure) throw cause;
          gradeResult = { grade: null, error: failure };
        }
        if (!active) return;
        if (comparison) {
          gradeResult = {
            grade: classifyStudyMove(
              attemptedMove,
              parentPosition.review,
              comparison,
            ),
            error: null,
          };
        }
      }
      if (gradeResult) {
        storeLru(
          studyGradeCache.current,
          currentStudyKey,
          gradeResult,
          EVALUATION_CACHE_LIMIT,
        );
      }
    }

    async function analyzeStudyPosition() {
      try {
        await ensureStudyPosition();
        if (!active || !studyPosition) return;
        await Promise.all([ensureCurrentReview(), ensureMoveGrade()]);
        if (!active || !studyPosition || !gradeResult) return;
        publishStudyState(studyPosition, gradeResult);
      } catch (cause) {
        if (!active || isAbortError(cause)) return;
        const engineFailure = cause instanceof StockfishError
          || cause instanceof StudyAnalysisError;
        if (engineFailure) {
          engine.current?.dispose();
          engine.current = null;
        } else {
          studyPositionCache.current.delete(currentStudyPositionKey);
          studyGradeCache.current.delete(currentStudyKey);
        }
        const message = messageFrom(cause);
        setStudyState({ status: "error", key: currentStudyKey, message });
        setLiveError({
          source: "engine",
          message,
          retryable: engineFailure,
        });
      } finally {
        updates?.cancel();
        if (active && analysisGeneration.current === generation) {
          setEvaluatingFen((fen) =>
            fen === currentStudyPositionKey ? null : fen,
          );
        }
      }
    }

    void analyzeStudyPosition();
    return () => {
      active = false;
      controller.abort();
      updates?.cancel();
      if (analysisGeneration.current === generation) {
        setEvaluatingFen((fen) =>
          fen === currentStudyPositionKey ? null : fen,
        );
      }
    };
  }, [
    current,
    currentStudyKey,
    currentStudyPositionKey,
    currentTerminal,
    playedMoves,
    reviewState.status,
    studyMode,
    studyRetry,
  ]);

  function recordBoardInteraction() {
    restoreCoachingMovePreview();
    boardInteractionGeneration.current += 1;
    setAutomaticCue(null);
  }

  function toggleHint() {
    if (hintRevealed) {
      setPinnedCue(null);
      setHoveredCue(null);
    }
    setHintRevealed(!hintRevealed);
  }

  function play(move: AttemptedMove) {
    recordBoardInteraction();
    setLinePreview(null);
    if (studyMode) {
      setPlayedMoves((moves) => [...moves, move]);
      setPinnedCue(null);
      setLiveError(null);
      return;
    }
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
    recordBoardInteraction();
    setPlayedMoves((moves) => moves.slice(0, -1));
    setPinnedCue(null);
    if (studyMode) setLiveError(null);
  }

  function resetBoard() {
    recordBoardInteraction();
    setPlayedMoves([]);
    setLinePreview(null);
    setFirstAttempt(null);
    setPendingAttempt(null);
    setHintRevealed(false);
    setStudyMode(false);
    setStudyState({ status: "idle" });
    studyPositionCache.current.clear();
    studyGradeCache.current.clear();
    setLiveError(null);
    setCoachingState({ status: "idle" });
    ratingGeneration.current += 1;
    setRatingState({ reviewId: null, status: "idle" });
    setPinnedCue(null);
    setHoveredCue(null);
    if (initialReview.current) {
      setReviewState({ status: "ready", review: initialReview.current });
    }
    const score = rootEngineScore();
    evaluationCache.current.clear();
    if (score) {
      evaluationCache.current.set(rootFen, { score, complete: true });
    }
    setCurrentScore(score);
  }

  function startStudy() {
    recordBoardInteraction();
    setPlayedMoves([]);
    setLinePreview(null);
    setPinnedCue(null);
    setHoveredCue(null);
    setLiveError(null);
    setStudyMode(true);
    try {
      const rootPosition = seedRootStudyPosition();
      if (!rootPosition) {
        setStudyState({ status: "idle" });
        return;
      }
      const rootStudyKey = studyPathKey(rootFen, []);
      setStudyState(settledStudyState(
        rootStudyKey,
        rootPosition,
        NO_STUDY_GRADE,
      ));
      setCurrentScore(rootPosition.lines[0]?.score ?? null);
    } catch (cause) {
      const message = messageFrom(cause);
      setStudyState({
        status: "error",
        key: studyPathKey(rootFen, []),
        message,
      });
      setLiveError({ source: "engine", message });
    }
  }

  function leaveStudy() {
    recordBoardInteraction();
    setStudyMode(false);
    setStudyState({ status: "idle" });
    setPlayedMoves([]);
    setLinePreview(null);
    setPinnedCue(null);
    setHoveredCue(null);
    setLiveError(null);
    restoreRootEvaluation();
  }

  function resetStudyBoard() {
    recordBoardInteraction();
    setLiveError(null);
    try {
      const rootPosition = seedRootStudyPosition();
      if (rootPosition) {
        setStudyState(settledStudyState(
          studyPathKey(rootFen, []),
          rootPosition,
          NO_STUDY_GRADE,
        ));
      }
    } catch (cause) {
      const message = messageFrom(cause);
      setStudyState({
        status: "error",
        key: studyPathKey(rootFen, []),
        message,
      });
      setLiveError({ source: "engine", message });
    }
    setPlayedMoves([]);
    setLinePreview(null);
    restoreRootEvaluation();
  }

  function retryStudyAnalysis() {
    const cachedPosition = studyPositionCache.current.get(
      currentStudyPositionKey,
    );
    const cachedGrade = studyGradeCache.current.get(currentStudyKey);
    const retryData = prepareStudyRetry(cachedPosition, cachedGrade);
    if (
      cachedPosition
      && retryData.position
      && retryData.position !== cachedPosition
    ) {
      storeLru(
        studyPositionCache.current,
        currentStudyPositionKey,
        retryData.position,
        EVALUATION_CACHE_LIMIT,
      );
    }
    if (cachedGrade && retryData.gradeResult === null) {
      studyGradeCache.current.delete(currentStudyKey);
    }
    setStudyState({ status: "idle" });
    setLiveError(null);
    setStudyRetry((value) => value + 1);
  }

  function seedRootStudyPosition(): StudyPosition | null {
    const lines = reviewLines.current.get(position.full_fen);
    const sourceReview = initialReview.current;
    if (!lines || !sourceReview) return null;
    const rootPosition: StudyPosition = {
      lines,
      topMoveSan: sanForEngineMove(rootFen, lines[0]?.pv[0] ?? null),
      review: sourceReview,
      reviewError: null,
      terminal: false,
    };
    storeLru(
      studyPositionCache.current,
      studyPositionKey(rootFen),
      rootPosition,
      EVALUATION_CACHE_LIMIT,
    );
    return rootPosition;
  }

  function restoreRootEvaluation() {
    const score = rootEngineScore();
    if (score) {
      storeLru(
        evaluationCache.current,
        rootFen,
        { score, complete: true },
        EVALUATION_CACHE_LIMIT,
      );
    }
    setCurrentScore(score);
    setEvaluatingFen(null);
  }

  function rootEngineScore(): EngineScore | null {
    return rootEvaluationScore(
      reviewLines.current.get(position.full_fen),
      evaluationCache.current.get(rootFen)?.score ?? null,
      initialReview.current?.score ?? null,
    );
  }

  function showLine(line: ReviewLine) {
    recordBoardInteraction();
    setStudyMode(false);
    setPlayedMoves(attemptedMovesFor(position.full_fen, line.moves));
    setLinePreview(
      line.role === "best_candidate" || line.role === "alternative_candidate"
        ? "best"
        : "attempt",
    );
    setPinnedCue(null);
    setHoveredCue(null);
  }

  function previewCoachingMove(
    previewId: string,
    source: CoachingPreviewSource,
    move: CoachingMoveSegment,
  ) {
    if (!review) return;
    const line = coachingLine(review, move.scope);
    if (!line || line.moves[move.ply]?.uci !== move.move.uci) return;
    const preview = coachingPreview.current ?? {
      playedMoves,
      linePreview,
      pinnedCue,
      hoveredCue,
      liveError,
      active: new Map<string, ActiveCoachingPreview>(),
      nextOrder: 0,
      restoreFrame: null,
      suspended: false,
    };
    coachingPreview.current = preview;
    preview.suspended = false;
    if (preview.restoreFrame !== null) {
      window.cancelAnimationFrame(preview.restoreFrame);
      preview.restoreFrame = null;
    }
    preview.nextOrder += 1;
    preview.active.set(`${source}:${previewId}`, {
      move,
      line,
      order: preview.nextOrder,
    });
    applyCoachingMovePreview(move, line);
  }

  function endCoachingMovePreview(
    previewId: string,
    source: CoachingPreviewSource,
  ) {
    const preview = coachingPreview.current;
    if (!preview) return;
    preview.active.delete(`${source}:${previewId}`);
    if (preview.suspended) {
      if (preview.active.size === 0) discardCoachingMovePreview();
      return;
    }
    const latest = [...preview.active.values()].sort(
      (left, right) => right.order - left.order,
    )[0];
    if (latest) {
      applyCoachingMovePreview(latest.move, latest.line);
      return;
    }
    if (preview.restoreFrame !== null) return;
    preview.restoreFrame = window.requestAnimationFrame(() => {
      if (coachingPreview.current !== preview || preview.active.size > 0) return;
      restoreCoachingMovePreview();
    });
  }

  function applyCoachingMovePreview(
    move: CoachingMoveSegment,
    line: ReviewLine,
  ) {
    const cacheKey = `${review?.review_id ?? "ephemeral"}:${move.scope}:${move.ply}`;
    let moves = coachingMoveCache.current.get(cacheKey);
    if (!moves) {
      moves = attemptedMovesFor(
        position.full_fen,
        line.moves.slice(0, move.ply + 1),
      );
      coachingMoveCache.current.set(cacheKey, moves);
    }
    setPlayedMoves(moves);
    setLinePreview(move.scope === "best_line" ? "best" : "attempt");
    setPinnedCue(coachingMoveCue(move));
    setHoveredCue(null);
  }

  function discardCoachingMovePreview(): CoachingPreview | null {
    const preview = coachingPreview.current;
    if (preview && preview.restoreFrame !== null) {
      window.cancelAnimationFrame(preview.restoreFrame);
    }
    coachingPreview.current = null;
    return preview;
  }

  function restoreCoachingMovePreview(): CoachingPreview | null {
    const preview = discardCoachingMovePreview();
    if (!preview) return null;
    applyCoachingPreviewBaseline(preview);
    return preview;
  }

  function suspendCoachingMovePreview() {
    const preview = coachingPreview.current;
    if (!preview) return;
    if (preview.restoreFrame !== null) {
      window.cancelAnimationFrame(preview.restoreFrame);
      preview.restoreFrame = null;
    }
    preview.suspended = true;
    applyCoachingPreviewBaseline(preview);
  }

  function resumeCoachingMovePreview() {
    const preview = coachingPreview.current;
    if (!preview?.suspended) return;
    preview.suspended = false;
    const latest = [...preview.active.values()].sort(
      (left, right) => right.order - left.order,
    )[0];
    if (latest) applyCoachingMovePreview(latest.move, latest.line);
    else discardCoachingMovePreview();
  }

  function applyCoachingPreviewBaseline(preview: CoachingPreview) {
    setPlayedMoves(preview.playedMoves);
    setLinePreview(preview.linePreview);
    setPinnedCue(preview.pinnedCue);
    setHoveredCue(preview.hoveredCue);
    setLiveError(preview.liveError);
  }

  function showCoachingMove(move: CoachingMoveSegment) {
    if (!review) return;
    const line = coachingLine(review, move.scope);
    if (!line || line.moves[move.ply]?.uci !== move.move.uci) return;
    discardCoachingMovePreview();
    recordBoardInteraction();
    setStudyMode(false);
    applyCoachingMovePreview(move, line);
    window.requestAnimationFrame(() => {
      boardColumn.current?.scrollIntoView({
        behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "auto"
          : "smooth",
        block: "start",
      });
    });
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
        ...coachingPresentation(coachingState, position.coaching_available),
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

  function previewCue(cue: ReviewAnnotation, sourceReview: PositionReviewData) {
    if (hasBoardCue(cue)) {
      setPlayedMoves(movesForCue(cue, sourceReview, position.full_fen));
      setLinePreview(linePreviewForCue(cue));
    }
    setHoveredCue(null);
  }

  function selectCue(cue: ReviewAnnotation, sourceReview: PositionReviewData) {
    recordBoardInteraction();
    previewCue(cue, sourceReview);
    setPinnedCue(cue);
  }

  function selectAutomaticCue(
    cue: ReviewAnnotation,
    sourceReview: PositionReviewData,
  ) {
    previewCue(cue, sourceReview);
    setAutomaticCue(cue);
  }

  function pinCue(cue: ReviewAnnotation) {
    if (review) selectCue(cue, review);
  }

  const cueEvents = (cue: ReviewAnnotation, defaultActive = false) => ({
    active:
      pinnedCue === cue ||
      automaticCue === cue ||
      (defaultActive &&
        pinnedCue === null &&
        automaticCue === null &&
        hoveredCue === null),
    onEnter: () => {
      suspendCoachingMovePreview();
      setHoveredCue(cue);
    },
    onLeave: () => {
      const preview = coachingPreview.current;
      if (preview?.hoveredCue === cue) preview.hoveredCue = null;
      setHoveredCue(null);
      resumeCoachingMovePreview();
    },
    onToggle: () => pinCue(cue),
  });

  return (
    <main className="position-review">
      <header className="position-review__header">
        <button
          type="button"
          className="brand position-review__brand"
          onClick={onScanAnother}
        >
          <span className="brand__mark" aria-hidden="true">
            <i />
            <i />
            <i />
            <i />
          </span>
          <span>
            <strong>Chess Scan</strong>
            <em>review</em>
          </span>
        </button>
        <div
          className={`engine-status is-${liveError ? "error" : reviewState.status}`}
          role="status"
          title={liveError?.message}
        >
          <i aria-hidden="true" />
          <span>
            {engineStatus(
              reviewState.status,
              evaluatingFen === currentFen,
              liveError,
              rootIsTerminal,
              checkingAttempt,
            )}
          </span>
        </div>
      </header>

      <section className="position-review__layout">
        <div className="position-review__board-column" ref={boardColumn}>
          <div className="board-context">
            <div>
              <span>
                {currentTerminal
                  ? terminalResultLabel(currentTerminal)
                  : `${turnName(current.turn())} to move`}
              </span>
              <small>
                {studyMode
                  ? currentTerminal
                    ? "Line complete"
                    : (lastMove ? `After ${lastMove.san}` : "Study position")
                  : boardPositionLabel(lastMove, linePreview, boardCue)}
              </small>
            </div>
            <span className="board-context__topic">
              {studyMode
                ? (currentStudyNode?.review?.topic.name
                  ?? (currentStudyState.status === "loading" ? "Reading this position…" : "Study line"))
                : revealed && review
                  ? (boardCue ? displayCueLabel(boardCue.label) : review.topic.name)
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
              loading={
                currentTerminal === null &&
                (evaluatingFen === currentFen
                  || (studyMode
                    && currentStudyState.status === "loading")
                  || (reviewState.status === "loading"
                    && !rootIsTerminal
                    && !currentScore))
              }
              terminal={currentTerminal}
            />
            <InteractiveBoard
              fen={position.full_fen}
              orientation={position.orientation}
              moves={moveUcis}
              interactive={
                currentTerminal === null
                && reviewState.status === "ready"
                && (studyMode
                  ? currentStudyState.status === "ready"
                  : !checkingAttempt && linePreview === null)
              }
              cue={boardCue}
              engineMove={studyMode ? studyTopMove : null}
              moveGrade={studyMode ? studyGrade : null}
              onMove={play}
            />
          </div>

          {revealed && review && !studyMode && (
            <DiagramStory
              cues={explanation}
              activeCue={boardCue}
              onSelect={pinCue}
            />
          )}

          <div className="board-tools" aria-label="Board controls">
            <div>
              <button
                type="button"
                onClick={undo}
                disabled={playedMoves.length === 0}
              >
                Undo
              </button>
              <button
                type="button"
                onClick={studyMode ? resetStudyBoard : resetBoard}
                disabled={studyMode
                  ? playedMoves.length === 0
                  : playedMoves.length === 0 && firstAttempt === null}
              >
                Reset
              </button>
              {studyMode && (
                <button type="button" onClick={leaveStudy}>
                  Review
                </button>
              )}
              {studyMode && liveError && liveError.retryable !== false && (
                <button type="button" onClick={retryStudyAnalysis}>
                  Retry analysis
                </button>
              )}
              {!studyMode && liveError?.source === "engine" && playedMoves.length > 0 && (
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
              Local eval{" "}
              <b>{evaluationLabel(currentScore, current.turn(), current)}</b>
            </span>
          </div>
        </div>

        <article
          className="review-notes"
          aria-live={reviewState.status === "loading" ? "off" : "polite"}
        >
          {reviewState.status === "loading" && (
            <div className="review-loading">
              <span className="review-loading__mark" aria-hidden="true" />
              <p className="commentary-kicker">
                {rootIsTerminal ? "Rules check" : "Reading the position"}
              </p>
              <h1>
                {rootIsTerminal
                  ? "Confirming the result…"
                  : "Finding the useful clue…"}
              </h1>
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
                <button
                  type="button"
                  className="primary-button"
                  onClick={() => setRetry((value) => value + 1)}
                >
                  Try the review again
                </button>
                <a
                  className="secondary-button"
                  href={position.lichess_url}
                  target="_blank"
                  rel="noreferrer"
                >
                  Open in Lichess ↗
                </a>
              </div>
            </>
          )}

          {reviewState.status === "ready" && (
            studyMode ? (
              <StudyPanel
                state={currentStudyState}
                turn={current.turn()}
                terminal={currentTerminal}
                onLeave={leaveStudy}
              />
            ) : (
            <>
              {revealed && (
                <BoardReference
                  className="topic-reference"
                  cue={reviewState.review.hint}
                  {...cueEvents(reviewState.review.hint)}
                >
                  <span>Position topic</span>
                  <strong>{reviewState.review.topic.name}</strong>
                  <small>
                    {hasBoardCue(reviewState.review.hint)
                      ? "Show on board"
                      : "Position guide"}
                  </small>
                </BoardReference>
              )}

              {reviewState.review.best_move === null ? (
                <div className="review-explanation">
                  <p className="commentary-kicker">
                    {reviewState.review.evaluation}
                  </p>
                  <h1>{reviewState.review.hint.text}</h1>
                  <div className="annotation-list">
                    {explanation.map((cue, index) => (
                      <BoardReference
                        key={annotationKey(cue)}
                        className="annotation-reference"
                        cue={cue}
                        {...cueEvents(cue, index === 0)}
                      >
                        <span className="annotation-index" aria-hidden="true">
                          {cueRoleMark(cue)}
                        </span>
                        <span>
                          <strong>{displayCueLabel(cue.label)}</strong>
                          <span className="annotation-reference__copy">
                            {cue.text}
                          </span>
                        </span>
                        <small>{hasBoardCue(cue) ? "Board" : "Note"}</small>
                      </BoardReference>
                    ))}
                  </div>
                </div>
              ) : !revealed ? (
                <div className="review-prompt">
                  <p className="commentary-kicker">
                    {turnName(root.turn())} to move
                  </p>
                  <h1>What is the first move?</h1>
                  <button
                    type="button"
                    className={`hint-reveal${hintRevealed ? " is-expanded" : ""}`}
                    aria-expanded={hintRevealed}
                    aria-controls="position-hint"
                    onClick={toggleHint}
                  >
                    <span className="annotation-index">Hint</span>
                    <span>
                      <strong>{hintRevealed ? "Hide hint" : "Show hint"}</strong>
                      <small>
                        {hintRevealed
                          ? "Return to the position without the clue"
                          : "Reveal one clue without naming the move"}
                      </small>
                    </span>
                    <span aria-hidden="true">{hintRevealed ? "−" : "+"}</span>
                  </button>
                  <div
                    id="position-hint"
                    className="hint-revealed"
                    hidden={!hintRevealed}
                  >
                    <BoardReference
                      className="hint-reference"
                      cue={reviewState.review.hint}
                      {...cueEvents(reviewState.review.hint)}
                    >
                      <span className="annotation-index">Hint</span>
                      <span className="hint-reference__copy">
                        {reviewState.review.hint.text}
                      </span>
                      {hasBoardCue(reviewState.review.hint) && (
                        <small>Tap to mark the clues on the board</small>
                      )}
                    </BoardReference>
                  </div>
                  <p className="spoiler-note">
                    <i aria-hidden="true" /> No move names until you try.
                  </p>
                </div>
              ) : checkingAttempt ? (
                <div className="review-loading">
                  <span className="review-loading__mark" aria-hidden="true" />
                  <p className="commentary-kicker">Your move</p>
                  <h1>Checking the strongest reply…</h1>
                  <p>
                    The line is hypothetical until the local engine finishes
                    comparing it.
                  </p>
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
                    {explanation.some(hasBoardCue)
                      ? "Move through the board story to see the attempt, tactical idea, and checked alternative."
                      : "Use each note to compare your move with the checked engine lines."}
                  </p>
                  <CoachingSummary
                    state={coachingState}
                    review={reviewState.review}
                    activeCueId={pinnedCue?.id ?? null}
                    onPreviewMove={previewCoachingMove}
                    onEndPreview={endCoachingMovePreview}
                    onShowMove={showCoachingMove}
                  />
                  <div className="annotation-list">
                    {explanation.map((cue, index) => (
                      <BoardReference
                        key={annotationKey(cue)}
                        className="annotation-reference"
                        cue={cue}
                        {...cueEvents(cue, index === 0)}
                      >
                        <span className="annotation-index" aria-hidden="true">
                          {cueRoleMark(cue)}
                        </span>
                        <span>
                          <strong>{displayCueLabel(cue.label)}</strong>
                          <span className="annotation-reference__copy">
                            {cue.text}
                          </span>
                        </span>
                        <small>{hasBoardCue(cue) ? "Board" : "Note"}</small>
                      </BoardReference>
                    ))}
                  </div>
                  <section className="study-entry">
                    <span className="study-entry__mark" aria-hidden="true">✦</span>
                    <span>
                      <strong>Keep playing from this position</strong>
                      <small>Explore both sides with the top move, tactics, and move grades.</small>
                    </span>
                    <button type="button" onClick={startStudy}>
                      Study position <span aria-hidden="true">→</span>
                    </button>
                  </section>
                  <div
                    className="review-actions"
                    aria-label="Hypothetical engine lines"
                  >
                    {reviewState.review.attempt && (
                      <button
                        type="button"
                        className="secondary-button"
                        onClick={() =>
                          showLine(reviewState.review.attempt!.line)
                        }
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
                    <i aria-hidden="true" /> These are engine continuations, not
                    moves already played.
                  </p>
                  {reviewState.review.review_id && (
                    <div
                      className="review-feedback"
                      aria-label="Rate this analysis"
                    >
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
                            onChange={(event) =>
                              setRatingReason(
                                event.target.value as typeof ratingReason,
                              )
                            }
                          >
                            <option value="incorrect_chess">
                              Incorrect chess
                            </option>
                            <option value="irrelevant_topic">
                              Irrelevant topic
                            </option>
                            <option value="unclear">Unclear explanation</option>
                            <option value="equivalent_move_rejected">
                              Equivalent move rejected
                            </option>
                            <option value="too_verbose">Too verbose</option>
                            <option value="missing_detail">
                              Missing useful detail
                            </option>
                            <option value="other">Other</option>
                          </select>
                          <button
                            type="button"
                            disabled={ratingStatus === "sending"}
                            onClick={() => void rateReview("unhelpful")}
                          >
                            Report issue
                          </button>
                          {ratingStatus === "error" && (
                            <span>Feedback could not be saved.</span>
                          )}
                        </>
                      )}
                    </div>
                  )}
                  <button
                    type="button"
                    className="try-again-button"
                    onClick={resetBoard}
                  >
                    Hide the answer & try again
                  </button>
                </div>
              )}
            </>
            )
          )}
        </article>
      </section>

      <footer className="position-review__footer">
        <span className="position-review__credits">
          <span>
            <a href="/stockfish/Copying.txt" target="_blank" rel="noreferrer">
              Stockfish 18 lite · GPLv3
            </a>
            {" · "}
            <a href="/stockfish/SOURCE.md" target="_blank" rel="noreferrer">
              Source
            </a>
          </span>
          <ChessPieceAttribution className="position-review__piece-credit" />
        </span>
        <div>
          <button type="button" onClick={onScanAnother}>
            Scan another position
          </button>
          <a href={position.lichess_url} target="_blank" rel="noreferrer">
            Lichess ↗
          </a>
        </div>
      </footer>
    </main>
  );
}

function coachingPresentation(
  state: CoachingState,
  coachingAvailable: boolean,
): {
  coaching_status: CoachingPresentationStatus;
  commentary_run_id: string | null;
} {
  if (state.status === "idle") {
    return {
      coaching_status: coachingAvailable ? "not_shown" : "disabled",
      commentary_run_id: null,
    };
  }
  if (state.status === "loading" || state.status === "unavailable") {
    return { coaching_status: state.status, commentary_run_id: null };
  }
  if (state.coaching.status === "disabled") {
    return { coaching_status: "disabled", commentary_run_id: null };
  }
  if (!hasCoachingNarrative(state.coaching)) {
    return { coaching_status: "not_shown", commentary_run_id: null };
  }
  return {
    coaching_status: state.coaching.status,
    commentary_run_id: state.coaching.run_id,
  };
}

function CoachingSummary({
  state,
  review,
  activeCueId,
  onPreviewMove,
  onEndPreview,
  onShowMove,
}: {
  state: CoachingState;
  review: PositionReviewData;
  activeCueId: string | null;
  onPreviewMove: (
    previewId: string,
    source: CoachingPreviewSource,
    move: CoachingMoveSegment,
  ) => void;
  onEndPreview: (previewId: string, source: CoachingPreviewSource) => void;
  onShowMove: (move: CoachingMoveSegment) => void;
}) {
  if (state.status === "idle") return null;
  if (state.status === "unavailable") {
    return (
      <aside className="coaching-summary is-unavailable" role="status">
        <strong>Extra coaching couldn’t be loaded.</strong>
        <small>You can still use the engine line and board explanations below.</small>
      </aside>
    );
  }
  if (state.status === "loading") {
    return (
      <aside className="coaching-summary is-loading" role="status">
        <span className="coaching-summary__pulse" aria-hidden="true" />
        <span>
          <strong>Coach is writing the calculation note…</strong>
          <small>The checked engine review is already available.</small>
        </span>
      </aside>
    );
  }

  const { coaching } = state;
  if (!hasCoachingNarrative(coaching)) return null;
  const rootTurn = new Chess(review.fen).turn();
  return (
    <aside
      className="coaching-note"
      aria-label="Evidence-backed coaching note"
      aria-live="polite"
    >
      <header className="coaching-note__header">
        <span className="coaching-note__eyebrow">Coach's note</span>
        <small>Select any move to open that exact position on the board.</small>
      </header>
      <div className="coaching-note__sections">
        {coaching.sections.map((section, sectionIndex) => (
          <section
            className={`coaching-note__section is-${section.kind}`}
            key={`${section.kind}-${section.title}`}
          >
            <h2>{section.title}</h2>
            <p>
              {section.segments.map((segment, index) =>
                segment.type === "text" ? (
                  <span key={`${section.kind}-text-${index}`}>{segment.text}</span>
                ) : (
                  <CoachingMoveToken
                    active={activeCueId === coachingMoveCue(segment).id}
                    key={`${segment.scope}-${segment.ply}-${index}`}
                    move={segment}
                    previewId={`${sectionIndex}-${index}`}
                    rootTurn={rootTurn}
                    onPreview={onPreviewMove}
                    onPreviewEnd={onEndPreview}
                    onShow={onShowMove}
                  />
                ),
              )}
            </p>
          </section>
        ))}
      </div>
    </aside>
  );
}

function CoachingMoveToken({
  move,
  previewId,
  rootTurn,
  active,
  onPreview,
  onPreviewEnd,
  onShow,
}: {
  move: CoachingMoveSegment;
  previewId: string;
  rootTurn: "w" | "b";
  active: boolean;
  onPreview: (
    previewId: string,
    source: CoachingPreviewSource,
    move: CoachingMoveSegment,
  ) => void;
  onPreviewEnd: (previewId: string, source: CoachingPreviewSource) => void;
  onShow: (move: CoachingMoveSegment) => void;
}) {
  const display = coachingMoveDisplay(
    move.move.san,
    coachingMoveColor(rootTurn, move.ply),
  );
  return (
    <button
      type="button"
      className={`coaching-move is-${move.role}${active ? " is-active" : ""}`}
      aria-label={coachingMoveLabel(move)}
      aria-pressed={active}
      onPointerEnter={() => onPreview(previewId, "pointer", move)}
      onPointerLeave={() => onPreviewEnd(previewId, "pointer")}
      onFocus={() => onPreview(previewId, "focus", move)}
      onBlur={() => onPreviewEnd(previewId, "focus")}
      onClick={() => onShow(move)}
    >
      <span className="coaching-move__piece" aria-hidden="true">
        {display.symbol}
      </span>
      <span>{display.notation}</span>
    </button>
  );
}

function coachingExplanationOrder(
  cues: ReviewAnnotation[],
  coaching: PositionCoaching | null,
): ReviewAnnotation[] {
  if (coaching?.status !== "accepted") return cues;
  const byId = new Map(cues.map((cue) => [cue.id, cue]));
  if (byId.size !== cues.length) throw new Error("Review contains duplicate annotation IDs");
  const selected = coaching.lesson_ids.map((lessonId) => {
    const cue = byId.get(lessonId);
    if (!cue) throw new Error(`Coaching references unknown lesson: ${lessonId}`);
    return cue;
  });
  const selectedIds = new Set(coaching.lesson_ids);
  return [...selected, ...cues.filter((cue) => !selectedIds.has(cue.id))];
}

function annotationKey(cue: ReviewAnnotation): string {
  return cue.id;
}

function DiagramStory({
  cues,
  activeCue,
  onSelect,
}: {
  cues: ReviewAnnotation[];
  activeCue: ReviewAnnotation | null;
  onSelect: (cue: ReviewAnnotation) => void;
}) {
  const diagrams = cues.filter(hasBoardCue);
  const selectedIndex = diagrams.findIndex((cue) => cue === activeCue);
  const selected =
    selectedIndex >= 0 ? (diagrams[selectedIndex] ?? null) : null;
  const stepRail = useRef<HTMLDivElement | null>(null);
  const selectedStep = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    const rail = stepRail.current;
    const step = selectedStep.current;
    if (!rail || !step) return;
    const left = step.offsetLeft;
    const right = left + step.offsetWidth;
    if (left < rail.scrollLeft) rail.scrollTo({ left });
    else if (right > rail.scrollLeft + rail.clientWidth) {
      rail.scrollTo({ left: right - rail.clientWidth });
    }
  }, [selectedIndex]);

  if (diagrams.length === 0) return null;
  return (
    <nav className="diagram-story" aria-label="Board story">
      <div className="diagram-story__heading">
        <span>Board story</span>
        <span>Tap to replay</span>
      </div>
      <div ref={stepRail} className="diagram-story__steps">
        {diagrams.map((cue) => {
          const badge = cueBadge(cue);
          const active = cue === selected;
          return (
            <button
              key={annotationKey(cue)}
              ref={active ? selectedStep : null}
              type="button"
              className={`diagram-step is-${cueRole(cue)}${active ? " is-active" : ""}`}
              aria-label={cueAccessibleLabel(cue)}
              aria-pressed={active}
              onClick={() => onSelect(cue)}
            >
              <span
                className={`diagram-step__mark is-${cueRole(cue)}`}
                aria-hidden="true"
              >
                {badge ? <ReviewGlyph badge={badge} /> : cueRoleMark(cue)}
              </span>
              <span>
                <strong>{displayCueLabel(cue.label)}</strong>
                <small>Tap to show</small>
              </span>
            </button>
          );
        })}
      </div>
      {selected && (
        <p
          key={annotationKey(selected)}
          className="diagram-story__caption"
        >
          {selected.text}
        </p>
      )}
    </nav>
  );
}

function cueBadge(cue: ReviewAnnotation): ReviewBadge | null {
  return cue.badge?.kind ?? null;
}

function cueRole(cue: ReviewAnnotation): string {
  return cue.arrows[0]?.role ?? cue.markers[0]?.role ?? "focus";
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
      aria-label={cueAccessibleLabel(cue)}
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
  const whiteShare =
    terminal === "white"
      ? 100
      : terminal === "black"
        ? 0
        : terminal === "draw"
          ? 50
          : evaluationShare(score, turn);
  const whiteAtBottom = orientation === "white";
  const whiteFavored = whiteShare >= 50;
  const scoreAtBottom = whiteFavored === whiteAtBottom;
  const label =
    terminal === "white"
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
      aria-valuetext={
        label === "—" ? "Evaluation loading" : `Evaluation ${label}`
      }
    >
      <div
        className={`eval-bar__white is-${whiteAtBottom ? "bottom" : "top"}`}
        style={{ height: `${whiteShare}%` }}
      />
      <span
        className={`eval-bar__score is-${scoreAtBottom ? "bottom" : "top"} on-${whiteFavored ? "white" : "black"}`}
      >
        {label}
      </span>
      <span
        className={`eval-bar__side is-top on-${whiteAtBottom ? "black" : "white"}`}
      >
        {whiteAtBottom ? "B" : "W"}
      </span>
      <span
        className={`eval-bar__side is-bottom on-${whiteAtBottom ? "white" : "black"}`}
      >
        {whiteAtBottom ? "W" : "B"}
      </span>
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
    storeLru(cache, fen, { score, complete }, EVALUATION_CACHE_LIMIT);
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

function cacheCandidateEvaluations(
  cache: Map<string, CachedEvaluation>,
  rootFen: string,
  lines: EngineLine[],
) {
  for (const line of lines) {
    const move = line.pv[0];
    if (move) cacheAttemptEvaluation(cache, rootFen, move, line.score);
  }
}

function cacheAttemptEvaluation(
  cache: Map<string, CachedEvaluation>,
  rootFen: string,
  attemptedMove: string,
  score: EngineScore,
) {
  const childFen = positionAt(rootFen, [attemptedMove]).fen();
  storeLru(cache, childFen, {
    score: oppositeScorePov(score),
    complete: true,
  }, EVALUATION_CACHE_LIMIT);
}

function positionAttemptAnalysis(
  bestLine: EngineLine,
  attempt: AttemptedMove,
  attemptLine: EngineLine,
): PositionAttemptRequest["analysis"] {
  return {
    score_pov: "side_to_move",
    best_line: reviewEngineLine({ ...bestLine, rank: 1 }),
    attempt: {
      move: attempt.uci,
      line: reviewEngineLine({ ...attemptLine, rank: 1 }),
    },
  };
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

function initialTeachingDiagram(
  explanation: ReviewAnnotation[],
): ReviewAnnotation | null {
  const diagrams = explanation.filter(hasBoardCue);
  return (
    diagrams.find((cue) => cue.arrows[0]?.role !== "played") ??
    diagrams[0] ??
    null
  );
}

function coachingLine(
  review: PositionReviewData,
  scope: CoachingMoveSegment["scope"],
): ReviewLine | undefined {
  return scope === "best_line" ? review.lines[0] : review.attempt?.line;
}


function linePreviewForCue(cue: ReviewAnnotation): LinePreview {
  if (cue.scope === "best_line") return "best";
  if (cue.scope === "attempt_line" || cue.scope === "attempt_refutation")
    return "attempt";
  return null;
}

function movesForCue(
  cue: ReviewAnnotation,
  review: PositionReviewData,
  fen: string,
): AttemptedMove[] {
  const preview = linePreviewForCue(cue);
  const line =
    preview === "best"
      ? review.lines.find((candidate) => candidate.rank === 1)
      : preview === "attempt"
        ? review.attempt?.line
        : undefined;
  if (!line) return [];
  return attemptedMovesFor(fen, line.moves.slice(0, cue.ply));
}

function attemptedMovesFor(
  fen: string,
  moves: ReviewMove[],
): AttemptedMove[] {
  const parentFens = parentFensForMoves(
    fen,
    moves.map((move) => move.uci),
  );
  return moves.map((move, index) => {
    const parentFen = parentFens[index];
    if (!parentFen) throw new Error("Review line is missing a parent position.");
    return { uci: move.uci, san: move.san, parentFen };
  });
}

function attemptVerdict(
  attempt: AttemptedMove | null,
  review: PositionReviewData,
): string {
  if (!review.best_move) return review.evaluation;
  if (review.attempt) return review.attempt.headline;
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
  if (score.kind === "mate")
    return signed >= 0 ? `M${Math.abs(signed)}` : `−M${Math.abs(signed)}`;
  const pawns = signed / 100;
  return `${pawns >= 0 ? "+" : "−"}${Math.abs(pawns).toFixed(1)}`;
}

function evaluationLabel(
  score: EngineScore | null,
  turn: "w" | "b",
  chess: Chess,
): string {
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

function terminalResultLabel(
  terminal: "white" | "black" | "draw",
): string {
  return terminal === "draw"
    ? "Draw"
    : `${terminal === "white" ? "White" : "Black"} wins`;
}

function turnName(turn: "w" | "b"): string {
  return turn === "w" ? "White" : "Black";
}

function boardPositionLabel(
  lastMove: AttemptedMove | null,
  linePreview: LinePreview,
  cue: ReviewAnnotation | null,
): string {
  if (lastMove)
    return `${linePreview ? "Hypothetical · after" : "After"} ${lastMove.san}`;
  const cuePreview = cue ? linePreviewForCue(cue) : null;
  if (cuePreview === "attempt") return "Your move · illustrated";
  if (cuePreview === "best") return "Best move · illustrated";
  if (cue?.scope === "terminal") return "Final position";
  return "Starting position";
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
  if (status === "loading" || evaluatingCurrent)
    return "Stockfish is checking locally";
  return "Local analysis ready";
}

function messageFrom(cause: unknown): string {
  return cause instanceof Error
    ? cause.message
    : "The position could not be reviewed.";
}

function isAbortError(cause: unknown): boolean {
  return cause instanceof DOMException && cause.name === "AbortError";
}
