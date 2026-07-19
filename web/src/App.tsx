import { useEffect, useMemo, useRef, useState } from "react";
import { confirmScan, getLearningStatus, reprocessScan, scanImage } from "./api";
import { countKings, fullFen, predictionNeedsReview } from "./board";
import AppHeader from "./components/AppHeader";
import BoardEditor from "./components/BoardEditor";
import CapturePanel from "./components/CapturePanel";
import CornerEditor from "./components/CornerEditor";
import RecognitionSuccess from "./components/RecognitionSuccess";
import type {
  ConfirmResult,
  LearningStatus,
  Orientation,
  Point,
  ScanResult,
  SideToMove,
} from "./types";

type BusyAction = "scan" | "reprocess" | "confirm" | null;

export default function App() {
  const [sourceFile, setSourceFile] = useState<File | null>(null);
  const [sourceUrl, setSourceUrl] = useState<string | null>(null);
  const [scan, setScan] = useState<ScanResult | null>(null);
  const [corners, setCorners] = useState<Point[]>([]);
  const [labels, setLabels] = useState<number[]>([]);
  const [rectifiedImageUrl, setRectifiedImageUrl] = useState<string | null>(null);
  const [orientation, setOrientation] = useState<Orientation>("white");
  const [sideToMove, setSideToMove] = useState<SideToMove>("w");
  const [castlingRights, setCastlingRights] = useState<string[]>([]);
  const [consentTraining, setConsentTraining] = useState(true);
  const [busy, setBusy] = useState<BusyAction>(null);
  const [error, setError] = useState<string | null>(null);
  const [confirmation, setConfirmation] = useState<ConfirmResult | null>(null);
  const [reviewReady, setReviewReady] = useState(false);
  const [geometryOpen, setGeometryOpen] = useState(false);
  const [learningStatus, setLearningStatus] = useState<LearningStatus | null>(null);
  const [clientSessionId] = useState(getClientSessionId);
  const requestGeneration = useRef(0);

  useEffect(() => {
    void getLearningStatus().then(setLearningStatus).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!sourceFile) {
      setSourceUrl(null);
      return;
    }
    const url = URL.createObjectURL(sourceFile);
    setSourceUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [sourceFile]);

  const rectifiedSourceUrl = scan?.rectified_image_url;
  useEffect(() => {
    setRectifiedImageUrl(null);
    if (!rectifiedSourceUrl) return;

    const controller = new AbortController();
    let active = true;
    let objectUrl: string | null = null;
    void fetch(rectifiedSourceUrl, { cache: "no-store", signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`Image request failed (${response.status})`);
        return response.blob();
      })
      .then((blob) => {
        const url = URL.createObjectURL(blob);
        if (!active) {
          URL.revokeObjectURL(url);
          return;
        }
        objectUrl = url;
        setRectifiedImageUrl(url);
      })
      .catch((cause: unknown) => {
        if (
          active
          && !(cause instanceof DOMException && cause.name === "AbortError")
        ) {
          setRectifiedImageUrl(rectifiedSourceUrl);
        }
      });
    return () => {
      active = false;
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [rectifiedSourceUrl]);

  const castling = castlingRights.length > 0 ? "KQkq".split("").filter((right) => castlingRights.includes(right)).join("") : "-";
  const fen = scan ? fullFen(labels, orientation, sideToMove, castling) : "";
  const changedSquares = labels.filter((label, index) => label !== scan?.labels[index]).length;
  const kings = countKings(labels);
  const confidence = scan
    ? Math.round((scan.confidences.reduce((sum, value) => sum + value, 0) / 64) * 100)
    : 0;
  const uncertainPredictionCount = scan
    ? scan.labels.filter((label, index) =>
        predictionNeedsReview(
          label,
          scan.confidences[index] ?? 0,
          scan.probabilities[index] ?? [],
        ),
      ).length
    : 0;

  async function handleImage(file: File) {
    const generation = ++requestGeneration.current;
    setError(null);
    setConfirmation(null);
    setBusy("scan");
    setSourceFile(file);
    try {
      const result = await scanImage(file);
      if (generation !== requestGeneration.current) return;
      applyScan(result);
      const needsManualFrame = result.detection_method === "manual_adjustment_needed";
      setReviewReady(needsManualFrame);
      setGeometryOpen(needsManualFrame);
    } catch (cause) {
      if (generation !== requestGeneration.current) return;
      setError(messageFrom(cause));
      setSourceFile(null);
    } finally {
      if (generation === requestGeneration.current) setBusy(null);
    }
  }

  function applyScan(result: ScanResult) {
    setScan(result);
    setCorners(result.corners);
    setLabels(result.labels);
  }

  async function handleReprocess() {
    if (!scan) return;
    const generation = ++requestGeneration.current;
    setError(null);
    setConfirmation(null);
    setBusy("reprocess");
    try {
      const result = await reprocessScan(scan.scan_id, corners);
      if (generation !== requestGeneration.current) return;
      applyScan(result);
      setReviewReady(true);
      setGeometryOpen(false);
    } catch (cause) {
      if (generation === requestGeneration.current) setError(messageFrom(cause));
    } finally {
      if (generation === requestGeneration.current) setBusy(null);
    }
  }

  async function handleConfirm() {
    if (!scan) return;
    const generation = ++requestGeneration.current;
    setError(null);
    setBusy("confirm");
    const lichessTab = window.open("about:blank", "_blank");
    if (lichessTab) {
      lichessTab.document.title = "Opening Lichess…";
      lichessTab.document.body.textContent = "Saving the corrected position…";
      lichessTab.opener = null;
    }
    try {
      const result = await confirmScan(scan.scan_id, {
        labels,
        orientation,
        side_to_move: sideToMove,
        castling,
        en_passant: "-",
        consent_training: consentTraining,
        client_session_id: clientSessionId,
      });
      if (generation !== requestGeneration.current) {
        lichessTab?.close();
        return;
      }
      setConfirmation(result);
      void getLearningStatus().then(setLearningStatus).catch(() => undefined);
      if (lichessTab) lichessTab.location.replace(result.lichess_url);
      else window.location.assign(result.lichess_url);
    } catch (cause) {
      lichessTab?.close();
      if (generation === requestGeneration.current) setError(messageFrom(cause));
    } finally {
      if (generation === requestGeneration.current) setBusy(null);
    }
  }

  function reset() {
    requestGeneration.current += 1;
    setSourceFile(null);
    setScan(null);
    setCorners([]);
    setLabels([]);
    setRectifiedImageUrl(null);
    setOrientation("white");
    setSideToMove("w");
    setCastlingRights([]);
    setConfirmation(null);
    setReviewReady(false);
    setGeometryOpen(false);
    setError(null);
    setBusy(null);
  }

  const positionWarnings = useMemo(() => {
    const warnings: string[] = [];
    if (kings.white !== 1) warnings.push(`Expected one white king; found ${kings.white}.`);
    if (kings.black !== 1) warnings.push(`Expected one black king; found ${kings.black}.`);
    return warnings;
  }, [kings.black, kings.white]);

  const captureWarnings = useMemo(() => {
    if (!scan) return [];
    const warnings: string[] = [];
    if (minimumBoardEdge(scan.corners) < 256) {
      warnings.push("The board is low resolution. A closer photo will be more accurate.");
    }
    if (uncertainPredictionCount > 0) {
      warnings.push(
        `${uncertainPredictionCount} square prediction${uncertainPredictionCount === 1 ? " needs" : "s need"} review. Check every outlined square.`,
      );
    } else if (confidence < 80) {
      warnings.push("Model confidence is low. Review every square.");
    }
    if (scan.detection_method === "manual_adjustment_needed") {
      warnings.push("Automatic framing failed. Place all four corner handles before checking pieces.");
    }
    return warnings;
  }, [confidence, scan, uncertainPredictionCount]);
  const reviewWarnings = [...captureWarnings, ...positionWarnings];

  return (
    <div className="app-frame">
      <AppHeader onReset={scan && busy === null ? reset : undefined} />
      {error && (
        <div className="error-banner" role="alert">
          <span>!</span>
          <p>{error}</p>
          <button type="button" onClick={() => setError(null)} aria-label="Dismiss error">×</button>
        </div>
      )}

      {!scan ? (
        <CapturePanel busy={busy === "scan"} status={learningStatus} onImage={handleImage} />
      ) : !reviewReady ? (
        <RecognitionSuccess
          imageUrl={rectifiedImageUrl}
          onContinue={() => setReviewReady(true)}
        />
      ) : (
        <main className="review-shell">
          <nav className="progress-rail" aria-label="Scan progress">
            <span className="is-complete"><b>1</b> Frame</span>
            <i />
            <span className="is-current"><b>2</b> Check</span>
            <i />
            <span><b>3</b> Lichess</span>
          </nav>

          {confirmation && (
            <section className="success-banner">
              <span aria-hidden="true">✓</span>
              <div>
                <strong>Position saved.</strong>
                <p>Lichess opened in a new tab. This board changed {confirmation.changed_squares} model prediction{confirmation.changed_squares === 1 ? "" : "s"}.</p>
              </div>
              <button type="button" className="secondary-button" disabled={busy !== null} onClick={reset}>Scan another</button>
            </section>
          )}

          <header className="review-heading">
            <div>
              <p className="eyebrow">Human check · Step 02</p>
              <h1>Make the position exact.</h1>
            </div>
            <div className="model-stamp">
              <span>{confidence}% avg.</span>
              <small>{scan.model_version}</small>
            </div>
          </header>

          {reviewWarnings.length > 0 && (
            <aside className="quality-banner" role="status">
              <span aria-hidden="true">!</span>
              <div>
                <strong>Give this scan a closer check</strong>
                {reviewWarnings.map((warning) => <p key={warning}>{warning}</p>)}
              </div>
            </aside>
          )}

          <details
            className="geometry-panel"
            open={geometryOpen}
            onToggle={(event) => setGeometryOpen(event.currentTarget.open)}
          >
            <summary>
              <span>
                <b>Board frame</b>
                <small>{geometryMessage(scan.detection_method)}</small>
              </span>
              <span className="summary-action">Adjust corners</span>
            </summary>
            {sourceUrl && (
              <div className="geometry-panel__body">
                <CornerEditor
                  imageUrl={sourceUrl}
                  width={scan.source_width}
                  height={scan.source_height}
                  corners={corners}
                  onChange={setCorners}
                  disabled={busy !== null}
                />
                <div className="geometry-panel__instructions">
                  <p>Place the numbered handles on the four outside corners of the board.</p>
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={busy !== null}
                    onClick={handleReprocess}
                  >
                    {busy === "reprocess" ? "Reading again…" : "Re-read this frame"}
                  </button>
                </div>
              </div>
            )}
          </details>

          <section className="position-workbench">
            <article className="source-board-panel">
              <div className="panel-label">
                <span>Source crop</span>
                <small>After perspective correction</small>
              </div>
              <div className="source-board-frame">
                {rectifiedImageUrl && (
                  <img src={rectifiedImageUrl} alt="Rectified workbook diagram" />
                )}
                <div className="source-board-grid" aria-hidden="true" />
              </div>
            </article>

            <article className="prediction-panel">
              <div className="panel-label">
                <span>Model reading</span>
                <small>Tap any square to correct it</small>
              </div>
              <BoardEditor
                labels={labels}
                predictedLabels={scan.labels}
                confidences={scan.confidences}
                probabilities={scan.probabilities}
                orientation={orientation}
                onChange={setLabels}
              />
            </article>
          </section>

          <section className="position-controls">
            <div className="control-group">
              <span className="control-label">Side to move</span>
              <div className="segmented-control">
                <button type="button" className={sideToMove === "w" ? "is-active" : ""} onClick={() => setSideToMove("w")}>White</button>
                <button type="button" className={sideToMove === "b" ? "is-active" : ""} onClick={() => setSideToMove("b")}>Black <small>• in book</small></button>
              </div>
            </div>
            <div className="control-group">
              <span className="control-label">Board orientation</span>
              <div className="segmented-control">
                <button type="button" className={orientation === "white" ? "is-active" : ""} onClick={() => setOrientation("white")}>White below</button>
                <button type="button" className={orientation === "black" ? "is-active" : ""} onClick={() => setOrientation("black")}>Black below</button>
              </div>
            </div>
          </section>

          <details className="advanced-panel">
            <summary>Advanced FEN options</summary>
            <fieldset>
              <legend>Castling rights</legend>
              {["K", "Q", "k", "q"].map((right) => (
                <label key={right}>
                  <input
                    type="checkbox"
                    checked={castlingRights.includes(right)}
                    onChange={() => setCastlingRights((current) => current.includes(right) ? current.filter((value) => value !== right) : [...current, right])}
                  />
                  <span>{castlingLabel(right)}</span>
                </label>
              ))}
            </fieldset>
          </details>

          <section className="fen-slip">
            <div>
              <span>Final FEN</span>
              <code>{fen}</code>
            </div>
            <div className="fen-slip__status">
              {changedSquares > 0 && <span className="correction-count">{changedSquares} corrected</span>}
              {positionWarnings.map((warning) => <span key={warning} className="warning-chip">{warning}</span>)}
            </div>
          </section>

          <label className="consent-row">
            <input type="checkbox" checked={consentTraining} onChange={(event) => setConsentTraining(event.target.checked)} />
            <span>
              <strong>Help the scanner learn from this board</strong>
              Only the rectified diagram, prediction, and final labels are retained for training—not the full photograph.
            </span>
          </label>

          <footer className="review-actions">
            <button type="button" className="text-button" disabled={busy !== null} onClick={reset}>Start over</button>
            <button type="button" className="primary-button analyse-button" disabled={busy !== null || confirmation !== null} onClick={handleConfirm}>
              <span>{busy === "confirm" ? "Saving…" : "Save & analyse on Lichess"}</span>
              <span aria-hidden="true">↗</span>
            </button>
          </footer>
        </main>
      )}
      <footer className="site-footer">
        <span>Chess Scan / experimental vision tool</span>
        <span>{learningStatus?.active_model ?? "model loading"}</span>
      </footer>
    </div>
  );
}

function minimumBoardEdge(corners: Point[]): number {
  if (corners.length !== 4) return 0;
  return Math.min(
    ...corners.map(([x, y], index) => {
      const next = corners[(index + 1) % corners.length];
      return next ? Math.hypot(next[0] - x, next[1] - y) : 0;
    }),
  );
}

function getClientSessionId(): string {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  return Array.from(crypto.getRandomValues(new Uint32Array(4)), (value) =>
    value.toString(16).padStart(8, "0"),
  ).join("");
}

function geometryMessage(method: string): string {
  if (method === "checkerboard") return "Internal grid found automatically";
  if (method === "contour") return "Outer board edge found automatically";
  if (method === "manual") return "Using your adjusted corners";
  return "Automatic framing was uncertain—please check it";
}

function castlingLabel(right: string): string {
  return ({ K: "White O-O", Q: "White O-O-O", k: "Black O-O", q: "Black O-O-O" } as Record<string, string>)[right] ?? right;
}

function messageFrom(cause: unknown): string {
  return cause instanceof Error ? cause.message : "Something went wrong";
}
