import { useEffect, useRef, useState, type CSSProperties } from "react";
import { detectBoard } from "../api";
import {
  cornerStability,
  minimumBoardEdge,
  requiredCaptureBoardEdge,
  polygonCenter,
  projectedBoardCell,
  scaleCorners,
} from "../camera";
import {
  canvasBlob,
  captureCameraFrame,
  captureSupplementalFrames,
  type CameraFrame,
} from "../cameraCapture";
import {
  AUTO_CAPTURE_PROFILES,
  autoCaptureProfile,
  automaticCaptureReady,
  labelAgreement,
  loadAutoCaptureMode,
  meanPredictionConfidence,
  saveAutoCaptureMode,
  scanReadiness,
  smoothScanReadiness,
  type AutoCaptureMode,
} from "../cameraReadiness";
import {
  finiteMediaSetting,
  startDiagnosticRecording,
  type DiagnosticRecording,
  type RecordedDiagnostic,
} from "../cameraDiagnostic";
import type {
  BoardDetection,
  CameraCapture,
  CaptureDiagnostic,
  CaptureGeometry,
  Point,
} from "../types";
import PhotoPicker from "./PhotoPicker";

const SUPPLEMENTAL_FRAME_COUNT = 3;

interface ReadyCapture {
  frame: CameraFrame;
  geometry: CaptureGeometry;
  detection: BoardDetection;
  roundTripMs: number;
}

type CameraPhase =
  | "requesting"
  | "searching"
  | "framing"
  | "steadying"
  | "capturing"
  | "locked"
  | "error";

interface LiveCameraProps {
  diagnosticsEnabled: boolean;
  onCapture: (capture: CameraCapture) => Promise<boolean>;
  onCancel: () => void;
}

export default function LiveCamera({
  diagnosticsEnabled,
  onCapture,
  onCancel,
}: LiveCameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recordingRef = useRef<DiagnosticRecording | null>(null);
  const photoDiagnosticRef = useRef<Promise<CaptureDiagnostic | null> | null>(null);
  const timerRef = useRef<number | null>(null);
  const stoppedRef = useRef(false);
  const lockedRef = useRef(false);
  const onCaptureRef = useRef(onCapture);
  const detectionAttemptsRef = useRef(0);
  const missedDetectionsRef = useRef(0);
  const previewRoundTripRef = useRef<number | null>(null);
  const previousCornersRef = useRef<Point[] | null>(null);
  const previousLabelsRef = useRef<number[] | null>(null);
  const stableDetectionCountRef = useRef(0);
  const consecutiveMissesRef = useRef(0);
  const readyCaptureRef = useRef<ReadyCapture | null>(null);
  const readinessRef = useRef(0);
  const [autoCaptureMode, setAutoCaptureModeState] = useState<AutoCaptureMode>(() =>
    loadAutoCaptureMode(browserStorage()),
  );
  const autoCaptureModeRef = useRef(autoCaptureMode);
  const [phase, setPhase] = useState<CameraPhase>("requesting");
  const [detection, setDetection] = useState<BoardDetection | null>(null);
  const [lockedFrameUrl, setLockedFrameUrl] = useState<string | null>(null);
  const [diagnosticActive, setDiagnosticActive] = useState(false);
  const [burstProgress, setBurstProgress] = useState(0);
  const [readiness, setReadiness] = useState(0);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    onCaptureRef.current = onCapture;
  }, [onCapture]);

  useEffect(() => {
    autoCaptureModeRef.current = autoCaptureMode;
  }, [autoCaptureMode]);

  useEffect(() => {
    let cancelled = false;
    stoppedRef.current = false;
    void startCamera(() => cancelled || stoppedRef.current);
    return () => {
      cancelled = true;
      stopCamera();
    };
  }, []);

  useEffect(() => () => {
    if (lockedFrameUrl) URL.revokeObjectURL(lockedFrameUrl);
  }, [lockedFrameUrl]);

  async function startCamera(isCancelled: () => boolean) {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCameraError("This browser cannot open a live camera. Choose an existing photo instead.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
      });
      if (isCancelled()) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) {
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        return;
      }
      video.srcObject = stream;
      await video.play();
      if (isCancelled()) {
        stream.getTracks().forEach((track) => track.stop());
        if (streamRef.current === stream) streamRef.current = null;
        return;
      }
      if (diagnosticsEnabled) {
        recordingRef.current = startDiagnosticRecording(stream);
        setDiagnosticActive(recordingRef.current !== null);
      }
      setPhase("searching");
      scheduleDetection(60);
    } catch (cause) {
      if (isCancelled()) return;
      const denied = cause instanceof DOMException && cause.name === "NotAllowedError";
      setCameraError(
        denied
          ? "Camera access was not allowed. Enable it in the browser or choose a photo."
          : "The camera could not be started. Choose an existing photo instead.",
      );
    }
  }

  function setCameraError(message: string) {
    setError(message);
    setPhase("error");
  }

  function scheduleDetection(delay: number) {
    if (stoppedRef.current || lockedRef.current) return;
    timerRef.current = window.setTimeout(() => void runDetection(), delay);
  }

  async function runDetection() {
    const video = videoRef.current;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      scheduleDetection(60);
      return;
    }
    try {
      const frame = await captureCameraFrame(video);
      detectionAttemptsRef.current += 1;
      const requestStarted = performance.now();
      const result = await detectBoard(frame.preview);
      const roundTripMs = Math.max(0, Math.round(performance.now() - requestStarted));
      if (stoppedRef.current || lockedRef.current) return;
      if (!result.found) {
        handleDetectionMiss();
        scheduleDetection(100);
        return;
      }

      consecutiveMissesRef.current = 0;
      setDetection(result);
      const capturedCorners = scaleCorners(
        result.corners,
        result.image_width,
        result.image_height,
        frame.source.width,
        frame.source.height,
      );
      const requiredBoardEdge = requiredCaptureBoardEdge(
        frame.source.width,
        frame.source.height,
      );
      const boardSizeRatio = minimumBoardEdge(capturedCorners) / requiredBoardEdge;
      const stability = cornerStability(previousCornersRef.current, result.corners);
      const prediction = result.prediction;
      const agreement = prediction
        ? labelAgreement(previousLabelsRef.current, prediction.labels)
        : null;
      const sameBoard = stability >= 0.7
        || (stability >= 0.35 && agreement !== null && agreement >= 0.88);
      stableDetectionCountRef.current = sameBoard
        ? stableDetectionCountRef.current + 1
        : 1;
      previousCornersRef.current = result.corners;
      previousLabelsRef.current = prediction?.labels ?? null;
      const measuredReadiness = scanReadiness({
        boardSizeRatio,
        geometryStability: stability,
        predictionAgreement: agreement,
        sharpnessScore: prediction?.sharpness_score ?? 0,
        glareScore: prediction?.glare_score ?? 1,
        detectionConfidence: result.confidence,
        predictionConfidence: meanPredictionConfidence(prediction?.confidences ?? []),
        stableRounds: stableDetectionCountRef.current,
      });
      const nextReadiness = updateScanReadiness(measuredReadiness);
      previewRoundTripRef.current = roundTripMs;
      readyCaptureRef.current = {
        frame,
        geometry: {
          corners: capturedCorners,
          method: result.method,
        },
        detection: result,
        roundTripMs,
      };

      if (boardSizeRatio < 1) {
        setPhase("framing");
        scheduleDetection(80);
        return;
      }
      const profile = autoCaptureProfile(autoCaptureModeRef.current);
      if (automaticCaptureReady(
        nextReadiness,
        profile,
        stableDetectionCountRef.current,
        boardSizeRatio,
        prediction !== null,
      )) {
        await captureAutomatically();
        return;
      }

      setPhase("steadying");
      scheduleDetection(120);
    } catch {
      if (stoppedRef.current) return;
      if (lockedRef.current) {
        stopStream();
        setCameraError("The locked frame could not be prepared. Close and try again.");
        return;
      }
      handleDetectionMiss();
      scheduleDetection(220);
    }
  }

  async function captureAutomatically() {
    const video = videoRef.current;
    const ready = readyCaptureRef.current;
    if (!video || !ready || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    lockedRef.current = true;
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    previewRoundTripRef.current = ready.roundTripMs;
    setBurstProgress(1);
    setPhase("capturing");
    navigator.vibrate?.(45);
    try {
      await captureFinalFrame(
        ready.frame.source,
        video,
        ready.geometry,
        "automatic",
        ready.detection,
      );
    } catch {
      lockedRef.current = false;
      recordingRef.current?.discard();
      recordingRef.current = null;
      stopStream();
      setCameraError("The frame could not be captured. Close and try again, or choose a photo.");
    }
  }

  async function captureFinalFrame(
    source: HTMLCanvasElement,
    video: HTMLVideoElement,
    geometry: CaptureGeometry | undefined,
    captureKind: "automatic",
    currentDetection: BoardDetection | null,
  ) {
    const primaryBlobPromise = canvasBlob(source, 0.88);
    const supplementalImages = await captureSupplementalFrames(
      video,
      SUPPLEMENTAL_FRAME_COUNT,
      (capturedCount) => setBurstProgress(capturedCount + 1),
    );
    const recordedPromise = recordingRef.current?.stop() ?? Promise.resolve(null);
    recordingRef.current = null;
    const primaryBlob = await primaryBlobPromise;
    if (stoppedRef.current) return;
    setLockedFrameUrl(URL.createObjectURL(primaryBlob));
    setPhase("locked");
    stopStream();
    const image = new File([primaryBlob], `chess-board-${Date.now()}.jpg`, {
      type: "image/jpeg",
    });
    const diagnostic = recordedPromise.then((recorded) => recorded
      ? buildCaptureDiagnostic(
          recorded,
          currentDetection,
          captureKind,
          supplementalImages.length + 1,
        )
      : null);
    const profile = autoCaptureProfile(autoCaptureModeRef.current);
    const accepted = await onCaptureRef.current({
      image,
      geometry,
      supplementalImages,
      readiness: {
        schema_version: "scan-readiness-1",
        capture_kind: "automatic",
        auto_capture_mode: profile.mode,
        readiness_score: readinessRef.current,
        readiness_threshold: profile.threshold,
        detection_attempts: detectionAttemptsRef.current,
        missed_detections: missedDetectionsRef.current,
      },
      diagnostic,
      diagnosticRequested: diagnosticsEnabled,
    });
    if (!accepted && !stoppedRef.current) {
      setCameraError("This frame could not be read. Close the camera and try again.");
    }
  }

  function buildCaptureDiagnostic(
    recorded: RecordedDiagnostic,
    currentDetection: BoardDetection | null,
    captureKind: "automatic" | "manual" | "photo_fallback",
    submittedFrames: number,
  ): CaptureDiagnostic {
    const prediction = currentDetection?.prediction;
    return {
      blob: recorded.blob,
      metadata: {
        schema_version: "capture-diagnostic-1",
        consent_version: "camera-diagnostic-consent-1",
        recording_duration_ms: recorded.durationMs,
        camera_width: finiteMediaSetting(recorded.settings.width),
        camera_height: finiteMediaSetting(recorded.settings.height),
        camera_frame_rate: finiteMediaSetting(recorded.settings.frameRate),
        detection_attempts: detectionAttemptsRef.current,
        missed_detections: missedDetectionsRef.current,
        lock_method: currentDetection?.method ?? null,
        preview_round_trip_ms: previewRoundTripRef.current,
        preview_processing_ms: currentDetection?.processing_ms ?? null,
        preview_sharpness_score: prediction?.sharpness_score ?? null,
        scan_round_trip_ms: 0,
        submitted_frames: submittedFrames,
        glare_square_count: prediction?.glare_squares.length ?? 0,
        glare_score: prediction?.glare_score ?? 0,
        capture_kind: captureKind,
      },
    };
  }

  function stopStream() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  function handleDetectionMiss() {
    missedDetectionsRef.current += 1;
    consecutiveMissesRef.current += 1;
    stableDetectionCountRef.current = Math.max(0, stableDetectionCountRef.current - 1);
    updateScanReadiness(null);
    if (consecutiveMissesRef.current >= 3) {
      previousCornersRef.current = null;
      previousLabelsRef.current = null;
      readyCaptureRef.current = null;
      setDetection(null);
      setPhase("searching");
      return;
    }
    setPhase("steadying");
  }

  function updateScanReadiness(measured: number | null): number {
    const next = smoothScanReadiness(readinessRef.current, measured);
    readinessRef.current = next;
    setReadiness(next);
    return next;
  }

  function chooseAutoCaptureMode(mode: AutoCaptureMode) {
    autoCaptureModeRef.current = mode;
    setAutoCaptureModeState(mode);
    saveAutoCaptureMode(browserStorage(), mode);
  }

  function resetCaptureReadiness() {
    previousCornersRef.current = null;
    previousLabelsRef.current = null;
    stableDetectionCountRef.current = 0;
    consecutiveMissesRef.current = 0;
    readyCaptureRef.current = null;
    readinessRef.current = 0;
  }

  function stopCamera() {
    stoppedRef.current = true;
    resetCaptureReadiness();
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    recordingRef.current?.discard();
    recordingRef.current = null;
    stopStream();
  }

  function cancel() {
    stopCamera();
    onCancel();
  }

  function openPhotoPicker() {
    stoppedRef.current = true;
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    const recorded = recordingRef.current?.stop() ?? Promise.resolve(null);
    recordingRef.current = null;
    photoDiagnosticRef.current = recorded.then((result) => result
      ? buildCaptureDiagnostic(result, detection, "photo_fallback", 0)
      : null);
    stopStream();
  }

  function choosePhoto(file: File) {
    stopCamera();
    void onCaptureRef.current({
      image: file,
      supplementalImages: [],
      diagnostic: photoDiagnosticRef.current ?? Promise.resolve(null),
      diagnosticRequested: diagnosticsEnabled,
    });
    onCancel();
  }

  const profile = autoCaptureProfile(autoCaptureMode);
  const status = cameraStatus(phase, burstProgress, readiness, profile.threshold);
  const locked = phase === "locked";
  const captureStarted = phase === "capturing" || locked;

  return (
    <section className={`live-camera is-${phase}`} aria-label="Live chess board camera">
      <video ref={videoRef} className="live-camera__video" muted playsInline />
      {lockedFrameUrl && (
        <img className="live-camera__video is-frozen" src={lockedFrameUrl} alt="" />
      )}
      <div className="live-camera__shade" aria-hidden="true" />
      {detection?.found && (
        <BoardGridOverlay
          detection={detection}
          readiness={readiness}
          locked={captureStarted}
        />
      )}
      <div className="live-camera__corners" aria-hidden="true">
        <i />
        <i />
        <i />
        <i />
      </div>
      <header className="live-camera__topbar">
        <button type="button" className="camera-icon-button" onClick={cancel} aria-label="Close camera">
          ×
        </button>
        <span className="live-camera__mode">Board scanner</span>
        <span className={`live-camera__live${diagnosticActive ? " is-diagnostic" : ""}`}>
          {diagnosticActive
            ? "Diagnostic"
            : diagnosticsEnabled
              ? "No diagnostic"
              : "Live"}
        </span>
      </header>
      {phase !== "error" && phase !== "capturing" && !locked && (
        <div className="live-camera__auto-controls">
          <div className="camera-auto-mode">
            <span>Auto-capture</span>
            <div role="group" aria-label="Automatic capture precision">
              {AUTO_CAPTURE_PROFILES.map((option) => (
                <button
                  type="button"
                  key={option.mode}
                  className={option.mode === autoCaptureMode ? "is-active" : ""}
                  aria-pressed={option.mode === autoCaptureMode}
                  aria-label={`${option.label}, capture at ${option.threshold}% readiness`}
                  title={option.detail}
                  onClick={() => chooseAutoCaptureMode(option.mode)}
                >
                  <strong>{option.label}</strong>
                  <small>{option.threshold}%</small>
                </button>
              ))}
            </div>
          </div>
          <PhotoPicker
            className="camera-photo-button"
            onOpen={openPhotoPicker}
            onCancel={cancel}
            onPhoto={choosePhoto}
          >
            Photos
          </PhotoPicker>
        </div>
      )}
      <div className="live-camera__feedback" aria-live="polite">
        <span className="live-camera__status-icon" aria-hidden="true">
          <i className="status-searching">⌗</i>
          <i className="status-locked">✓</i>
        </span>
        <div>
          <strong>{status.title}</strong>
          <p>{status.detail}</p>
          {phase === "capturing" && (
            <span
              className="camera-burst-progress"
              aria-label={`${burstProgress} of ${SUPPLEMENTAL_FRAME_COUNT + 1} frames captured`}
            >
              {Array.from({ length: SUPPLEMENTAL_FRAME_COUNT + 1 }, (_, index) => (
                <i key={index} className={index < burstProgress ? "is-complete" : ""} />
              ))}
            </span>
          )}
        </div>
      </div>
      {phase === "error" && (
        <div className="live-camera__error">
          <p>{error}</p>
          <PhotoPicker
            className="camera-fallback-button"
            onOpen={openPhotoPicker}
            onCancel={cancel}
            onPhoto={choosePhoto}
          >
            Choose a photo
          </PhotoPicker>
        </div>
      )}
    </section>
  );
}

function BoardGridOverlay({
  detection,
  readiness,
  locked,
}: {
  detection: BoardDetection;
  readiness: number;
  locked: boolean;
}) {
  const rows = Array.from({ length: 9 }, (_, row) =>
    detection.grid_points.slice(row * 9, row * 9 + 9),
  );
  const columns = Array.from({ length: 9 }, (_, col) =>
    Array.from({ length: 9 }, (_, row) => detection.grid_points[row * 9 + col]).filter(
      (point): point is Point => point !== undefined,
    ),
  );
  const prediction = detection.prediction;
  const center = polygonCenter(detection.corners);
  const boardEdge = averageEdge(detection.corners);
  const style = {
    "--camera-readiness-color": readinessColor(readiness),
  } as CSSProperties;
  return (
    <svg
      className={`camera-grid ${locked ? "is-locked" : ""}`}
      viewBox={`0 0 ${detection.image_width} ${detection.image_height}`}
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
      style={style}
    >
      <polygon points={pointString(detection.corners)} className="camera-grid__fill" />
      {[...rows, ...columns].map((points, index) => (
        <polyline key={index} points={pointString(points)} className="camera-grid__line" />
      ))}
      {prediction?.glare_squares.map((index) => {
        const cell = projectedBoardCell(detection.grid_points, index);
        return cell
          ? <polygon key={`glare-${index}`} points={pointString(cell)} className="camera-grid__glare" />
          : null;
      })}
      {center && (
        <g className="camera-grid__readiness">
          <rect
            x={center[0] - boardEdge * 0.3}
            y={center[1] - boardEdge * 0.14}
            width={boardEdge * 0.6}
            height={boardEdge * 0.28}
            rx={boardEdge * 0.045}
          />
          <text
            x={center[0]}
            y={center[1] - boardEdge * 0.012}
            className="camera-grid__readiness-value"
            fontSize={Math.max(34, boardEdge * 0.15)}
          >
            {readiness}%
          </text>
          <text
            x={center[0]}
            y={center[1] + boardEdge * 0.075}
            className="camera-grid__readiness-label"
            fontSize={Math.max(9, boardEdge * 0.027)}
          >
            SCAN READINESS
          </text>
        </g>
      )}
      {detection.corners.map(([x, y], index) => (
        <circle key={index} cx={x} cy={y} r={9} className="camera-grid__corner" />
      ))}
    </svg>
  );
}

function cameraStatus(
  phase: CameraPhase,
  burstProgress: number,
  readiness: number,
  threshold: number,
) {
  if (phase === "requesting") {
    return { title: "Opening camera", detail: "Camera permission is required once." };
  }
  if (phase === "framing") {
    return {
      title: `${readiness}% scan readiness`,
      detail: "Move closer while keeping all four board edges visible.",
    };
  }
  if (phase === "steadying") {
    return {
      title: `${readiness}% scan readiness`,
      detail: `Hold steady—capture starts automatically at ${threshold}%.`,
    };
  }
  if (phase === "capturing") {
    return {
      title: `Capturing ${burstProgress} of ${SUPPLEMENTAL_FRAME_COUNT + 1}`,
      detail: "Hold still—the camera is still taking the short burst.",
    };
  }
  if (phase === "locked") {
    return {
      title: "Frames captured",
      detail: "The camera is off. Identifying the pieces now.",
    };
  }
  if (phase === "error") {
    return { title: "Camera unavailable", detail: "Use an existing photograph instead." };
  }
  if (readiness > 0) {
    return {
      title: `${readiness}% scan readiness`,
      detail: "Keep one complete board centered while tracking recovers.",
    };
  }
  return { title: "Find one board", detail: "Fill the guide with one complete diagram." };
}

function pointString(points: Point[]): string {
  return points.map(([x, y]) => `${x},${y}`).join(" ");
}

function averageEdge(points: Point[]): number {
  if (points.length < 2) return 0;
  return points.reduce((sum, [x, y], index) => {
    const next = points[(index + 1) % points.length];
    return next ? sum + Math.hypot(next[0] - x, next[1] - y) : sum;
  }, 0) / points.length;
}

function readinessColor(readiness: number): string {
  const progress = Math.min(100, Math.max(0, readiness)) / 100;
  const hue = 35 + 110 * progress ** 1.4;
  return `hsl(${hue} 82% 58%)`;
}

function browserStorage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}
