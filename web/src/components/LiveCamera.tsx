import { useEffect, useRef, useState } from "react";
import { detectBoard } from "../api";
import {
  MIN_PREVIEW_BOARD_EDGE,
  minimumBoardEdge,
  polygonCenter,
  projectedBoardCell,
  scaleCorners,
} from "../camera";
import {
  canvasBlob,
  captureCameraFrame,
  captureSupplementalFrames,
  SOURCE_MAX_DIMENSION,
  videoFrameCanvas,
} from "../cameraCapture";
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

const SUPPLEMENTAL_FRAME_COUNT = 2;

type CameraPhase =
  | "requesting"
  | "searching"
  | "framing"
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
  const [phase, setPhase] = useState<CameraPhase>("requesting");
  const [detection, setDetection] = useState<BoardDetection | null>(null);
  const [lockedFrameUrl, setLockedFrameUrl] = useState<string | null>(null);
  const [diagnosticActive, setDiagnosticActive] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    onCaptureRef.current = onCapture;
  }, [onCapture]);

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
      setDetection(result);
      if (!result.found) {
        missedDetectionsRef.current += 1;
        setPhase("searching");
        scheduleDetection(100);
        return;
      }

      if (minimumBoardEdge(result.corners) < MIN_PREVIEW_BOARD_EDGE) {
        setPhase("framing");
        scheduleDetection(90);
        return;
      }

      previewRoundTripRef.current = roundTripMs;
      lockedRef.current = true;
      setPhase("locked");
      navigator.vibrate?.(45);
      await captureFinalFrame(
        frame.source,
        video,
        {
          corners: scaleCorners(
            result.corners,
            result.image_width,
            result.image_height,
            frame.source.width,
            frame.source.height,
          ),
          method: result.method,
        },
        "automatic",
        result,
      );
    } catch {
      if (stoppedRef.current) return;
      if (lockedRef.current) {
        stopStream();
        setCameraError("The locked frame could not be prepared. Close and try again.");
        return;
      }
      missedDetectionsRef.current += 1;
      setPhase("searching");
      scheduleDetection(220);
    }
  }

  async function captureManually() {
    const video = videoRef.current;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    lockedRef.current = true;
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    setPhase("capturing");
    try {
      await captureFinalFrame(
        videoFrameCanvas(video, SOURCE_MAX_DIMENSION),
        video,
        undefined,
        "manual",
        detection,
      );
    } catch {
      lockedRef.current = false;
      setCameraError("The frame could not be captured. Close and try again, or choose a photo.");
    }
  }

  async function captureFinalFrame(
    source: HTMLCanvasElement,
    video: HTMLVideoElement,
    geometry: CaptureGeometry | undefined,
    captureKind: "automatic" | "manual",
    currentDetection: BoardDetection | null,
  ) {
    const primaryBlobPromise = canvasBlob(source, 0.88);
    const supplementalCount = geometry && currentDetection?.prediction?.glare_squares.length
      ? SUPPLEMENTAL_FRAME_COUNT
      : 0;
    const supplementalPromise = supplementalCount > 0
      ? captureSupplementalFrames(video, supplementalCount)
      : Promise.resolve([]);
    const recordedPromise = recordingRef.current?.stop() ?? Promise.resolve(null);
    const primaryBlob = await primaryBlobPromise;
    if (stoppedRef.current) return;
    setLockedFrameUrl(URL.createObjectURL(primaryBlob));
    setPhase("locked");

    const supplementalImages = await supplementalPromise;
    if (stoppedRef.current) return;
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
    const accepted = await onCaptureRef.current({
      image,
      geometry,
      supplementalImages,
      diagnostic,
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

  function stopCamera() {
    stoppedRef.current = true;
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
    });
    onCancel();
  }

  const status = cameraStatus(phase, detection);
  const locked = phase === "locked";

  return (
    <section className={`live-camera is-${phase}`} aria-label="Live chess board camera">
      <video ref={videoRef} className="live-camera__video" muted playsInline />
      {lockedFrameUrl && (
        <img className="live-camera__video is-frozen" src={lockedFrameUrl} alt="" />
      )}
      <div className="live-camera__shade" aria-hidden="true" />
      {detection?.found && (
        <BoardGridOverlay detection={detection} locked={locked} />
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
          {diagnosticActive ? "Diagnostic" : "Live"}
        </span>
      </header>
      {phase !== "error" && !locked && (
        <div className="live-camera__manual-controls">
          <button
            type="button"
            className="camera-shutter"
            disabled={phase === "requesting" || phase === "capturing"}
            onClick={() => void captureManually()}
          >
            <span aria-hidden="true"><i /></span>
            <strong>Capture now</strong>
          </button>
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

function BoardGridOverlay({ detection, locked }: { detection: BoardDetection; locked: boolean }) {
  const rows = Array.from({ length: 9 }, (_, row) =>
    detection.grid_points.slice(row * 9, row * 9 + 9),
  );
  const columns = Array.from({ length: 9 }, (_, col) =>
    Array.from({ length: 9 }, (_, row) => detection.grid_points[row * 9 + col]).filter(
      (point): point is Point => point !== undefined,
    ),
  );
  const prediction = detection.prediction;
  return (
    <svg
      className={`camera-grid ${locked ? "is-locked" : ""}`}
      viewBox={`0 0 ${detection.image_width} ${detection.image_height}`}
      preserveAspectRatio="xMidYMid slice"
      aria-hidden="true"
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
      {prediction?.labels.map((label, index) => {
        if (label === 0 || (prediction.confidences[index] ?? 0) < 0.45) return null;
        const cell = projectedBoardCell(detection.grid_points, index);
        const center = cell ? polygonCenter(cell) : null;
        if (!cell || !center) return null;
        const size = Math.max(15, averageEdge(cell) * 0.68);
        return (
          <text
            key={`piece-${index}`}
            x={center[0]}
            y={center[1]}
            className="camera-grid__piece"
            fontSize={size}
            style={{ animationDelay: `${Math.min(index, 40) * 9}ms` }}
          >
            {PIECE_GLYPHS[label]}
          </text>
        );
      })}
      {detection.corners.map(([x, y], index) => (
        <circle key={index} cx={x} cy={y} r={9} className="camera-grid__corner" />
      ))}
    </svg>
  );
}

function cameraStatus(phase: CameraPhase, detection: BoardDetection | null) {
  if (phase === "requesting") {
    return { title: "Opening camera", detail: "Camera permission is required once." };
  }
  if (phase === "framing") {
    return { title: "Move closer", detail: "Keep all four edges visible and fill the guide." };
  }
  if (phase === "capturing") {
    return { title: "Photo captured", detail: "Identifying the pieces now." };
  }
  if (phase === "locked") {
    const glareSquares = detection?.prediction?.glare_squares.length ?? 0;
    return glareSquares > 0
      ? {
          title: "Board locked",
          detail: `Combining nearby frames to reduce glare on ${glareSquares} square${glareSquares === 1 ? "" : "s"}.`,
        }
      : { title: "Board locked", detail: "Pieces found. Preparing the board now." };
  }
  if (phase === "error") {
    return { title: "Camera unavailable", detail: "Use an existing photograph instead." };
  }
  return { title: "Find the board", detail: "Fill the guide with one complete diagram." };
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

const PIECE_GLYPHS: Record<number, string> = {
  1: "♙",
  2: "♘",
  3: "♗",
  4: "♖",
  5: "♕",
  6: "♔",
  7: "♟",
  8: "♞",
  9: "♝",
  10: "♜",
  11: "♛",
  12: "♚",
};
