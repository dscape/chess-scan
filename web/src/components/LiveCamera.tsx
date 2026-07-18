import { useEffect, useRef, useState } from "react";
import { detectBoard } from "../api";
import type { BoardDetection, Point } from "../types";

type CameraPhase =
  | "requesting"
  | "searching"
  | "framing"
  | "aligning"
  | "capturing"
  | "locked"
  | "error";

interface LiveCameraProps {
  onCapture: (file: File) => void;
  onCancel: () => void;
  onChoosePhoto: () => void;
}

export default function LiveCamera({
  onCapture,
  onCancel,
  onChoosePhoto,
}: LiveCameraProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<number | null>(null);
  const stoppedRef = useRef(false);
  const lockedRef = useRef(false);
  const previousCornersRef = useRef<Point[] | null>(null);
  const stableFramesRef = useRef(0);
  const onCaptureRef = useRef(onCapture);
  const [phase, setPhase] = useState<CameraPhase>("requesting");
  const [detection, setDetection] = useState<BoardDetection | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    onCaptureRef.current = onCapture;
  }, [onCapture]);

  useEffect(() => {
    stoppedRef.current = false;
    void startCamera();
    return stopCamera;
  }, []);

  async function startCamera() {
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
      if (stoppedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }
      streamRef.current = stream;
      const video = videoRef.current;
      if (!video) return;
      video.srcObject = stream;
      await video.play();
      setPhase("searching");
      scheduleDetection(120);
    } catch (cause) {
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
      scheduleDetection(180);
      return;
    }
    try {
      const frame = await frameBlob(video, 960, 0.74);
      const result = await detectBoard(frame);
      if (stoppedRef.current) return;
      setDetection(result);
      if (!result.found) {
        previousCornersRef.current = null;
        stableFramesRef.current = 0;
        setPhase("searching");
        scheduleDetection(360);
        return;
      }

      if (boardFillRatio(result) < 0.22) {
        previousCornersRef.current = result.corners;
        stableFramesRef.current = 0;
        setPhase("framing");
        scheduleDetection(360);
        return;
      }

      const stable = cornersAreStable(previousCornersRef.current, result);
      stableFramesRef.current = stable ? stableFramesRef.current + 1 : 1;
      previousCornersRef.current = result.corners;
      if (stableFramesRef.current >= 3) {
        lockedRef.current = true;
        setPhase("locked");
        navigator.vibrate?.(45);
        await new Promise((resolve) => window.setTimeout(resolve, 720));
        if (!stoppedRef.current) await captureFinalFrame(video);
        return;
      }

      setPhase("aligning");
      scheduleDetection(300);
    } catch {
      if (stoppedRef.current) return;
      lockedRef.current = false;
      stableFramesRef.current = 0;
      setPhase("searching");
      scheduleDetection(700);
    }
  }

  async function captureManually() {
    const video = videoRef.current;
    if (!video || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
    lockedRef.current = true;
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    setPhase("capturing");
    try {
      await new Promise((resolve) => window.setTimeout(resolve, 120));
      if (!stoppedRef.current) await captureFinalFrame(video);
    } catch {
      lockedRef.current = false;
      setCameraError("The frame could not be captured. Close and try again, or choose a photo.");
    }
  }

  async function captureFinalFrame(video: HTMLVideoElement) {
    const blob = await frameBlob(video, 2000, 0.93);
    const file = new File([blob], `chess-board-${Date.now()}.jpg`, { type: "image/jpeg" });
    stopStream();
    onCaptureRef.current(file);
  }

  function stopStream() {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
  }

  function stopCamera() {
    stoppedRef.current = true;
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    stopStream();
  }

  function cancel() {
    stopCamera();
    onCancel();
  }

  function choosePhoto() {
    stopCamera();
    onChoosePhoto();
  }

  const status = cameraStatus(phase, stableFramesRef.current);

  return (
    <section className={`live-camera is-${phase}`} aria-label="Live chess board camera">
      <video ref={videoRef} className="live-camera__video" muted playsInline />
      <div className="live-camera__shade" aria-hidden="true" />
      {detection?.found && (
        <BoardGridOverlay detection={detection} locked={phase === "locked"} />
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
        <span className="live-camera__live">Live</span>
      </header>
      {phase !== "error" && (
        <div className="live-camera__manual-controls">
          <button
            type="button"
            className="camera-shutter"
            disabled={phase === "requesting" || phase === "capturing" || phase === "locked"}
            onClick={() => void captureManually()}
          >
            <span aria-hidden="true"><i /></span>
            <strong>Capture now</strong>
          </button>
          <button type="button" className="camera-photo-button" onClick={choosePhoto}>
            Photos
          </button>
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
          <button type="button" className="camera-fallback-button" onClick={choosePhoto}>
            Choose a photo
          </button>
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
      {detection.corners.map(([x, y], index) => (
        <circle key={index} cx={x} cy={y} r={9} className="camera-grid__corner" />
      ))}
    </svg>
  );
}

async function frameBlob(video: HTMLVideoElement, maxDimension: number, quality: number) {
  const sourceWidth = video.videoWidth;
  const sourceHeight = video.videoHeight;
  if (!sourceWidth || !sourceHeight) throw new Error("Camera has no frame yet");
  const scale = Math.min(1, maxDimension / Math.max(sourceWidth, sourceHeight));
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(1, Math.round(sourceWidth * scale));
  canvas.height = Math.max(1, Math.round(sourceHeight * scale));
  const context = canvas.getContext("2d");
  if (!context) throw new Error("Canvas is unavailable");
  context.drawImage(video, 0, 0, canvas.width, canvas.height);
  return new Promise<Blob>((resolve, reject) => {
    canvas.toBlob(
      (blob) => (blob ? resolve(blob) : reject(new Error("Could not capture camera frame"))),
      "image/jpeg",
      quality,
    );
  });
}

function cornersAreStable(previous: Point[] | null, current: BoardDetection): boolean {
  if (!previous || previous.length !== 4 || current.corners.length !== 4) return false;
  const diagonal = Math.hypot(current.image_width, current.image_height);
  const maximumDrift = Math.max(
    ...current.corners.map(([x, y], index) => {
      const prior = previous[index];
      return prior ? Math.hypot(x - prior[0], y - prior[1]) : diagonal;
    }),
  );
  return maximumDrift / diagonal < 0.014;
}

function cameraStatus(phase: CameraPhase, stableFrames: number) {
  if (phase === "requesting") {
    return { title: "Opening camera", detail: "Camera permission is required once." };
  }
  if (phase === "framing") {
    return { title: "Move closer", detail: "Keep all four edges visible and fill the guide." };
  }
  if (phase === "aligning") {
    return {
      title: "Board identified",
      detail: stableFrames >= 2 ? "Almost there — hold steady." : "Squares found — hold steady.",
    };
  }
  if (phase === "capturing") {
    return { title: "Photo captured", detail: "You can adjust the four board corners next." };
  }
  if (phase === "locked") {
    return { title: "Board locked", detail: "64 squares are in place. Capturing now." };
  }
  if (phase === "error") {
    return { title: "Camera unavailable", detail: "Use an existing photograph instead." };
  }
  return { title: "Find the board", detail: "Fill the guide with one complete diagram." };
}

function boardFillRatio(detection: BoardDetection): number {
  const points = detection.corners;
  let twiceArea = 0;
  for (let index = 0; index < points.length; index += 1) {
    const current = points[index];
    const next = points[(index + 1) % points.length];
    if (!current || !next) continue;
    twiceArea += current[0] * next[1] - next[0] * current[1];
  }
  return Math.abs(twiceArea) / 2 / (detection.image_width * detection.image_height);
}

function pointString(points: Point[]): string {
  return points.map(([x, y]) => `${x},${y}`).join(" ");
}
