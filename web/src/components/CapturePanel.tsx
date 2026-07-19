import { useRef, useState } from "react";
import type { LearningStatus } from "../types";
import LiveCamera from "./LiveCamera";

interface CapturePanelProps {
  busy: boolean;
  status: LearningStatus | null;
  onImage: (file: File) => void;
}

export default function CapturePanel({
  busy,
  status,
  onImage,
}: CapturePanelProps) {
  const libraryInputRef = useRef<HTMLInputElement>(null);
  const [cameraOpen, setCameraOpen] = useState(false);

  function handleFile(file: File | undefined, input: HTMLInputElement) {
    if (file) onImage(file);
    input.value = "";
  }

  if (cameraOpen) {
    return (
      <main className="camera-shell">
        <LiveCamera
          onCapture={(file) => {
            setCameraOpen(false);
            onImage(file);
          }}
          onCancel={() => setCameraOpen(false)}
          onChoosePhoto={() => {
            setCameraOpen(false);
            window.setTimeout(() => libraryInputRef.current?.click(), 0);
          }}
        />
        <input
          ref={libraryInputRef}
          className="visually-hidden"
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={(event) =>
            handleFile(event.target.files?.[0], event.currentTarget)
          }
        />
      </main>
    );
  }

  return (
    <main className="capture-shell">
      <section className="capture-intro">
        <p className="eyebrow">Workbook position → Lichess</p>
        <h1>
          Read the board.
          <br />
          Check the pieces.
          <br />
          Keep learning.
        </h1>
        <p className="capture-intro__copy">
          Frame one Chess Steps diagram as if it were a QR code. We turn it into
          a position; you make the final call.
        </p>
      </section>

      <section className={`capture-card ${busy ? "is-scanning" : ""}`}>
        <div className="capture-card__target" aria-hidden="true">
          <div className="capture-card__grid" />
          {busy && <div className="scan-line" />}
        </div>
        <div className="capture-card__content">
          <span className="step-number">01</span>
          <div>
            <h2>{busy ? "Reading the diagram…" : "Photograph one board"}</h2>
            <p>
              Fill the frame, keep the page flat, and include the black turn dot
              if present.
            </p>
          </div>
        </div>
        <button
          type="button"
          className="primary-button capture-button"
          disabled={busy}
          onClick={() => setCameraOpen(true)}
        >
          <span>{busy ? "Scanning" : "Open camera"}</span>
        </button>
        <button
          type="button"
          className="text-button capture-library-button"
          disabled={busy}
          onClick={() => libraryInputRef.current?.click()}
        >
          Choose an existing photo instead
        </button>
        <input
          ref={libraryInputRef}
          className="visually-hidden"
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={(event) =>
            handleFile(event.target.files?.[0], event.currentTarget)
          }
        />
      </section>

      <aside className="learning-note">
        <span className="learning-note__mark">↻</span>
        <p>
          <strong>{learningProgress(status)}</strong> The learner trains,
          tests, and activates a model only when it beats the current one.
        </p>
      </aside>
    </main>
  );
}

function learningProgress(status: LearningStatus | null): string {
  if (!status) return "Automatic learning is starting.";
  if (status.learning_state === "training") return "A new model is training.";
  if (status.learning_state === "benchmarking") return "A new model is being tested.";
  if (status.learning_state === "shadowing") {
    return `A candidate has ${status.learning_progress}/${status.learning_target} fresh checks.`;
  }
  return `${status.learning_progress}/${status.learning_target} boards collected for the next model.`;
}
