import { useState } from "react";
import LiveCamera from "./LiveCamera";
import PhotoPicker from "./PhotoPicker";

interface CapturePanelProps {
  busy: boolean;
  onImage: (file: File) => void;
}

export default function CapturePanel({ busy, onImage }: CapturePanelProps) {
  const [cameraOpen, setCameraOpen] = useState(false);

  if (cameraOpen) {
    return (
      <main className="camera-shell">
        <LiveCamera
          onCapture={(file) => {
            setCameraOpen(false);
            onImage(file);
          }}
          onCancel={() => setCameraOpen(false)}
        />
      </main>
    );
  }

  return (
    <main className="capture-shell">
      <section className={`capture-card ${busy ? "is-scanning" : ""}`}>
        <div className="capture-card__brand">
          <div className="brand">
            <span className="brand__mark" aria-hidden="true">
              <i />
              <i />
              <i />
              <i />
            </span>
            <span>
              <strong>Chess</strong>
              <em>Scan</em>
            </span>
          </div>
        </div>
        <div className="capture-card__target" aria-hidden="true">
          <div className="capture-card__grid" />
          {busy && <div className="scan-line" />}
        </div>
        <div className="capture-card__content">
          <span className="step-number">01</span>
          <div>
            <h1>{busy ? "Reading the diagram…" : "Photograph one board"}</h1>
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
        <PhotoPicker
          className="text-button capture-library-button"
          disabled={busy}
          onPhoto={onImage}
        >
          Choose an existing photo instead
        </PhotoPicker>
      </section>
    </main>
  );
}
