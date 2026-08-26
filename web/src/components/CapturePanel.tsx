import { useState } from "react";
import { diagnosticRecordingSupported } from "../cameraDiagnostic";
import {
  loadDataSharingConsent,
  saveDataSharingConsent,
} from "../dataSharing";
import type { CameraCapture } from "../types";
import LiveCamera from "./LiveCamera";
import PhotoPicker from "./PhotoPicker";

interface CapturePanelProps {
  busy: boolean;
  captureDiagnosticsEnabled: boolean;
  diagnosticRetentionDays: number;
  onCapture: (capture: CameraCapture) => Promise<boolean>;
}

export default function CapturePanel({
  busy,
  captureDiagnosticsEnabled,
  diagnosticRetentionDays,
  onCapture,
}: CapturePanelProps) {
  const [cameraOpen, setCameraOpen] = useState(false);
  const [dataSharingEnabled, setDataSharingEnabled] = useState(() =>
    loadDataSharingConsent(browserStorage()),
  );
  const diagnosticAvailable = captureDiagnosticsEnabled && diagnosticRecordingSupported();

  function updateDataSharing(enabled: boolean) {
    setDataSharingEnabled(enabled);
    saveDataSharingConsent(browserStorage(), enabled);
  }

  if (cameraOpen) {
    return (
      <main className="camera-shell">
        <LiveCamera
          dataSharingEnabled={dataSharingEnabled}
          diagnosticsAvailable={diagnosticAvailable}
          diagnosticRetentionDays={diagnosticRetentionDays}
          onDataSharingChange={updateDataSharing}
          onCapture={async (capture) => {
            const accepted = await onCapture(capture);
            if (accepted) setCameraOpen(false);
            return accepted;
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
          onPhoto={(file) => void onCapture({
            image: file,
            supplementalImages: [],
            consentTraining: dataSharingEnabled,
            diagnostic: Promise.resolve(null),
            diagnosticRequested: false,
          })}
        >
          Choose an existing photo instead
        </PhotoPicker>
      </section>
    </main>
  );
}

function browserStorage(): Storage | null {
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}
