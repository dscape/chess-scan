const MAX_DIAGNOSTIC_DURATION_MS = 10_000;
const DIAGNOSTIC_VIDEO_BITS_PER_SECOND = 700_000;

export interface RecordedDiagnostic {
  blob: Blob;
  durationMs: number;
  settings: MediaTrackSettings;
}

export interface DiagnosticRecording {
  stop: () => Promise<RecordedDiagnostic | null>;
  discard: () => void;
}

export function diagnosticRecordingSupported(): boolean {
  return diagnosticMimeTypes().length > 0;
}

export function startDiagnosticRecording(stream: MediaStream): DiagnosticRecording | null {
  for (const mimeType of diagnosticMimeTypes()) {
    const recording = startRecordingWithMimeType(stream, mimeType);
    if (recording) return recording;
  }
  return null;
}

function startRecordingWithMimeType(
  stream: MediaStream,
  mimeType: string,
): DiagnosticRecording | null {
  try {
    const startedAt = performance.now();
    const settings = stream.getVideoTracks()[0]?.getSettings() ?? {};
    const recordingStream = new MediaStream(stream.getVideoTracks());
    const recorder = createMediaRecorder(recordingStream, mimeType);
    const chunks: Blob[] = [];
    let discarded = false;
    let resolveResult: (value: RecordedDiagnostic | null) => void = () => undefined;
    const result = new Promise<RecordedDiagnostic | null>((resolve) => {
      resolveResult = resolve;
    });
    let timeout = 0;
    const finish = () => {
      window.clearTimeout(timeout);
      if (discarded || chunks.length === 0) {
        resolveResult(null);
        return;
      }
      const type = recorder.mimeType || mimeType;
      resolveResult({
        blob: new Blob(chunks, { type }),
        durationMs: Math.min(
          MAX_DIAGNOSTIC_DURATION_MS,
          Math.max(0, Math.round(performance.now() - startedAt)),
        ),
        settings,
      });
    };
    recorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) chunks.push(event.data);
    });
    recorder.addEventListener("stop", finish, { once: true });
    recorder.addEventListener("error", () => resolveResult(null), { once: true });
    recorder.start(250);
    timeout = window.setTimeout(() => {
      if (recorder.state === "inactive") return;
      try {
        recorder.stop();
      } catch {
        resolveResult(null);
      }
    }, MAX_DIAGNOSTIC_DURATION_MS);
    return {
      stop: async () => {
        if (recorder.state !== "inactive") {
          try {
            recorder.requestData();
          } catch {
            // Some mobile recorders only flush on stop.
          }
          try {
            recorder.stop();
          } catch {
            resolveResult(null);
          }
        }
        return resultWithTimeout(result);
      },
      discard: () => {
        discarded = true;
        if (recorder.state === "inactive") return;
        try {
          recorder.stop();
        } catch {
          resolveResult(null);
        }
      },
    };
  } catch {
    return null;
  }
}

export function finiteMediaSetting(value: number | undefined): number | null {
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : null;
}

function createMediaRecorder(stream: MediaStream, mimeType: string): MediaRecorder {
  try {
    return new MediaRecorder(stream, {
      mimeType,
      videoBitsPerSecond: DIAGNOSTIC_VIDEO_BITS_PER_SECOND,
    });
  } catch {
    return new MediaRecorder(stream, { mimeType });
  }
}

function resultWithTimeout(
  result: Promise<RecordedDiagnostic | null>,
): Promise<RecordedDiagnostic | null> {
  return new Promise((resolve) => {
    const timeout = window.setTimeout(() => resolve(null), 3000);
    void result.then((recorded) => {
      window.clearTimeout(timeout);
      resolve(recorded);
    });
  });
}

function diagnosticMimeTypes(): string[] {
  if (
    typeof MediaRecorder === "undefined"
    || typeof MediaRecorder.isTypeSupported !== "function"
  ) return [];
  const candidates = [
    "video/mp4;codecs=avc1.42000a",
    "video/mp4;codecs=avc1",
    "video/webm;codecs=vp8",
    "video/webm",
    "video/mp4",
  ];
  return candidates.filter((type) => MediaRecorder.isTypeSupported(type));
}
