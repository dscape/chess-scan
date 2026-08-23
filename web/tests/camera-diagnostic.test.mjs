import assert from "node:assert/strict";
import test from "node:test";

import {
  diagnosticRecordingSupported,
  startDiagnosticRecording,
} from "../src/cameraDiagnostic.ts";

test("detects Safari-style H.264 diagnostic recording support", () => {
  const original = globalThis.MediaRecorder;
  class SafariMediaRecorder {
    static isTypeSupported(type) {
      return type === "video/mp4;codecs=avc1.42000a";
    }
  }
  globalThis.MediaRecorder = SafariMediaRecorder;
  try {
    assert.equal(diagnosticRecordingSupported(), true);
  } finally {
    if (original === undefined) delete globalThis.MediaRecorder;
    else globalThis.MediaRecorder = original;
  }
});

test("falls back when an advertised MP4 recorder cannot start", () => {
  const originals = {
    MediaRecorder: globalThis.MediaRecorder,
    MediaStream: globalThis.MediaStream,
    window: globalThis.window,
  };
  class FallbackMediaRecorder extends EventTarget {
    static isTypeSupported(type) {
      return type.startsWith("video/mp4") || type === "video/webm";
    }

    constructor(_stream, options) {
      super();
      this.mimeType = options.mimeType;
      this.state = "inactive";
    }

    start() {
      if (this.mimeType.startsWith("video/mp4")) {
        throw new DOMException("MP4 failed to start", "NotSupportedError");
      }
      this.state = "recording";
    }

    stop() {
      this.state = "inactive";
      this.dispatchEvent(new Event("stop"));
    }
  }
  globalThis.MediaRecorder = FallbackMediaRecorder;
  globalThis.MediaStream = class {
    constructor(tracks) {
      this.tracks = tracks;
    }
  };
  globalThis.window = globalThis;
  try {
    const stream = {
      getVideoTracks: () => [{ getSettings: () => ({ width: 1280, height: 720 }) }],
    };
    const recording = startDiagnosticRecording(stream);
    assert.ok(recording);
    recording.discard();
  } finally {
    for (const [key, value] of Object.entries(originals)) {
      if (value === undefined) delete globalThis[key];
      else globalThis[key] = value;
    }
  }
});

test("rejects a MediaRecorder without a supported video container", () => {
  const original = globalThis.MediaRecorder;
  class UnsupportedMediaRecorder {
    static isTypeSupported() {
      return false;
    }
  }
  globalThis.MediaRecorder = UnsupportedMediaRecorder;
  try {
    assert.equal(diagnosticRecordingSupported(), false);
  } finally {
    if (original === undefined) delete globalThis.MediaRecorder;
    else globalThis.MediaRecorder = original;
  }
});
