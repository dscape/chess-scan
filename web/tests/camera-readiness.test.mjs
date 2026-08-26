import assert from "node:assert/strict";
import test from "node:test";

import {
  autoCaptureProfile,
  automaticCaptureReady,
  labelAgreement,
  loadAutoCaptureMode,
  meanPredictionConfidence,
  saveAutoCaptureMode,
  scanReadiness,
  smoothScanReadiness,
} from "../src/cameraReadiness.ts";

test("combines stable geometry, focus, and temporal agreement into readiness", () => {
  const poor = scanReadiness({
    boardSizeRatio: 0.7,
    geometryStability: 0.1,
    predictionAgreement: 0.5,
    sharpnessScore: 9,
    glareScore: 0.12,
    detectionConfidence: 0.4,
    predictionConfidence: 0.6,
    stableRounds: 1,
  });
  const ready = scanReadiness({
    boardSizeRatio: 1.2,
    geometryStability: 0.98,
    predictionAgreement: 0.98,
    sharpnessScore: 42,
    glareScore: 0,
    detectionConfidence: 0.8,
    predictionConfidence: 0.9,
    stableRounds: 4,
  });

  assert.ok(poor < autoCaptureProfile("quick").threshold);
  assert.ok(ready >= autoCaptureProfile("precise").threshold);
});

test("auto-captures only after size, stability, prediction, and threshold gates", () => {
  const profile = autoCaptureProfile("balanced");

  assert.equal(automaticCaptureReady(80, profile, 2, 1, true), true);
  assert.equal(automaticCaptureReady(79, profile, 2, 1, true), false);
  assert.equal(automaticCaptureReady(90, profile, 1, 1, true), false);
  assert.equal(automaticCaptureReady(90, profile, 2, 0.99, true), false);
  assert.equal(automaticCaptureReady(90, profile, 2, 1, false), false);
});

test("limits readiness until the board fills the retained source", () => {
  const score = scanReadiness({
    boardSizeRatio: 0.5,
    geometryStability: 1,
    predictionAgreement: 1,
    sharpnessScore: 100,
    glareScore: 0,
    detectionConfidence: 1,
    predictionConfidence: 1,
    stableRounds: 10,
  });

  assert.equal(score, 32);
});

test("smooths progress upward and decays instead of flashing to zero", () => {
  const first = smoothScanReadiness(0, 48);
  const second = smoothScanReadiness(first, 88);
  const missed = smoothScanReadiness(second, null);

  assert.equal(first, 33);
  assert.equal(second, 70);
  assert.equal(missed, 57);
});

test("measures temporal labels and bounded model confidence", () => {
  const labels = Array.from({ length: 64 }, (_, index) => index % 13);
  const changed = labels.slice();
  changed[0] = 12;
  changed[1] = 12;

  assert.equal(labelAgreement(null, labels), null);
  assert.equal(labelAgreement(labels, changed), 62 / 64);
  assert.equal(meanPredictionConfidence([0.5, 0.75, 2]), 0.75);
});

test("persists auto-capture mode and tolerates blocked storage", () => {
  const values = new Map();
  const storage = {
    getItem: (key) => values.get(key) ?? null,
    setItem: (key, value) => values.set(key, value),
  };
  const blocked = {
    getItem: () => { throw new Error("blocked"); },
    setItem: () => { throw new Error("blocked"); },
  };

  assert.equal(loadAutoCaptureMode(storage), "balanced");
  saveAutoCaptureMode(storage, "precise");
  assert.equal(loadAutoCaptureMode(storage), "precise");
  assert.equal(loadAutoCaptureMode(blocked), "balanced");
  assert.doesNotThrow(() => saveAutoCaptureMode(blocked, "quick"));
});
