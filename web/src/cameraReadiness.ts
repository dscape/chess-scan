export type AutoCaptureMode = "quick" | "balanced" | "precise";

export interface AutoCaptureProfile {
  mode: AutoCaptureMode;
  label: string;
  threshold: number;
  detail: string;
}

export interface ScanReadinessInput {
  boardSizeRatio: number;
  geometryStability: number;
  predictionAgreement: number | null;
  sharpnessScore: number;
  glareScore: number;
  detectionConfidence: number;
  predictionConfidence: number;
  stableRounds: number;
}

interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

export const AUTO_CAPTURE_PROFILES: readonly AutoCaptureProfile[] = [
  {
    mode: "quick",
    label: "Quick",
    threshold: 70,
    detail: "Captures sooner",
  },
  {
    mode: "balanced",
    label: "Balanced",
    threshold: 80,
    detail: "Recommended",
  },
  {
    mode: "precise",
    label: "Precise",
    threshold: 90,
    detail: "Waits for more agreement",
  },
];

export const DEFAULT_AUTO_CAPTURE_MODE: AutoCaptureMode = "balanced";
const STORAGE_KEY = "chess-scan:auto-capture-mode:v1";

export function autoCaptureProfile(mode: AutoCaptureMode): AutoCaptureProfile {
  return AUTO_CAPTURE_PROFILES.find((profile) => profile.mode === mode)
    ?? AUTO_CAPTURE_PROFILES[1]!;
}

export function automaticCaptureReady(
  readiness: number,
  profile: AutoCaptureProfile,
  stableRounds: number,
  boardSizeRatio: number,
  hasPrediction: boolean,
): boolean {
  return hasPrediction
    && stableRounds >= 2
    && boardSizeRatio >= 1
    && readiness >= profile.threshold;
}

export function scanReadiness(input: ScanReadinessInput): number {
  const boardSize = clamp(input.boardSizeRatio);
  const geometry = clamp(input.geometryStability);
  const agreement = clamp(input.predictionAgreement ?? 0.35);
  const focus = clamp((input.sharpnessScore - 5) / 24);
  const detector = clamp((input.detectionConfidence - 0.3) / 0.65);
  const prediction = clamp((input.predictionConfidence - 0.45) / 0.45);
  const glare = 1 - clamp(input.glareScore * 5);
  const settled = clamp(input.stableRounds / 3);
  const weighted = 100 * (
    0.18 * boardSize
    + 0.12 * geometry
    + 0.25 * agreement
    + 0.08 * focus
    + 0.06 * detector
    + 0.05 * prediction
    + 0.04 * glare
    + 0.22 * settled
  );
  const sizeLimited = input.boardSizeRatio < 1
    ? Math.min(weighted, 64 * clamp(input.boardSizeRatio))
    : weighted;
  return Math.round(clamp(sizeLimited / 100) * 100);
}

export function smoothScanReadiness(
  previous: number,
  measured: number | null,
): number {
  if (measured === null) return Math.round(clamp(previous / 100) * 82);
  const rate = measured >= previous ? 0.68 : 0.38;
  return Math.round(previous + (measured - previous) * rate);
}

export function labelAgreement(previous: number[] | null, current: number[]): number | null {
  if (!previous || previous.length !== 64 || current.length !== 64) return null;
  const matches = current.reduce(
    (count, label, index) => count + (label === previous[index] ? 1 : 0),
    0,
  );
  return matches / 64;
}

export function meanPredictionConfidence(confidences: number[]): number {
  if (confidences.length === 0) return 0;
  return confidences.reduce((sum, confidence) => sum + clamp(confidence), 0)
    / confidences.length;
}

export function loadAutoCaptureMode(storage: StorageLike | null): AutoCaptureMode {
  if (!storage) return DEFAULT_AUTO_CAPTURE_MODE;
  try {
    const stored = storage.getItem(STORAGE_KEY);
    return AUTO_CAPTURE_PROFILES.some((profile) => profile.mode === stored)
      ? stored as AutoCaptureMode
      : DEFAULT_AUTO_CAPTURE_MODE;
  } catch {
    return DEFAULT_AUTO_CAPTURE_MODE;
  }
}

export function saveAutoCaptureMode(
  storage: StorageLike | null,
  mode: AutoCaptureMode,
): void {
  try {
    storage?.setItem(STORAGE_KEY, mode);
  } catch {
    // Storage can be blocked in private or embedded browser contexts.
  }
}

function clamp(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.min(1, Math.max(0, value));
}
