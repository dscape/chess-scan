import type {
  BoardDetection,
  CaptureGeometry,
  CoachingPresentationStatus,
  ConfirmResult,
  Orientation,
  Point,
  PositionAttemptRequest,
  PositionCoaching,
  PositionReview,
  PositionReviewRequest,
  ReviewAttempt,
  ReviewedPosition,
  ScanResult,
  SideToMove,
} from "./types";

const RECORD_ID_PATTERN = /^[0-9a-f]{32}$/;
const COACHING_RETRY_WINDOW_MS = 20_000;

export type ApiFailure = {
  kind: "api";
  message: string;
  status: number | null;
  retryable: boolean;
};

export class ApiError extends Error {
  readonly status: number;
  readonly retryAfterMs: number | null;

  constructor(message: string, status: number, retryAfterMs: number | null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.retryAfterMs = retryAfterMs;
  }
}

export function apiFailureFrom(cause: unknown): ApiFailure | null {
  if (cause instanceof ApiError) {
    return {
      kind: "api",
      message: cause.message,
      status: cause.status,
      retryable: cause.status === 408
        || cause.status === 429
        || cause.status >= 500,
    };
  }
  if (cause instanceof TypeError) {
    return {
      kind: "api",
      message: cause.message || "The review service could not be reached.",
      status: null,
      retryable: true,
    };
  }
  return null;
}

function isRecordId(value: string): boolean {
  return RECORD_ID_PATTERN.test(value);
}

export const isScanId = isRecordId;
export const isFeedbackId = isRecordId;

export async function detectBoard(image: Blob): Promise<BoardDetection> {
  const form = new FormData();
  form.append("image", image, "camera-frame.jpg");
  return request<BoardDetection>("/api/detect-board", { method: "POST", body: form });
}

export async function scanImage(
  image: File,
  geometry?: CaptureGeometry,
): Promise<ScanResult> {
  const form = new FormData();
  form.append("image", image);
  if (geometry) {
    form.append("corners", JSON.stringify(geometry.corners));
    form.append("detection_method", geometry.method);
  }
  return request<ScanResult>("/api/scans", { method: "POST", body: form });
}

export async function getScan(scanId: string, signal?: AbortSignal): Promise<ScanResult> {
  return request<ScanResult>(scanEndpoint(scanId), { signal });
}

export async function reprocessScan(scanId: string, corners: Point[]): Promise<ScanResult> {
  return request<ScanResult>(scanEndpoint(scanId, "/reprocess"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ corners }),
  });
}

export async function confirmScan(
  scanId: string,
  payload: {
    labels: number[];
    orientation: Orientation;
    side_to_move: SideToMove;
    castling: string;
    en_passant: string;
    consent_training: boolean;
    client_session_id: string;
  },
): Promise<ConfirmResult> {
  return request<ConfirmResult>(scanEndpoint(scanId, "/confirm"), {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function getReviewedPosition(
  feedbackId: string,
  signal?: AbortSignal,
): Promise<ReviewedPosition> {
  return request<ReviewedPosition>(recordEndpoint("/api/reviews", feedbackId, "feedback"), {
    signal,
  });
}

export async function comparePositionAttempt(
  payload: PositionAttemptRequest,
  signal?: AbortSignal,
): Promise<ReviewAttempt> {
  return request<ReviewAttempt>("/api/position-attempts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
}

export async function createPositionReview(
  payload: PositionReviewRequest,
  signal?: AbortSignal,
): Promise<PositionReview> {
  return request<PositionReview>("/api/position-reviews", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });
}

export async function createPositionCoaching(
  reviewId: string,
  signal?: AbortSignal,
): Promise<PositionCoaching> {
  const endpoint = positionReviewEndpoint(reviewId, "/coaching");
  const retryDeadline = Date.now() + COACHING_RETRY_WINDOW_MS;
  for (let attempt = 0; ; attempt += 1) {
    try {
      return await request<PositionCoaching>(endpoint, { method: "POST", signal });
    } catch (cause) {
      if (!(cause instanceof ApiError) || cause.status !== 503) throw cause;
      const exponentialDelay = Math.min(4000, 500 * 2 ** attempt);
      const minimumDelay = cause.retryAfterMs ?? 0;
      const jitter = Math.floor(Math.random() * 250);
      const delay = Math.max(exponentialDelay, minimumDelay) + jitter;
      if (Date.now() + delay > retryDeadline) throw cause;
      await abortableDelay(delay, signal);
    }
  }
}

export async function ratePositionReview(
  reviewId: string,
  payload: {
    rating: "helpful" | "unhelpful";
    reason: "correct" | "incorrect_chess" | "irrelevant_topic" | "unclear"
      | "equivalent_move_rejected" | "too_verbose" | "missing_detail" | "other";
    detail?: string;
    coaching_status: CoachingPresentationStatus;
    commentary_run_id: string | null;
  },
): Promise<{ feedback_id: string }> {
  return request<{ feedback_id: string }>(
    positionReviewEndpoint(reviewId, "/feedback"),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    },
  );
}

async function abortableDelay(milliseconds: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted) throw new DOMException("The request was aborted.", "AbortError");
  await new Promise<void>((resolve, reject) => {
    const aborted = () => {
      window.clearTimeout(timeout);
      reject(new DOMException("The request was aborted.", "AbortError"));
    };
    const timeout = window.setTimeout(() => {
      signal?.removeEventListener("abort", aborted);
      resolve();
    }, milliseconds);
    signal?.addEventListener("abort", aborted, { once: true });
  });
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: unknown };
      const detail = apiErrorDetail(payload.detail);
      if (detail) message = detail;
    } catch {
      // Keep the status-based message when the response is not JSON.
    }
    throw new ApiError(
      message,
      response.status,
      retryAfterMilliseconds(response.headers.get("Retry-After")),
    );
  }
  return (await response.json()) as T;
}

function retryAfterMilliseconds(value: string | null): number | null {
  if (value === null) return null;
  const seconds = Number(value);
  if (Number.isFinite(seconds) && seconds >= 0) return Math.min(30_000, seconds * 1000);
  const date = Date.parse(value);
  if (Number.isNaN(date)) return null;
  return Math.min(30_000, Math.max(0, date - Date.now()));
}

function apiErrorDetail(detail: unknown): string | null {
  if (typeof detail === "string") return detail;
  if (!Array.isArray(detail)) return null;
  const messages = detail.flatMap((item) => {
    if (!item || typeof item !== "object") return [];
    const message = (item as Record<string, unknown>).msg;
    return typeof message === "string" ? [message] : [];
  });
  return messages.length > 0 ? messages.join(" ") : null;
}

function recordEndpoint(
  collection: string,
  recordId: string,
  label: string,
  suffix = "",
): string {
  if (!isRecordId(recordId)) throw new Error(`Invalid ${label} ID`);
  return `${collection}/${encodeURIComponent(recordId)}${suffix}`;
}

function scanEndpoint(scanId: string, suffix = ""): string {
  return recordEndpoint("/api/scans", scanId, "scan", suffix);
}

function positionReviewEndpoint(reviewId: string, suffix = ""): string {
  return recordEndpoint("/api/position-reviews", reviewId, "review", suffix);
}
