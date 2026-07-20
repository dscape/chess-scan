import type {
  BoardDetection,
  ConfirmResult,
  Orientation,
  Point,
  ScanResult,
  SideToMove,
} from "./types";

const SCAN_ID_PATTERN = /^[0-9a-f]{32}$/;

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

export function isScanId(value: string): boolean {
  return SCAN_ID_PATTERN.test(value);
}

export async function detectBoard(image: Blob): Promise<BoardDetection> {
  const form = new FormData();
  form.append("image", image, "camera-frame.jpg");
  return request<BoardDetection>("/api/detect-board", { method: "POST", body: form });
}

export async function scanImage(image: File): Promise<ScanResult> {
  const form = new FormData();
  form.append("image", image);
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

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) message = payload.detail;
    } catch {
      // Keep the status-based message when the response is not JSON.
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

function scanEndpoint(scanId: string, suffix = ""): string {
  if (!isScanId(scanId)) throw new Error("Invalid scan ID");
  return `/api/scans/${encodeURIComponent(scanId)}${suffix}`;
}
