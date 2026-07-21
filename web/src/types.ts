import type { EngineLine, EngineScore } from "./engine/uci";

export type Point = [number, number];
export type Orientation = "white" | "black";
export type SideToMove = "w" | "b";

export interface BoardDetection {
  found: boolean;
  confidence: number;
  method: string;
  image_width: number;
  image_height: number;
  corners: Point[];
  grid_points: Point[];
}

export interface ScanResult {
  scan_id: string;
  source_width: number;
  source_height: number;
  corners: Point[];
  detection_method: string;
  labels: number[];
  probabilities: number[][];
  confidences: number[];
  board_fen: string;
  model_version: string;
  prediction_revision: string;
  source_image_url: string;
  rectified_image_url: string;
}

export interface ConfirmResult {
  feedback_id: string;
  full_fen: string;
  lichess_url: string;
  changed_squares: number;
  warnings: string[];
}

export interface ReviewedPosition {
  feedback_id: string;
  full_fen: string;
  orientation: Orientation;
  changed_squares: number;
  lichess_url: string;
}

export type ReviewMode = "general" | "mix" | "thinking_ahead";

export interface ReviewEvidence {
  kind: string;
  summary: string;
  squares: string[];
  moves: string[];
}

export interface ReviewFinding {
  topic_id: string;
  topic: string;
  level: number;
  confidence: number;
  evidence: ReviewEvidence[];
}

export interface ReviewMove {
  uci: string;
  san: string;
}

export interface ReviewedLine {
  multipv: number;
  depth: number;
  score: EngineScore & { bound: NonNullable<EngineScore["bound"]> | null };
  wdl: [number, number, number] | null;
  moves: ReviewMove[];
}

export interface PositionReview {
  fen: string;
  engine: string;
  evaluation: string;
  best_move: ReviewMove | null;
  lines: ReviewedLine[];
  primary_finding: ReviewFinding | null;
  findings: ReviewFinding[];
  explanation: string;
  verbalizer: "mock";
}

export interface PositionReviewRequest {
  fen: string;
  study_level: number;
  mode: ReviewMode;
  lines: EngineLine[];
}
