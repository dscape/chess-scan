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

export interface ReviewMove {
  uci: string;
  san: string;
}

export interface ReviewArrow {
  from_square: string;
  to_square: string;
  kind: "move" | "idea";
}

export interface ReviewAnnotation {
  label: string;
  text: string;
  squares: string[];
  arrows: ReviewArrow[];
}

export interface PositionTopic {
  id: string;
  name: string;
}

export interface PositionReview {
  fen: string;
  engine: string;
  evaluation: string;
  score: (EngineScore & { bound: NonNullable<EngineScore["bound"]> | null }) | null;
  best_move: ReviewMove | null;
  topic: PositionTopic;
  hint: ReviewAnnotation;
  explanation: ReviewAnnotation[];
}

export interface PositionReviewRequest {
  fen: string;
  line: EngineLine | null;
}
