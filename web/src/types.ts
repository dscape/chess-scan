import type { EngineScore } from "./engine/uci";

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

export interface CaptureGeometry {
  corners: Point[];
  method: string;
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

export type ReviewArrowRole = "played" | "engine" | "reply" | "attack" | "ray" | "threat";
export type ReviewBadge =
  | "fork"
  | "pin"
  | "xray"
  | "trap"
  | "capture"
  | "clearance"
  | "discovery"
  | "interference"
  | "attraction"
  | "intermezzo"
  | "mate"
  | "engine";
export type ReviewMarkerRole = "focus" | "target" | "danger" | "vacated" | "blocked";

export interface ReviewArrow {
  from_square: string;
  to_square: string;
  role: ReviewArrowRole;
}

export interface ReviewSquareMarker {
  square: string;
  role: ReviewMarkerRole;
}

export interface ReviewDiagramBadge {
  kind: ReviewBadge;
  square: string;
  role: ReviewArrowRole;
  arrow_index: number;
}

export interface ReviewAnnotation {
  label: string;
  text: string;
  scope: "root" | "best_line" | "attempt_line" | "attempt_refutation" | "terminal";
  ply: number;
  markers: ReviewSquareMarker[];
  arrows: ReviewArrow[];
  badge: ReviewDiagramBadge | null;
  evidence_ids: string[];
}

export interface PositionTopic {
  id: string;
  name: string;
}

export interface ReviewPieceRef {
  color: "white" | "black";
  piece: string;
  square: string;
}

export interface ReviewEvidence {
  id: string;
  kind: string;
  scope: "best_line" | "attempt_line" | "attempt_refutation" | "terminal";
  proof: "legal_geometry" | "line_consequence" | "direct_rule";
  ply: number;
  actor: ReviewPieceRef | null;
  targets: ReviewPieceRef[];
  from_square: string | null;
  to_square: string | null;
  squares: string[];
  moves: string[];
  score: (EngineScore & { bound: NonNullable<EngineScore["bound"]> | null }) | null;
  wdl: [number, number, number] | null;
  expected_score_loss: number | null;
  centipawn_loss: number | null;
  lost_forced_mate: boolean | null;
  mate_delay: number | null;
  verdict: string | null;
}

export interface ReviewLine {
  role: "best_candidate" | "alternative_candidate" | "attempt_line" | "attempt_refutation";
  rank: number;
  depth: number;
  score: EngineScore & { bound: NonNullable<EngineScore["bound"]> | null };
  wdl: [number, number, number];
  moves: ReviewMove[];
}

export interface ReviewAttempt {
  move: ReviewMove;
  verdict: "best" | "excellent" | "good" | "inaccuracy" | "mistake" | "blunder";
  equivalent: boolean;
  expected_score_loss: number;
  centipawn_loss: number | null;
  lost_forced_mate: boolean;
  mate_delay: number | null;
  line: ReviewLine;
}

export interface PositionReview {
  schema_version: "position-analysis-4";
  review_id: string | null;
  fen: string;
  engine: string;
  evaluation: string;
  score: (EngineScore & { bound: NonNullable<EngineScore["bound"]> | null }) | null;
  score_pov: "side_to_move" | null;
  best_move: ReviewMove | null;
  lines: ReviewLine[];
  attempt: ReviewAttempt | null;
  topic: PositionTopic;
  findings: Array<{ topic: PositionTopic; evidence_ids: string[] }>;
  evidence: ReviewEvidence[];
  hint: ReviewAnnotation;
  explanation: ReviewAnnotation[];
}

export interface ReviewEngineLine {
  rank: number;
  depth: number;
  score: EngineScore;
  wdl: [number, number, number];
  pv: string[];
  stable: boolean;
}

export interface ReviewAnalysis {
  score_pov: "side_to_move";
  lines: ReviewEngineLine[];
  attempt?: {
    move: string;
    line: ReviewEngineLine;
  } | null;
}

export interface PositionReviewRequest {
  fen: string;
  feedback_id: string;
  analysis: ReviewAnalysis | null;
}
