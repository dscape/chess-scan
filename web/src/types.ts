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

export interface LearningStatus {
  confirmed_boards: number;
  corrected_boards: number;
  training_boards: number;
  active_model: string;
  learning_state: "collecting" | "training" | "benchmarking" | "shadowing";
  learning_progress: number;
  learning_target: number;
  candidate_model: string | null;
}
