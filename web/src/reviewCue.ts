import type { ReviewAnnotation } from "./types";

export interface BoardCue {
  id: string;
  ply: number;
}

export function displayCueLabel(label: string): string {
  return label.split("·").at(-1)?.trim() || label;
}

export function cueAccessibleLabel(cue: ReviewAnnotation): string {
  return `${cueRoleDescription(cue)}: ${displayCueLabel(cue.label)}. ${cue.text} Show on board.`;
}

export function cueRoleDescription(cue: ReviewAnnotation): string {
  switch (cue.arrows[0]?.role) {
    case "played":
      return "Learner move";
    case "reply":
      return "Hypothetical reply";
    case "engine":
      return "Engine line";
    default:
      return "Position idea";
  }
}

export function cueRoleMark(cue: ReviewAnnotation): string {
  switch (cue.arrows[0]?.role) {
    case "played":
      return "↗";
    case "reply":
      return "!";
    case "engine":
      return "✦";
    default:
      return "•";
  }
}

export function hasBoardCue(cue: ReviewAnnotation): boolean {
  return cue.markers.length > 0 || cue.arrows.length > 0 || cue.badge !== null;
}

export function displayedBoardCue<T extends BoardCue>(
  pinned: T | null,
  hovered: T | null,
  automatic: T | null,
  defaultCue: T | null,
  playedMoveCount: number,
): T | null {
  if (pinned) return pinned;
  if (playedMoveCount === 0 && hovered?.ply === 0) return hovered;
  if (hovered !== null && hovered.id !== automatic?.id) return null;
  if (automatic) return automatic;
  return playedMoveCount === 0 && defaultCue?.ply === 0 ? defaultCue : null;
}
