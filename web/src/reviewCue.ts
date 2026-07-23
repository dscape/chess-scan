export interface BoardCue {
  id: string;
  ply: number;
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
