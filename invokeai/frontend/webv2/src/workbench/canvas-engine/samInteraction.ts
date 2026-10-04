import type { Rect, Vec2 } from '@workbench/canvas-engine/types';

/** Core interaction-only state for point and bounding-box manipulation. */
export interface SamVisualInput {
  type: 'visual';
  includePoints: Vec2[];
  excludePoints: Vec2[];
  bbox: Rect | null;
}

export interface SamInteractionState {
  input: SamVisualInput;
  pointLabel: 'include' | 'exclude';
  sourceRect: Rect;
}
