import type { Rect, Vec2 } from '@workbench/canvas-engine/types';

/** HUD presets only. Wheel zoom stays continuous because capture bands make slow scrolling stick at preset levels. */
export const ZOOM_PRESETS: readonly number[] = [0.25, 0.33, 0.5, 0.67, 0.75, 1, 1.25, 1.5, 2, 3, 4, 5];

/** Minimum allowed zoom, used by `clampZoom`. */
export const ZOOM_MIN = 0.1;

/** Maximum allowed zoom, used by `clampZoom`. */
export const ZOOM_MAX = 20;

/** Clamps `zoom` to `[ZOOM_MIN, ZOOM_MAX]`. */
export const clampZoom = (zoom: number): number => Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, zoom));

/** Snaps a single value to the nearest multiple of `grid`. Returns `value` unchanged if `grid <= 0`. */
export const snapToGrid = (value: number, grid: number): number => {
  if (grid <= 0) {
    return value;
  }
  return Math.round(value / grid) * grid;
};

/**
 * Snap the resulting absolute position so off-grid origins align. Preserve axes with zero delta, avoiding
 * unintended movement on locked axes.
 */
export const snapMovedPoint = (origin: Vec2, delta: Vec2, grid: number): Vec2 => ({
  x: delta.x === 0 ? origin.x : snapToGrid(origin.x + delta.x, grid),
  y: delta.y === 0 ? origin.y : snapToGrid(origin.y + delta.y, grid),
});

/** Snaps a rect's position and size to the nearest multiple of `grid`. */
export const snapRectToGrid = (rect: Rect, grid: number): Rect => {
  const x = snapToGrid(rect.x, grid);
  const y = snapToGrid(rect.y, grid);
  const right = snapToGrid(rect.x + rect.width, grid);
  const bottom = snapToGrid(rect.y + rect.height, grid);
  return { x, y, width: right - x, height: bottom - y };
};

/** The corner a corner-handle resize keeps fixed. */
export type AspectAnchor = 'nw' | 'ne' | 'sw' | 'se';
