/**
 * Pure snapping/clamping helpers for zoom, grid, and aspect-ratio math.
 *
 * No classes, no mutation — every function returns a new value.
 */

import type { Rect, Vec2 } from '@workbench/canvas-engine/types';

/**
 * The zoom levels the HUD dropdown offers. Presets only — wheel zoom is
 * continuous and deliberately does NOT snap to these: a ~3% capture band around
 * every level meant a slow scroll stalled at each one and had to be pushed out,
 * which reads as the zoom sticking rather than as a helpful detent.
 */
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
 * The snapped absolute position for a drag of `delta` away from `origin`.
 *
 * Snaps the RESULT, not the delta, so something that starts off-grid seats onto
 * the grid instead of carrying its misalignment along forever.
 *
 * Per axis, and only for axes the drag actually moved: an axis with a zero delta
 * (shift locked it, or the pointer simply never moved along it) is passed through
 * untouched rather than being pulled onto a grid line the user never crossed.
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

/** The corner or center kept fixed when `constrainAspect` resizes a rect. */
export type AspectAnchor = 'nw' | 'ne' | 'sw' | 'se' | 'center';

/**
 * Resizes `rect` to match `aspect` (width / height), keeping the named
 * anchor point fixed. The rect's area is preserved as closely as possible
 * by scaling both dimensions from the current size to fit the target
 * aspect ratio (the dimension that would grow beyond the current bounding
 * box is chosen based on which axis needs less change).
 */
export const constrainAspect = (rect: Rect, aspect: number, anchor: AspectAnchor): Rect => {
  if (aspect <= 0 || rect.width <= 0 || rect.height <= 0) {
    return rect;
  }

  const currentAspect = rect.width / rect.height;
  let width: number;
  let height: number;

  if (currentAspect > aspect) {
    // Too wide for the target aspect: keep width, shrink/grow height to match.
    width = rect.width;
    height = width / aspect;
  } else {
    // Too tall (or exact): keep height, shrink/grow width to match.
    height = rect.height;
    width = height * aspect;
  }

  const left = rect.x;
  const top = rect.y;
  const right = rect.x + rect.width;
  const bottom = rect.y + rect.height;
  const centerX = rect.x + rect.width / 2;
  const centerY = rect.y + rect.height / 2;

  switch (anchor) {
    case 'nw':
      return { x: left, y: top, width, height };
    case 'ne':
      return { x: right - width, y: top, width, height };
    case 'sw':
      return { x: left, y: bottom - height, width, height };
    case 'se':
      return { x: right - width, y: bottom - height, width, height };
    case 'center':
      return { x: centerX - width / 2, y: centerY - height / 2, width, height };
  }
};
