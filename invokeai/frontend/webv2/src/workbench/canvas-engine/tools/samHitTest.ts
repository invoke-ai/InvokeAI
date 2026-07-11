import type { Mat2d, Rect, Vec2 } from '@workbench/canvas-engine/types';

import { applyToPoint } from '@workbench/canvas-engine/math/mat2d';
import { transformBounds } from '@workbench/canvas-engine/math/rect';

import { BBOX_HANDLES, bboxHandleAt, type BboxHandle, isInsideRect } from './bboxHitTest';

export const SAM_POINT_HIT_RADIUS_PX = 8;
export const SAM_BBOX_HANDLE_HIT_PX = 14;

export type SamPointLabel = 'include' | 'exclude';

export type SamHitTarget =
  | { kind: 'point'; label: SamPointLabel; index: number }
  | { kind: 'bbox-handle'; handle: BboxHandle }
  | { kind: 'bbox-body' };

export interface SamHitTestOptions {
  view: Mat2d;
  screenPoint: Vec2;
  includePoints: readonly Vec2[];
  excludePoints: readonly Vec2[];
  bbox: Rect | null;
}

export const samHitTest = (options: SamHitTestOptions): SamHitTarget | null => {
  for (const label of ['include', 'exclude'] as const) {
    const points = label === 'include' ? options.includePoints : options.excludePoints;
    for (let index = points.length - 1; index >= 0; index -= 1) {
      const point = points[index];
      if (!point) {
        continue;
      }
      const screen = applyToPoint(options.view, point);
      if (Math.hypot(options.screenPoint.x - screen.x, options.screenPoint.y - screen.y) <= SAM_POINT_HIT_RADIUS_PX) {
        return { index, kind: 'point', label };
      }
    }
  }

  if (!options.bbox) {
    return null;
  }
  const screenRect = transformBounds(options.view, options.bbox);
  const handle = bboxHandleAt(screenRect, options.screenPoint, SAM_BBOX_HANDLE_HIT_PX);
  if (handle) {
    return { handle, kind: 'bbox-handle' };
  }
  return isInsideRect(screenRect, options.screenPoint) ? { kind: 'bbox-body' } : null;
};

const clamp = (value: number, min: number, max: number): number => Math.min(max, Math.max(min, value));

export const clipPointToRect = (point: Vec2, bounds: Rect): Vec2 => ({
  x: clamp(point.x, bounds.x, bounds.x + bounds.width),
  y: clamp(point.y, bounds.y, bounds.y + bounds.height),
});

export const clipRectToRect = (rect: Rect, bounds: Rect): Rect => {
  const left = clamp(rect.x, bounds.x, bounds.x + bounds.width);
  const top = clamp(rect.y, bounds.y, bounds.y + bounds.height);
  const right = clamp(rect.x + rect.width, left, bounds.x + bounds.width);
  const bottom = clamp(rect.y + rect.height, top, bounds.y + bounds.height);
  return { height: bottom - top, width: right - left, x: left, y: top };
};

export const rectFromPoints = (start: Vec2, end: Vec2, bounds: Rect): Rect =>
  clipRectToRect(
    {
      height: Math.abs(end.y - start.y),
      width: Math.abs(end.x - start.x),
      x: Math.min(start.x, end.x),
      y: Math.min(start.y, end.y),
    },
    bounds
  );

export const moveSamBbox = (start: Rect, dx: number, dy: number, bounds: Rect): Rect => ({
  height: start.height,
  width: start.width,
  x: clamp(start.x + dx, bounds.x, bounds.x + bounds.width - start.width),
  y: clamp(start.y + dy, bounds.y, bounds.y + bounds.height - start.height),
});

const hasWest = (handle: BboxHandle): boolean => BBOX_HANDLES.includes(handle) && handle.includes('w');
const hasEast = (handle: BboxHandle): boolean => handle.includes('e');
const hasNorth = (handle: BboxHandle): boolean => handle.includes('n');
const hasSouth = (handle: BboxHandle): boolean => handle.includes('s');

export const resizeSamBbox = (options: { start: Rect; handle: BboxHandle; delta: Vec2; bounds: Rect }): Rect => {
  const { bounds, delta, handle, start } = options;
  let left = start.x;
  let top = start.y;
  let right = start.x + start.width;
  let bottom = start.y + start.height;

  if (hasWest(handle)) {
    left = clamp(start.x + delta.x, bounds.x, right - 1);
  }
  if (hasEast(handle)) {
    right = clamp(start.x + start.width + delta.x, left + 1, bounds.x + bounds.width);
  }
  if (hasNorth(handle)) {
    top = clamp(start.y + delta.y, bounds.y, bottom - 1);
  }
  if (hasSouth(handle)) {
    bottom = clamp(start.y + start.height + delta.y, top + 1, bounds.y + bounds.height);
  }
  return { height: bottom - top, width: right - left, x: left, y: top };
};
