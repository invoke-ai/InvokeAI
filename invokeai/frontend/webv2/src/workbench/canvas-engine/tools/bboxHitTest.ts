/**
 * Pure geometry for the bbox (generation-frame) tool: handle layout, screen-space
 * hit-testing, and resize/move math.
 *
 * The bbox is an axis-aligned document-space rectangle. Its eight resize handles
 * (four corners + four edge midpoints) are hit-tested in SCREEN space so their
 * grab areas stay a constant pixel size regardless of zoom — callers project the
 * bbox to screen coordinates first. Resize math runs in DOCUMENT space: the edge
 * opposite the grabbed handle stays fixed, the moved edge(s) follow a pointer
 * delta, then snapping, min-size clamping, and (optionally) an aspect constraint
 * are applied.
 *
 * Zero React, zero import-time side effects.
 */

import type { Rect, Vec2 } from '@workbench/canvas-engine/types';

import { type AspectAnchor, constrainAspect, snapToGrid } from '@workbench/canvas-engine/math/snapping';

/** The eight resize handles: four corners + four edge midpoints. */
export type BboxHandle = 'nw' | 'n' | 'ne' | 'e' | 'se' | 's' | 'sw' | 'w';

/** What a pointer-down landed on: a resize handle, the interior (move), or nothing. */
export type BboxTarget = BboxHandle | 'move';

/**
 * Handles in hit-test priority order: corners first, so at a small bbox where a
 * corner and an adjacent edge midpoint overlap, the corner wins.
 */
export const BBOX_HANDLES: readonly BboxHandle[] = ['nw', 'ne', 'se', 'sw', 'n', 'e', 's', 'w'];

/** Side length (screen px) of a handle's square grab area. */
export const BBOX_HANDLE_HIT_PX = 14;

const hasWest = (h: BboxHandle): boolean => h === 'nw' || h === 'w' || h === 'sw';
const hasEast = (h: BboxHandle): boolean => h === 'ne' || h === 'e' || h === 'se';
const hasNorth = (h: BboxHandle): boolean => h === 'nw' || h === 'n' || h === 'ne';
const hasSouth = (h: BboxHandle): boolean => h === 'sw' || h === 's' || h === 'se';

/** True for the four corner handles (two-letter ids), false for edge midpoints. */
export const isCornerHandle = (h: BboxHandle): boolean => h.length === 2;

/** The point of a handle on `rect` (works in any space — screen or document). */
export const bboxHandlePoint = (rect: Rect, handle: BboxHandle): Vec2 => {
  const x = hasWest(handle) ? rect.x : hasEast(handle) ? rect.x + rect.width : rect.x + rect.width / 2;
  const y = hasNorth(handle) ? rect.y : hasSouth(handle) ? rect.y + rect.height : rect.y + rect.height / 2;
  return { x, y };
};

/** The corner of `rect` opposite `handle` (stays fixed while `handle` is dragged). */
const OPPOSITE_CORNER: Record<'nw' | 'ne' | 'se' | 'sw', AspectAnchor> = {
  ne: 'sw',
  nw: 'se',
  se: 'nw',
  sw: 'ne',
};

/** True when `point` is inside (or on the border of) `rect`. */
export const isInsideRect = (rect: Rect, point: Vec2): boolean =>
  point.x >= rect.x && point.x <= rect.x + rect.width && point.y >= rect.y && point.y <= rect.y + rect.height;

/**
 * The handle whose square grab area (side `hitSize`, screen px) contains
 * `point`, or `null`. `screenRect` is the bbox already projected to screen space.
 */
export const bboxHandleAt = (
  screenRect: Rect,
  point: Vec2,
  hitSize: number = BBOX_HANDLE_HIT_PX
): BboxHandle | null => {
  const half = hitSize / 2;
  for (const handle of BBOX_HANDLES) {
    const center = bboxHandlePoint(screenRect, handle);
    if (Math.abs(point.x - center.x) <= half && Math.abs(point.y - center.y) <= half) {
      return handle;
    }
  }
  return null;
};

/**
 * What a screen-space `point` targets on the bbox: a resize handle (grab area
 * first), the interior (`'move'`), or `null` (outside — the bbox is never
 * deselected).
 */
export const bboxTargetAt = (
  screenRect: Rect,
  point: Vec2,
  hitSize: number = BBOX_HANDLE_HIT_PX
): BboxTarget | null => {
  const handle = bboxHandleAt(screenRect, point, hitSize);
  if (handle) {
    return handle;
  }
  return isInsideRect(screenRect, point) ? 'move' : null;
};

/** Minimum bbox side (document px): one grid cell, but never below 1px. */
export const bboxMinSize = (grid: number): number => Math.max(1, grid);

/** Translates `start` by a document-space delta, snapping the origin to `grid` unless bypassed. */
export const moveBbox = (start: Rect, dx: number, dy: number, grid: number, snap: boolean): Rect => {
  let x = start.x + dx;
  let y = start.y + dy;
  if (snap) {
    x = snapToGrid(x, grid);
    y = snapToGrid(y, grid);
  }
  return { height: start.height, width: start.width, x, y };
};

export interface ResizeBboxParams {
  /** The bbox at gesture start. */
  start: Rect;
  handle: BboxHandle;
  /** Document-space pointer delta since gesture start. */
  dx: number;
  dy: number;
  /** Model grid size (document px). */
  grid: number;
  /** Whether to snap moved edges to the grid (false = alt-bypass). */
  snap: boolean;
  /** Whether to preserve `ratio` (aspect lock, or shift). */
  constrain: boolean;
  /** Target width / height ratio, used when `constrain`. */
  ratio: number;
}

/** Builds a rect from a fixed anchor corner extending by `width`/`height` in the handle's directions. */
const rectFromCorner = (fixed: Vec2, width: number, height: number, signX: number, signY: number): Rect => {
  const x = signX >= 0 ? fixed.x : fixed.x - width;
  const y = signY >= 0 ? fixed.y : fixed.y - height;
  return { height, width, x, y };
};

/** Unconstrained resize: moved edges follow the delta; the opposite edge stays fixed. */
const resizeFree = (p: ResizeBboxParams): Rect => {
  const { dx, dy, grid, handle, snap, start } = p;
  let left = start.x;
  let top = start.y;
  let right = start.x + start.width;
  let bottom = start.y + start.height;

  if (hasWest(handle)) {
    left = snap ? snapToGrid(left + dx, grid) : left + dx;
  }
  if (hasEast(handle)) {
    right = snap ? snapToGrid(right + dx, grid) : right + dx;
  }
  if (hasNorth(handle)) {
    top = snap ? snapToGrid(top + dy, grid) : top + dy;
  }
  if (hasSouth(handle)) {
    bottom = snap ? snapToGrid(bottom + dy, grid) : bottom + dy;
  }

  const min = bboxMinSize(grid);
  // Clamp so the moved edge never crosses within `min` of the fixed edge.
  if (hasWest(handle)) {
    left = Math.min(left, right - min);
  }
  if (hasEast(handle)) {
    right = Math.max(right, left + min);
  }
  if (hasNorth(handle)) {
    top = Math.min(top, bottom - min);
  }
  if (hasSouth(handle)) {
    bottom = Math.max(bottom, top + min);
  }

  return { height: bottom - top, width: right - left, x: left, y: top };
};

/** Aspect-constrained corner resize: anchored at the opposite corner, grid-aligning width. */
const resizeCornerConstrained = (p: ResizeBboxParams): Rect => {
  const { dx, dy, grid, handle, ratio, snap, start } = p;
  const anchorCorner = OPPOSITE_CORNER[handle as 'nw' | 'ne' | 'se' | 'sw'];
  const fixed = bboxHandlePoint(start, anchorCorner as BboxHandle);
  const moveStart = bboxHandlePoint(start, handle);
  const mx = moveStart.x + dx;
  const my = moveStart.y + dy;
  const signX = moveStart.x >= fixed.x ? 1 : -1;
  const signY = moveStart.y >= fixed.y ? 1 : -1;

  const rawW = Math.abs(mx - fixed.x);
  const rawH = Math.abs(my - fixed.y);

  // Pick the driven axis: whichever keeps the frame closest to the pointer.
  let width = rawW / rawH >= ratio ? rawW : rawH * ratio;
  const min = bboxMinSize(grid);
  width = snap ? Math.max(min, snapToGrid(width, grid)) : Math.max(min, width);
  let height = width / ratio;
  if (height < 1) {
    height = 1;
    width = height * ratio;
  }
  return rectFromCorner(fixed, width, height, signX, signY);
};

/** Aspect-constrained edge resize: the free axis follows the delta, the other axis re-derives from `ratio`, centered. */
const resizeEdgeConstrained = (p: ResizeBboxParams): Rect => {
  const { dx, dy, grid, handle, ratio, snap, start } = p;
  const min = bboxMinSize(grid);
  const left = start.x;
  const right = start.x + start.width;
  const top = start.y;
  const bottom = start.y + start.height;
  const centerX = start.x + start.width / 2;
  const centerY = start.y + start.height / 2;

  if (handle === 'e' || handle === 'w') {
    let width: number;
    if (handle === 'e') {
      const edge = snap ? snapToGrid(right + dx, grid) : right + dx;
      width = Math.max(min, edge - left);
    } else {
      const edge = snap ? snapToGrid(left + dx, grid) : left + dx;
      width = Math.max(min, right - edge);
    }
    const height = Math.max(1, width / ratio);
    const x = handle === 'e' ? left : right - width;
    return { height, width, x, y: centerY - height / 2 };
  }

  // 'n' | 's': height-driven.
  let height: number;
  if (handle === 's') {
    const edge = snap ? snapToGrid(bottom + dy, grid) : bottom + dy;
    height = Math.max(min, edge - top);
  } else {
    const edge = snap ? snapToGrid(top + dy, grid) : top + dy;
    height = Math.max(min, bottom - edge);
  }
  const width = Math.max(1, height * ratio);
  const y = handle === 's' ? top : bottom - height;
  return { height, width, x: centerX - width / 2, y };
};

/**
 * Resizes `start` for a handle drag. Applies (in order) the delta to the moved
 * edge(s), grid snapping (unless bypassed), min-size clamping, and — when
 * `constrain` — an aspect-ratio constraint anchored at the opposite corner
 * (corners) or the center (edges).
 */
export const resizeBbox = (p: ResizeBboxParams): Rect => {
  if (p.constrain && p.ratio > 0) {
    return isCornerHandle(p.handle) ? resizeCornerConstrained(p) : resizeEdgeConstrained(p);
  }
  return resizeFree(p);
};

/** Re-fits `rect` to `ratio`, keeping its center fixed, then snaps to `grid` and clamps min size. */
export const constrainBboxToRatio = (rect: Rect, ratio: number, grid: number): Rect => {
  if (ratio <= 0) {
    return rect;
  }
  const fitted = constrainAspect(rect, ratio, 'center');
  const min = bboxMinSize(grid);
  const width = Math.max(min, snapToGrid(fitted.width, grid));
  const height = Math.max(1, width / ratio);
  const centerX = rect.x + rect.width / 2;
  const centerY = rect.y + rect.height / 2;
  return { height, width, x: centerX - width / 2, y: centerY - height / 2 };
};

/** Rounds a rect's fields to integers (document px), enforcing a ≥1px size. */
export const roundBbox = (rect: Rect): Rect => ({
  height: Math.max(1, Math.round(rect.height)),
  width: Math.max(1, Math.round(rect.width)),
  x: Math.round(rect.x),
  y: Math.round(rect.y),
});

/** True when two rects are equal after rounding to integer document px. */
export const bboxEquals = (a: Rect, b: Rect): boolean => {
  const ra = roundBbox(a);
  const rb = roundBbox(b);
  return ra.x === rb.x && ra.y === rb.y && ra.width === rb.width && ra.height === rb.height;
};
