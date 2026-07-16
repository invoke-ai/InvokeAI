/**
 * Pure functions on `Rect`, the engine's axis-aligned rectangle type.
 *
 * No classes, no mutation — every function returns a new `Rect` (or a
 * primitive / array of `Rect`).
 */

import type { Mat2d, Rect, Vec2 } from '@workbench/canvas-engine/types';

import { applyToPoint } from './mat2d';

/** True if the rect has non-positive width or height. */
export const isEmpty = (r: Rect): boolean => r.width <= 0 || r.height <= 0;

/** Intersection of two rects, or `null` if they don't overlap (or either is empty). */
export const intersect = (a: Rect, b: Rect): Rect | null => {
  if (isEmpty(a) || isEmpty(b)) {
    return null;
  }
  const x = Math.max(a.x, b.x);
  const y = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.width, b.x + b.width);
  const bottom = Math.min(a.y + a.height, b.y + b.height);
  if (right <= x || bottom <= y) {
    return null;
  }
  return { x, y, width: right - x, height: bottom - y };
};

/**
 * Union (bounding box) of two rects. An empty rect is treated as
 * contributing no area — the union of an empty rect and a non-empty rect is
 * the non-empty rect; the union of two empty rects is the first rect.
 */
export const union = (a: Rect, b: Rect): Rect => {
  if (isEmpty(a) && isEmpty(b)) {
    return a;
  }
  if (isEmpty(a)) {
    return b;
  }
  if (isEmpty(b)) {
    return a;
  }
  const x = Math.min(a.x, b.x);
  const y = Math.min(a.y, b.y);
  const right = Math.max(a.x + a.width, b.x + b.width);
  const bottom = Math.max(a.y + a.height, b.y + b.height);
  return { x, y, width: right - x, height: bottom - y };
};

/** True if `p` lies within `r`, inclusive of the min edges and exclusive of the max edges. */
export const containsPoint = (r: Rect, p: Vec2): boolean =>
  p.x >= r.x && p.x < r.x + r.width && p.y >= r.y && p.y < r.y + r.height;

/** Grows (or shrinks, for negative `margin`) a rect uniformly on all sides. */
export const expand = (r: Rect, margin: number): Rect => ({
  x: r.x - margin,
  y: r.y - margin,
  width: r.width + margin * 2,
  height: r.height + margin * 2,
});

/**
 * Rounds a rect outward to integer bounds (floor the min edges, ceil the
 * max edges). Used to align dirty rects to pixel boundaries so patches
 * fully cover the region that changed.
 */
export const roundOut = (r: Rect): Rect => {
  const x = Math.floor(r.x);
  const y = Math.floor(r.y);
  const right = Math.ceil(r.x + r.width);
  const bottom = Math.ceil(r.y + r.height);
  return { x, y, width: right - x, height: bottom - y };
};

/** Axis-aligned bounds of `r` after being transformed by `m`. */
export const transformBounds = (m: Mat2d, r: Rect): Rect => {
  const corners: Vec2[] = [
    { x: r.x, y: r.y },
    { x: r.x + r.width, y: r.y },
    { x: r.x, y: r.y + r.height },
    { x: r.x + r.width, y: r.y + r.height },
  ].map((p) => applyToPoint(m, p));

  const xs = corners.map((p) => p.x);
  const ys = corners.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);

  return { x: minX, y: minY, width: maxX - minX, height: maxY - minY };
};

/** Rect area, treating empty rects as zero area. */
const area = (r: Rect): number => (isEmpty(r) ? 0 : r.width * r.height);

/**
 * True if two rects overlap or touch within `slack` pixels of each other
 * (i.e. their `slack`-expanded bounds intersect).
 */
const isNearby = (a: Rect, b: Rect, slack: number): boolean => {
  const expandedA = slack === 0 ? a : expand(a, slack);
  return intersect(expandedA, b) !== null;
};

/**
 * Coalesces a list of dirty rects into at most `maxRegions` rects.
 *
 * Algorithm: greedily merges any two rects that overlap or are within a
 * small proximity slack (derived from the average rect size) whenever the
 * merge doesn't waste too much area — specifically, whenever the merged
 * rect's area is no more than 2x the sum of the two input areas, or the
 * rects already overlap/touch. This repeats to a fixed point. If more than
 * `maxRegions` rects remain, it falls back to merging everything into a
 * single bounding rect (merge-all fallback) — simpler than picking which
 * regions to keep, and cheap dirty-rect accounting favors correctness
 * (never under-repaint) over minimizing painted area in the pathological
 * case.
 */
export const mergeDirtyRects = (rects: Rect[], maxRegions = 4): Rect[] => {
  const nonEmpty = rects.filter((r) => !isEmpty(r));
  if (nonEmpty.length <= 1) {
    return nonEmpty;
  }

  let current = nonEmpty.slice();
  const avgDimension = current.reduce((sum, r) => sum + (r.width + r.height) / 2, 0) / current.length;
  const slack = avgDimension * 0.1;

  const findMergeablePair = (rectList: Rect[]): [number, number] | null => {
    for (let i = 0; i < rectList.length; i++) {
      for (let j = i + 1; j < rectList.length; j++) {
        const a = rectList[i];
        const b = rectList[j];
        if (!a || !b) {
          continue;
        }
        const overlaps = intersect(a, b) !== null;
        const merged = union(a, b);
        const wasteOk = area(merged) <= (area(a) + area(b)) * 2;
        if (overlaps || (isNearby(a, b, slack) && wasteOk)) {
          return [i, j];
        }
      }
    }
    return null;
  };

  let pair = findMergeablePair(current);
  while (pair !== null && current.length > 1) {
    const [i, j] = pair;
    const a = current[i] as Rect;
    const b = current[j] as Rect;
    const next = current.filter((_, idx) => idx !== i && idx !== j);
    next.push(union(a, b));
    current = next;
    pair = findMergeablePair(current);
  }

  if (current.length > maxRegions) {
    return [current.reduce((acc, r) => union(acc, r))];
  }

  return current;
};
