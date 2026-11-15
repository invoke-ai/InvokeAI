import type { CanvasBezierPathContract } from '@workbench/canvas-engine/contracts';
import type { Mat2d, Vec2 } from '@workbench/canvas-engine/types';

import { applyToPoint } from './mat2d';

/** Missing Corner handles get grab markers; existing handles always show their real positions. */
export const vectorHandleLayout = (
  path: CanvasBezierPathContract,
  pointIndex: number,
  matrix: Mat2d
): {
  inHandle: Vec2 | null;
  outHandle: Vec2 | null;
} => {
  const point = path.points[pointIndex]!;
  if (point.type !== 'corner') {
    return {
      inHandle: point.inHandle ? applyToPoint(matrix, point.inHandle) : null,
      outHandle: point.outHandle ? applyToPoint(matrix, point.outHandle) : null,
    };
  }
  const previous = path.points[pointIndex - 1] ?? (path.isClosed ? path.points.at(-1) : undefined);
  const next = path.points[pointIndex + 1] ?? (path.isClosed ? path.points[0] : undefined);
  const anchor = applyToPoint(matrix, point.anchor);
  const extend = (handle: Vec2 | null, fallback: Vec2 | undefined, minimum: number): Vec2 | null => {
    if (handle) {
      return applyToPoint(matrix, handle);
    }
    const direction = handle ?? fallback;
    if (!direction) {
      return null;
    }
    const projected = applyToPoint(matrix, direction);
    const dx = projected.x - anchor.x,
      dy = projected.y - anchor.y;
    const length = Math.hypot(dx, dy);
    if (length < 1e-6) {
      return null;
    }
    return { x: anchor.x + (dx * minimum) / length, y: anchor.y + (dy * minimum) / length };
  };
  const inHandle = extend(point.inHandle, previous?.anchor, 28);
  let outHandle = extend(point.outHandle, next?.anchor, 28);
  if (inHandle && outHandle && Math.hypot(inHandle.x - outHandle.x, inHandle.y - outHandle.y) < 24) {
    // At acute corners, stagger tips along their own tangent rays.
    outHandle = extend(point.outHandle, next?.anchor, Math.hypot(inHandle.x - anchor.x, inHandle.y - anchor.y) + 24);
  }
  return { inHandle, outHandle };
};
