/** Rect drags shared by the marquee and shape tools: Shift squares the rect, Alt grows it from the press point. */

import type { Rect, Vec2 } from '@workbench/canvas-engine/types';

/** How the drag modifiers shape the rect. */
export interface DragRectConstraints {
  /** Equal width and height. */
  square: boolean;
  /** The press point is the rect's centre rather than a corner. */
  fromCenter: boolean;
}

/** The drag delta, optionally forced to equal magnitude on both axes (sign kept). */
const constrainedDelta = (start: Vec2, end: Vec2, square: boolean): Vec2 => {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  if (!square) {
    return { x: dx, y: dy };
  }
  const side = Math.max(Math.abs(dx), Math.abs(dy));
  return { x: (dx < 0 ? -1 : 1) * side, y: (dy < 0 ? -1 : 1) * side };
};

/** Normalized integer bounds of a drag; centered mode mirrors the constrained delta around the press point. */
export const dragRect = (start: Vec2, end: Vec2, constraints: DragRectConstraints): Rect => {
  const delta = constrainedDelta(start, end, constraints.square);
  const a = constraints.fromCenter ? { x: start.x - delta.x, y: start.y - delta.y } : start;
  const b = { x: start.x + delta.x, y: start.y + delta.y };
  const left = Math.round(Math.min(a.x, b.x));
  const top = Math.round(Math.min(a.y, b.y));
  return {
    height: Math.round(Math.max(a.y, b.y)) - top,
    width: Math.round(Math.max(a.x, b.x)) - left,
    x: left,
    y: top,
  };
};
