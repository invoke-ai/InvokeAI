/**
 * Shared click-to-place polyline with cursor band. Close on double-click, Enter or first-vertex screen-radius hit;
 * preview that hit with a ring. Freehand traces share the same CSS-pixel input decimation.
 */

import type { LassoPreview } from '@workbench/canvas-engine/engineStores';
import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';

import type { ToolContext } from './tool';

/** Fewest distinct points that make a fillable polygon. */
export const MIN_POLYLINE_POINTS = 3;

/** Screen-pixel radius around the first vertex where a click closes the polygon. */
export const POLYLINE_CLOSE_HIT_PX = 8;

/** Longest gap (ms) between two presses that still reads as a double-click. */
export const POLYLINE_DOUBLE_CLICK_MS = 350;

/** Screen-pixel radius within which two presses count as the same spot. */
const DOUBLE_CLICK_PX = 4;

const distance = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);

export interface PolylineSession {
  points: Vec2[];
  /** The rubber-band endpoint (the last cursor position), or `null` before any move. */
  cursor: Vec2 | null;
  /** Whether the cursor sits on the first vertex, where a click closes the polygon. */
  closeArmed: boolean;
  /** Screen position and time of the previous press, for double-click detection. */
  lastPressScreen: Vec2;
  lastPressAt: number;
}

/** Starts a session with its first vertex at the press. */
export const startPolyline = (input: PointerInput): PolylineSession => ({
  closeArmed: false,
  cursor: null,
  lastPressAt: input.timeStamp,
  lastPressScreen: input.screenPoint,
  points: [{ x: input.documentPoint.x, y: input.documentPoint.y }],
});

/** True when a press at `screenPoint` lands on the polygon's first vertex (once it is fillable). */
export const isOnFirstVertex = (ctx: ToolContext, session: PolylineSession, screenPoint: Vec2): boolean => {
  const first = session.points[0];
  if (!first || session.points.length < MIN_POLYLINE_POINTS) {
    return false;
  }
  // Project through the viewport rather than caching the vertex's screen
  // position: the user may pan (space-hold) between vertices.
  return distance(ctx.viewport.documentToScreen(first), screenPoint) <= POLYLINE_CLOSE_HIT_PX;
};

export const pressPolyline = (ctx: ToolContext, session: PolylineSession, input: PointerInput): 'close' | 'place' => {
  const isDoubleClick =
    input.timeStamp - session.lastPressAt <= POLYLINE_DOUBLE_CLICK_MS &&
    distance(session.lastPressScreen, input.screenPoint) <= DOUBLE_CLICK_PX;
  if (isDoubleClick || isOnFirstVertex(ctx, session, input.screenPoint)) {
    return 'close';
  }
  session.points.push({ x: input.documentPoint.x, y: input.documentPoint.y });
  session.lastPressAt = input.timeStamp;
  session.lastPressScreen = input.screenPoint;
  return 'place';
};

/** Tracks the rubber band and the close cue; true when the cue flipped (the cursor should refresh). */
export const movePolyline = (ctx: ToolContext, session: PolylineSession, input: PointerInput): boolean => {
  session.cursor = { x: input.documentPoint.x, y: input.documentPoint.y };
  const closeArmed = isOnFirstVertex(ctx, session, input.screenPoint);
  if (closeArmed === session.closeArmed) {
    return false;
  }
  session.closeArmed = closeArmed;
  return true;
};

/** The overlay's view of the session. */
export const polylinePreview = (session: PolylineSession): LassoPreview => ({
  closeArmed: session.closeArmed,
  closeRadiusPx: session.points.length >= MIN_POLYLINE_POINTS ? POLYLINE_CLOSE_HIT_PX : null,
  cursor: session.cursor,
  kind: 'polygon',
  points: session.points.slice(),
});

/** Minimum on-screen gap (CSS px) between stored freehand points, independent of zoom. */
export const FREEHAND_MIN_POINT_SPACING_PX = 2;

/** A dragged freehand outline: document-space points, decimated by on-screen travel. */
export interface FreehandTrace {
  points: Vec2[];
  lastScreen: Vec2;
}

export const startFreehandTrace = (input: PointerInput): FreehandTrace => ({
  lastScreen: input.screenPoint,
  points: [{ x: input.documentPoint.x, y: input.documentPoint.y }],
});

export const extendFreehandTrace = (trace: FreehandTrace, samples: readonly PointerInput[]): void => {
  for (const sample of samples) {
    if (distance(trace.lastScreen, sample.screenPoint) >= FREEHAND_MIN_POINT_SPACING_PX) {
      trace.points.push({ x: sample.documentPoint.x, y: sample.documentPoint.y });
      trace.lastScreen = sample.screenPoint;
    }
  }
};

/** The completed outline, always ending on the release point. */
export const finishFreehandTrace = (trace: FreehandTrace, input: PointerInput): Vec2[] => {
  const last = trace.points[trace.points.length - 1];
  const final = input.documentPoint;
  return last && last.x === final.x && last.y === final.y
    ? trace.points.slice()
    : [...trace.points, { x: final.x, y: final.y }];
};
