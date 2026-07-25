/**
 * The lasso tool: freehand and polygonal pixel-selection.
 *
 * Both shapes commit the same thing — a closed polygon through
 * {@link ToolContext.commitSelection}, with the boolean op resolved by
 * {@link selectionOpFor} (shift = add, alt = subtract, shift+alt = intersect;
 * with none held, the persistent op mode from `lassoOptions`). Only the way the
 * points are gathered differs, so they share one tool rather than two:
 *
 * - **freehand** (`shape: 'freehand'`) — a primary-button DRAG accumulates
 *   document-space points, coalesced by the pipeline then distance-decimated
 *   here like the brush thins its input. Pointer-up closes and commits.
 * - **polygon** (`shape: 'polygon'`) — a click-to-place-vertex SESSION spanning
 *   many clicks. Each press appends a vertex; a rubber-band segment tracks the
 *   cursor. The session closes and commits on a double-click, on Enter, or on a
 *   click landing within {@link POLYGON_CLOSE_HIT_PX} screen pixels of the first
 *   vertex. Escape drops it.
 *
 * A live dashed preview of the in-progress path is published to the engine's
 * `lassoPreview` store (a transient channel — no dispatch, no reducer traffic).
 * The polygon session survives a temporary modifier-hold tool switch (space →
 * view, so the user can pan mid-polygon), the same way the transform session
 * does; a REAL tool switch drops it.
 *
 * Selection edits are transient interaction state: they are NOT dispatches and
 * NOT recorded on the engine's undo history (legacy parity — selection changes
 * aren't undoable). Zero React, zero import-time side effects.
 */

import type { Vec2 } from '@workbench/canvas-engine/types';

import { polygonBounds, polygonToSvgPath } from '@workbench/canvas-engine/freehand';
import { selectionOpFor } from '@workbench/canvas-engine/selection/selectionOpMode';

import type { Tool, ToolContext } from './tool';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Minimum document-space gap between stored polygon points (input decimation). */
export const LASSO_MIN_POINT_DISTANCE = 2;

/** Fewest distinct points that make a fillable selection polygon. */
const MIN_POLYGON_POINTS = 3;

/** Screen-pixel radius around the first vertex where a click closes the polygon. */
export const POLYGON_CLOSE_HIT_PX = 8;

/** Longest gap (ms) between two presses that still reads as a double-click. */
export const POLYGON_DOUBLE_CLICK_MS = 350;

/** Screen-pixel radius within which two presses count as the same spot. */
const POLYGON_DOUBLE_CLICK_PX = 4;

const distance = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);

/** A freehand drag in progress. */
interface FreehandSession {
  kind: 'freehand';
  points: Vec2[];
}

/** A polygon vertex-placing session spanning multiple clicks. */
interface PolygonSession {
  kind: 'polygon';
  points: Vec2[];
  /** The rubber-band endpoint (the last cursor position), or `null` before any move. */
  cursor: Vec2 | null;
  /** Screen position and time of the previous press, for double-click detection. */
  lastPressScreen: Vec2;
  lastPressAt: number;
}

type Session = FreehandSession | PolygonSession;

/** Creates a fresh lasso tool with its own per-gesture point buffer. */
export const createLassoTool = (): Tool => {
  let session: Session | null = null;

  const reset = (): void => {
    session = null;
  };

  /** Appends a point if it is far enough from the last stored one. */
  const pushDecimated = (points: Vec2[], p: Vec2): void => {
    const last = points[points.length - 1];
    if (!last || distance(last, p) >= LASSO_MIN_POINT_DISTANCE) {
      points.push({ x: p.x, y: p.y });
    }
  };

  /**
   * Publishes the in-progress outline. For a polygon the rubber-band endpoint is
   * appended, so the overlay (which closes the loop) shows both the segment to
   * the cursor and the implied closing edge.
   */
  const publishPreview = (ctx: ToolContext): void => {
    if (!session) {
      ctx.stores.lassoPreview.set(null);
      ctx.invalidate({ overlay: true });
      return;
    }
    const points =
      session.kind === 'polygon' && session.cursor ? [...session.points, session.cursor] : session.points.slice();
    ctx.stores.lassoPreview.set(points.length > 0 ? points : null);
    ctx.invalidate({ overlay: true });
  };

  const clearPreview = (ctx: ToolContext): void => {
    ctx.stores.lassoPreview.set(null);
    ctx.invalidate({ overlay: true });
  };

  /** Commits `polygon` as a selection under the modifiers held at close time. */
  const commit = (ctx: ToolContext, polygon: readonly Vec2[], modifiers: { shift: boolean; alt: boolean }): void => {
    if (polygon.length < MIN_POLYGON_POINTS || !ctx.commitSelection) {
      return;
    }
    ctx.commitSelection({
      bounds: polygonBounds(polygon),
      op: selectionOpFor(modifiers, ctx.stores.lassoOptions.get().mode),
      path: ctx.createPath2D(polygonToSvgPath(polygon)),
    });
  };

  /** True when a press at `screenPoint` lands on the polygon's first vertex. */
  const isOnFirstVertex = (ctx: ToolContext, polygon: PolygonSession, screenPoint: Vec2): boolean => {
    const first = polygon.points[0];
    if (!first || polygon.points.length < MIN_POLYGON_POINTS) {
      return false;
    }
    // Project through the viewport rather than caching the vertex's screen
    // position: the user may pan (space-hold) between vertices.
    return distance(ctx.viewport.documentToScreen(first), screenPoint) <= POLYGON_CLOSE_HIT_PX;
  };

  const closePolygon = (
    ctx: ToolContext,
    polygon: PolygonSession,
    modifiers: { shift: boolean; alt: boolean }
  ): void => {
    const points = polygon.points.slice();
    reset();
    clearPreview(ctx);
    commit(ctx, points, modifiers);
  };

  return {
    cursor: () => 'crosshair',
    id: 'lasso',
    onDeactivate: (ctx, opts) => {
      if (opts?.temporary) {
        // A modifier-hold switch (space → view to pan mid-polygon) must not
        // discard the session; the pipeline suppresses temp switches mid-drag,
        // so a freehand gesture cannot be interrupted here either.
        return;
      }
      reset();
      clearPreview(ctx);
    },
    onKeyCommand: (ctx, command) => {
      if (!session) {
        return;
      }
      if (command === 'cancel') {
        reset();
        clearPreview(ctx);
        return;
      }
      if (session.kind === 'polygon') {
        // Enter closes the polygon. No modifiers are in play on a key command,
        // so the persistent op mode applies.
        closePolygon(ctx, session, { alt: false, shift: false });
      }
    },
    onPointerCancel: (ctx) => {
      reset();
      clearPreview(ctx);
    },
    onPointerDown: (ctx, input) => {
      if ((input.buttons & PRIMARY_BUTTON) === 0) {
        return;
      }
      const shape = ctx.stores.lassoOptions.get().shape;

      if (shape === 'freehand') {
        if (session) {
          return;
        }
        session = { kind: 'freehand', points: [{ x: input.documentPoint.x, y: input.documentPoint.y }] };
        publishPreview(ctx);
        return;
      }

      if (!session) {
        session = {
          cursor: null,
          kind: 'polygon',
          lastPressAt: input.timeStamp,
          lastPressScreen: input.screenPoint,
          points: [{ x: input.documentPoint.x, y: input.documentPoint.y }],
        };
        publishPreview(ctx);
        return;
      }
      if (session.kind !== 'polygon') {
        // The shape option flipped mid-drag; let the freehand gesture finish.
        return;
      }

      const isDoubleClick =
        input.timeStamp - session.lastPressAt <= POLYGON_DOUBLE_CLICK_MS &&
        distance(session.lastPressScreen, input.screenPoint) <= POLYGON_DOUBLE_CLICK_PX;
      if (isDoubleClick || isOnFirstVertex(ctx, session, input.screenPoint)) {
        closePolygon(ctx, session, input.modifiers);
        return;
      }

      session.points.push({ x: input.documentPoint.x, y: input.documentPoint.y });
      session.lastPressAt = input.timeStamp;
      session.lastPressScreen = input.screenPoint;
      publishPreview(ctx);
    },
    onPointerMove: (ctx, input, batch) => {
      if (!session) {
        return;
      }
      if (session.kind === 'polygon') {
        session.cursor = { x: input.documentPoint.x, y: input.documentPoint.y };
      } else {
        for (const sample of batch) {
          pushDecimated(session.points, sample.documentPoint);
        }
      }
      publishPreview(ctx);
    },
    onPointerUp: (ctx, input) => {
      // A polygon session spans many clicks — it closes on double-click, Enter,
      // or a click on the first vertex, never on a plain release.
      if (!session || session.kind === 'polygon') {
        return;
      }
      pushDecimated(session.points, input.documentPoint);
      const polygon = session.points.slice();
      reset();
      clearPreview(ctx);
      commit(ctx, polygon, input.modifiers);
    },
  };
};
