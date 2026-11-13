/**
 * Freehand lasso keeps drag corners, coalescing only exact straight runs. Polygon lasso spans clicks and closes on
 * double-click, Enter or first-vertex proximity. Both commit through `commitSelection` with shared modifier/persistent-op resolution.
 * Previews never dispatch. Temporary tool switches preserve polygon sessions; real switches cancel them. Selection
 * history is owned by the engine wrapper.
 */

import type { Vec2 } from '@workbench/canvas-engine/types';

import { polygonBounds, polygonToSvgPath } from '@workbench/canvas-engine/freehand';
import { selectionOpFor } from '@workbench/canvas-engine/selection/selectionOpMode';

import type { Tool, ToolContext } from './tool';

import { movePolyline, polylinePreview, pressPolyline, startPolyline, type PolylineSession } from './polylineSession';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Keep subpixel corners and winding; only redundant points on a forward straight run may be dropped. */
const appendFreehandPoint = (points: Vec2[], point: Vec2): void => {
  const last = points[points.length - 1];
  if (last && last.x === point.x && last.y === point.y) {
    return;
  }
  const previous = points[points.length - 2];
  if (previous && last) {
    const dx = last.x - previous.x;
    const dy = last.y - previous.y;
    const nextDx = point.x - last.x;
    const nextDy = point.y - last.y;
    if (dx * nextDy === dy * nextDx && dx * nextDx + dy * nextDy > 0) {
      points[points.length - 1] = { x: point.x, y: point.y };
      return;
    }
  }
  points.push({ x: point.x, y: point.y });
};

/** A freehand drag in progress. */
interface FreehandSession {
  kind: 'freehand';
  points: Vec2[];
}

/** A polygon vertex-placing session spanning multiple clicks. */
interface PolygonSession extends PolylineSession {
  kind: 'polygon';
}

type Session = FreehandSession | PolygonSession;

/** Creates a fresh lasso tool with its own per-gesture point buffer. */
export const createLassoTool = (): Tool => {
  let session: Session | null = null;

  const reset = (): void => {
    session = null;
  };

  /** Publishes the in-progress outline for the overlay. */
  const publishPreview = (ctx: ToolContext): void => {
    if (!session || session.points.length === 0) {
      ctx.stores.lassoPreview.set(null);
    } else if (session.kind === 'polygon') {
      ctx.stores.lassoPreview.set(polylinePreview(session));
    } else {
      ctx.stores.lassoPreview.set({ kind: 'freehand', points: session.points.slice() });
    }
    ctx.invalidate({ overlay: true });
  };

  const clearPreview = (ctx: ToolContext): void => {
    ctx.stores.lassoPreview.set(null);
    ctx.invalidate({ overlay: true });
  };

  /** Commits `polygon` as a selection under the modifiers held at close time. */
  const commit = (ctx: ToolContext, polygon: readonly Vec2[], modifiers: { shift: boolean; alt: boolean }): void => {
    const first = polygon[0];
    const second = polygon.find((point) => point.x !== first?.x || point.y !== first?.y);
    // A line has no fill, but signed area alone would also reject valid self-crossing outlines.
    if (
      !first ||
      !second ||
      !polygon.some(
        (point) => (second.x - first.x) * (point.y - first.y) !== (second.y - first.y) * (point.x - first.x)
      ) ||
      !ctx.commitSelection
    ) {
      return;
    }
    const bounds = polygonBounds(polygon);
    ctx.commitSelection({
      bounds,
      op: selectionOpFor(modifiers, ctx.stores.lassoOptions.get().mode),
      path: ctx.createPath2D(polygonToSvgPath(polygon)),
    });
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
    cursor: () => (session?.kind === 'polygon' && session.closeArmed ? 'pointer' : 'crosshair'),
    id: 'lasso',
    usesAltKey: true,
    onDeactivate: (ctx, opts) => {
      if (opts?.temporary) {
        // Temporary switches preserve polygon sessions for panning; mid-drag switches are already blocked by the
        // pipeline.
        return;
      }
      reset();
      clearPreview(ctx);
    },
    onKeyCommand: (ctx, command) => {
      if (!session) {
        return false;
      }
      if (command === 'cancel') {
        // Drops the outline only; the committed selection stays.
        reset();
        clearPreview(ctx);
        return true;
      }
      if (session.kind !== 'polygon') {
        return false;
      }
      // Enter closes the polygon. No modifiers are in play on a key command,
      // so the persistent op mode applies.
      closePolygon(ctx, session, { alt: false, shift: false });
      return true;
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
        session = { kind: 'freehand', points: [{ ...input.documentPoint }] };
        publishPreview(ctx);
        return;
      }

      if (!session) {
        session = { ...startPolyline(input), kind: 'polygon' };
        publishPreview(ctx);
        return;
      }
      if (session.kind !== 'polygon') {
        // The shape option flipped mid-drag; let the freehand gesture finish.
        return;
      }

      if (pressPolyline(ctx, session, input) === 'close') {
        closePolygon(ctx, session, input.modifiers);
        return;
      }
      publishPreview(ctx);
    },
    onPointerMove: (ctx, input, batch) => {
      if (!session) {
        return;
      }
      if (session.kind === 'polygon') {
        if (movePolyline(ctx, session, input)) {
          ctx.updateCursor();
        }
      } else {
        for (const sample of batch) {
          appendFreehandPoint(session.points, sample.documentPoint);
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
      appendFreehandPoint(session.points, input.documentPoint);
      const polygon = session.points;
      reset();
      clearPreview(ctx);
      commit(ctx, polygon, input.modifiers);
    },
  };
};
