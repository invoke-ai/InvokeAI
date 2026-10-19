/**
 * Freehand lasso gathers decimated drag points; polygon lasso spans clicks and closes on double-click, Enter or
 * first-vertex proximity. Both commit through `commitSelection` with shared modifier/persistent-op resolution.
 * Previews never dispatch. Temporary tool switches preserve polygon sessions; real switches cancel them. Selection
 * history is owned by the engine wrapper.
 */

import type { Vec2 } from '@workbench/canvas-engine/types';

import { polygonBounds, polygonToSvgPath } from '@workbench/canvas-engine/freehand';
import { selectionOpFor } from '@workbench/canvas-engine/selection/selectionOpMode';

import type { Tool, ToolContext } from './tool';

import {
  extendFreehandTrace,
  finishFreehandTrace,
  MIN_POLYLINE_POINTS,
  movePolyline,
  polylinePreview,
  pressPolyline,
  startFreehandTrace,
  startPolyline,
  type FreehandTrace,
  type PolylineSession,
} from './polylineSession';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

const distance = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);

/** A freehand drag in progress. */
interface FreehandSession extends FreehandTrace {
  kind: 'freehand';
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
    const bounds = polygonBounds(polygon);
    // A double-click's second press lands on the previous vertex, so count
    // distinct points; and a flat polygon would select a line of nothing.
    const distinct = polygon.filter((point, index) => index === 0 || distance(point, polygon[index - 1]!) >= 1);
    if (distinct.length < MIN_POLYLINE_POINTS || bounds.width < 1 || bounds.height < 1 || !ctx.commitSelection) {
      return;
    }
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
        session = { ...startFreehandTrace(input), kind: 'freehand' };
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
        extendFreehandTrace(session, batch);
      }
      publishPreview(ctx);
    },
    onPointerUp: (ctx, input) => {
      // A polygon session spans many clicks — it closes on double-click, Enter,
      // or a click on the first vertex, never on a plain release.
      if (!session || session.kind === 'polygon') {
        return;
      }
      const polygon = finishFreehandTrace(session, input);
      reset();
      clearPreview(ctx);
      commit(ctx, polygon, input.modifiers);
    },
  };
};
