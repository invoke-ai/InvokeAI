/**
 * Freehand lasso thins its drag by sub-pixel screen spacing and straight runs, keeping corners. Polygon lasso spans
 * clicks and closes on double-click, Enter or first-vertex proximity. Both commit through `commitSelection` with shared modifier/persistent-op resolution.
 * Previews never dispatch. Temporary tool switches preserve polygon sessions; real switches cancel them. Selection
 * history is owned by the engine wrapper.
 */

import type { PointerInput, Vec2 } from '@workbench/canvas-engine/types';

import { polygonBounds, polygonToSvgPath } from '@workbench/canvas-engine/freehand';
import { selectionOpFor } from '@workbench/canvas-engine/selection/selectionOpMode';

import type { Tool, ToolContext } from './tool';

import { movePolyline, polylinePreview, pressPolyline, startPolyline, type PolylineSession } from './polylineSession';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

const distance = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);

/**
 * Least on-screen gap (CSS px) between a kept freehand point and the one before it. Below a pixel, so it never removes
 * an outline the user can see, at any zoom; it only collapses dense pointer samples (high-rate mice, pens) that would
 * otherwise each become a vertex the preview copies and strokes on every frame.
 */
export const FREEHAND_MIN_SPACING_PX = 0.5;

/** A freehand drag in progress: document points, with the screen point each was sampled at. */
interface FreehandSession {
  kind: 'freehand';
  points: Vec2[];
  screenPoints: Vec2[];
}

/**
 * The last point gives way to a new sample while both lie within `FREEHAND_MIN_SPACING_PX` of the point before it;
 * once one leaves that radius the last point stays. Every dropped sample lies within the spacing of a kept point, so
 * thinning moves the outline by less than the spacing and keeps corners and reversals beyond it, and the outline
 * always ends at the latest sample. Redundant points on a forward straight run are dropped too.
 */
const appendFreehandPoint = (session: FreehandSession, sample: PointerInput): void => {
  const { points, screenPoints } = session;
  const point = sample.documentPoint;
  const screen = sample.screenPoint;
  const end = points.length - 1;
  const last = points[end]!;
  if (last.x === point.x && last.y === point.y) {
    return;
  }
  const previous = points[end - 1];
  const previousScreen = screenPoints[end - 1];
  if (previous && previousScreen) {
    const withinSpacing =
      distance(previousScreen, screenPoints[end]!) < FREEHAND_MIN_SPACING_PX &&
      distance(previousScreen, screen) < FREEHAND_MIN_SPACING_PX;
    const dx = last.x - previous.x;
    const dy = last.y - previous.y;
    const nextDx = point.x - last.x;
    const nextDy = point.y - last.y;
    const forwardStraight = dx * nextDy === dy * nextDx && dx * nextDx + dy * nextDy > 0;
    if (withinSpacing || forwardStraight) {
      points[end] = { x: point.x, y: point.y };
      screenPoints[end] = { x: screen.x, y: screen.y };
      return;
    }
  }
  points.push({ x: point.x, y: point.y });
  screenPoints.push({ x: screen.x, y: screen.y });
};

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
        session = { kind: 'freehand', points: [{ ...input.documentPoint }], screenPoints: [{ ...input.screenPoint }] };
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
          appendFreehandPoint(session, sample);
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
      appendFreehandPoint(session, input);
      const polygon = session.points;
      reset();
      clearPreview(ctx);
      commit(ctx, polygon, input.modifiers);
    },
  };
};
