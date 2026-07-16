/**
 * The lasso tool: freehand pixel-selection.
 *
 * A primary-button drag accumulates document-space polygon points (coalesced by
 * the pipeline, then distance-decimated here like the brush thins its input); a
 * live dashed preview of the in-progress path is published to the engine's
 * `lassoPreview` store (a transient channel — no dispatch, no reducer traffic).
 * Pointer-up closes the path and commits it to the engine's selection through
 * {@link ToolContext.commitSelection}, with the boolean op resolved from the
 * modifiers (shift = add, alt = subtract, shift+alt = intersect) or, with none
 * held, the persistent op mode from `lassoOptions`. Escape / pointercancel drops
 * the in-progress path.
 *
 * Selection edits are transient interaction state: they are NOT dispatches and
 * NOT recorded on the engine's undo history this phase (legacy parity — selection
 * changes aren't undoable). Zero React, zero import-time side effects.
 */

import type { PointerInput, SelectionOp, Vec2 } from '@workbench/canvas-engine/types';

import { polygonBounds, polygonToSvgPath } from '@workbench/canvas-engine/freehand';

import type { Tool, ToolContext } from './tool';

/** Bit for the primary (usually left) mouse button in `PointerEvent.buttons`. */
const PRIMARY_BUTTON = 1;

/** Minimum document-space gap between stored polygon points (input decimation). */
export const LASSO_MIN_POINT_DISTANCE = 2;

/** Fewest distinct points that make a fillable selection polygon. */
const MIN_POLYGON_POINTS = 3;

/** Resolves the boolean op for a commit: modifiers win, else the persistent mode. */
export const lassoOpFor = (modifiers: PointerInput['modifiers'], mode: SelectionOp): SelectionOp => {
  if (modifiers.shift && modifiers.alt) {
    return 'intersect';
  }
  if (modifiers.shift) {
    return 'add';
  }
  if (modifiers.alt) {
    return 'subtract';
  }
  return mode;
};

const distance = (a: Vec2, b: Vec2): number => Math.hypot(a.x - b.x, a.y - b.y);

/** Creates a fresh lasso tool with its own per-gesture point buffer. */
export const createLassoTool = (): Tool => {
  let points: Vec2[] = [];
  let active = false;

  const reset = (): void => {
    points = [];
    active = false;
  };

  /** Appends a point if it is far enough from the last stored one. */
  const pushPoint = (p: Vec2): void => {
    const last = points[points.length - 1];
    if (!last || distance(last, p) >= LASSO_MIN_POINT_DISTANCE) {
      points.push({ x: p.x, y: p.y });
    }
  };

  const publishPreview = (ctx: ToolContext): void => {
    ctx.stores.lassoPreview.set(points.length > 0 ? points.slice() : null);
    ctx.invalidate({ overlay: true });
  };

  const clearPreview = (ctx: ToolContext): void => {
    ctx.stores.lassoPreview.set(null);
    ctx.invalidate({ overlay: true });
  };

  return {
    cursor: () => 'crosshair',
    id: 'lasso',
    onDeactivate: (ctx) => {
      reset();
      clearPreview(ctx);
    },
    onPointerCancel: (ctx) => {
      reset();
      clearPreview(ctx);
    },
    onPointerDown: (ctx, input) => {
      if (active || (input.buttons & PRIMARY_BUTTON) === 0) {
        return;
      }
      active = true;
      points = [{ x: input.documentPoint.x, y: input.documentPoint.y }];
      publishPreview(ctx);
    },
    onPointerMove: (ctx, input, batch) => {
      if (!active) {
        return;
      }
      for (const sample of batch) {
        pushPoint(sample.documentPoint);
      }
      publishPreview(ctx);
    },
    onPointerUp: (ctx, input) => {
      if (!active) {
        return;
      }
      pushPoint(input.documentPoint);
      const polygon = points.slice();
      reset();
      clearPreview(ctx);
      if (polygon.length < MIN_POLYGON_POINTS || !ctx.commitSelection) {
        return;
      }
      const path = ctx.createPath2D(polygonToSvgPath(polygon));
      const bounds = polygonBounds(polygon);
      const op = lassoOpFor(input.modifiers, ctx.stores.lassoOptions.get().mode);
      ctx.commitSelection({ bounds, op, path });
    },
  };
};
