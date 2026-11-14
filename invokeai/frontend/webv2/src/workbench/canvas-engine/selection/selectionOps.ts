/**
 * Masked fill/erase primitives; engine callers own guards, history and persistence. Scratch compositing confines
 * edits to selection coverage. `rect` and the target live in the layer's local space; the mask is placed through
 * `maskToLocal`, so a moved, scaled or rotated layer is edited where the selection covers it.
 */

import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Mat2d, Rect, Vec2 } from '@workbench/canvas-engine/types';

import { identity, multiply, translate } from '@workbench/canvas-engine/math/mat2d';

/** Shared inputs for a masked layer edit over a layer-local `rect`. */
export interface MaskedEditParams {
  backend: RasterBackend;
  /** The layer cache surface (content-sized). */
  target: RasterSurface;
  /** The target surface's layer-local origin (its cache rect's `x`/`y`). */
  targetOrigin: Vec2;
  /** The selection mask surface (alpha 255 inside). */
  mask: RasterSurface;
  /** The mask surface's origin in its own space (its selection rect's `x`/`y`). */
  maskOrigin: Vec2;
  /** Maps the mask's space into the target's; absent when both share one space. */
  maskToLocal?: Mat2d;
  /** The dirty region to edit (layer-local), integer bounds. */
  rect: Rect;
}

/** Draws the mask into a scratch surface whose origin is `rect`'s, through `maskToLocal`. */
const drawMask = (
  ctx: RasterSurface['ctx'],
  { mask, maskOrigin, maskToLocal, rect }: Pick<MaskedEditParams, 'mask' | 'maskOrigin' | 'maskToLocal' | 'rect'>
): void => {
  if (!maskToLocal) {
    ctx.drawImage(mask.canvas, maskOrigin.x - rect.x, maskOrigin.y - rect.y);
    return;
  }
  const m = multiply(translate(identity(), { x: -rect.x, y: -rect.y }), maskToLocal);
  ctx.setTransform(m.a, m.b, m.c, m.d, m.e, m.f);
  ctx.drawImage(mask.canvas, maskOrigin.x, maskOrigin.y);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
};

/**
 * Fill a region-local scratch, intersect the mask, then composite onto target. Default source-over fills coverage;
 * source-atop preserves transparency lock.
 */
export const fillMaskedRegion = ({
  backend,
  color,
  composite = 'source-over',
  mask,
  maskOrigin,
  maskToLocal,
  rect,
  target,
  targetOrigin,
}: MaskedEditParams & { color: string; composite?: 'source-over' | 'source-atop' }): void => {
  if (rect.width <= 0 || rect.height <= 0) {
    return;
  }
  const scratch = backend.createSurface(rect.width, rect.height);
  const sctx = scratch.ctx;
  sctx.setTransform(1, 0, 0, 1, 0, 0);
  sctx.globalCompositeOperation = 'source-over';
  sctx.globalAlpha = 1;
  sctx.fillStyle = color;
  sctx.fillRect(0, 0, rect.width, rect.height);
  sctx.globalCompositeOperation = 'destination-in';
  drawMask(sctx, { mask, maskOrigin, maskToLocal, rect });

  const ctx = target.ctx;
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = composite;
  ctx.globalAlpha = 1;
  ctx.drawImage(scratch.canvas, rect.x - targetOrigin.x, rect.y - targetOrigin.y);
  ctx.restore();
};

export const eraseMaskedRegion = ({
  backend,
  mask,
  maskOrigin,
  maskToLocal,
  rect,
  target,
  targetOrigin,
}: MaskedEditParams): void => {
  if (rect.width <= 0 || rect.height <= 0) {
    return;
  }
  const scratch = backend.createSurface(rect.width, rect.height);
  const sctx = scratch.ctx;
  sctx.setTransform(1, 0, 0, 1, 0, 0);
  sctx.globalCompositeOperation = 'source-over';
  sctx.globalAlpha = 1;
  drawMask(sctx, { mask, maskOrigin, maskToLocal, rect });

  const ctx = target.ctx;
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = 'destination-out';
  ctx.globalAlpha = 1;
  ctx.drawImage(scratch.canvas, rect.x - targetOrigin.x, rect.y - targetOrigin.y);
  ctx.restore();
};
