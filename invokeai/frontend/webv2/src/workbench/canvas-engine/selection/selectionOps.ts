/**
 * Masked fill/erase primitives; engine callers own guards, history and persistence. Scratch compositing confines
 * edits to selection coverage. Rect coordinates are document=layer-local here; subtract target/mask origins to
 * address each content-sized surface.
 */

import type { RasterBackend, RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { Rect, Vec2 } from '@workbench/canvas-engine/types';

/** Shared inputs for a masked layer edit over a document-space `rect`. */
export interface MaskedEditParams {
  backend: RasterBackend;
  /** The layer cache surface (content-sized) and its document-space origin. */
  target: RasterSurface;
  /** The target surface's document-space origin (its cache rect's `x`/`y`). */
  targetOrigin: Vec2;
  /** The selection mask surface (alpha 255 inside). */
  mask: RasterSurface;
  /** The mask surface's document-space origin (its selection rect's `x`/`y`). */
  maskOrigin: Vec2;
  /** The dirty region to edit (document space), integer bounds. */
  rect: Rect;
}

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
  sctx.drawImage(mask.canvas, maskOrigin.x - rect.x, maskOrigin.y - rect.y);

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
  sctx.drawImage(mask.canvas, maskOrigin.x - rect.x, maskOrigin.y - rect.y);

  const ctx = target.ctx;
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.globalCompositeOperation = 'destination-out';
  ctx.globalAlpha = 1;
  ctx.drawImage(scratch.canvas, rect.x - targetOrigin.x, rect.y - targetOrigin.y);
  ctx.restore();
};
