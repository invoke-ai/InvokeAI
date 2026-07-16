/**
 * Selection-constrained raster primitives: fill and erase a layer's pixels
 * within the selection mask.
 *
 * These are the pure pixel ops behind the engine's `fillSelection` /
 * `eraseSelection` (the engine owns the layer-eligibility guards, dirty-rect
 * before/after capture, undo history, and persistence around them). Each op
 * composites through a scratch surface so the write lands ONLY where the mask is
 * opaque — pixels outside the selection are never touched — and flows entirely
 * through the {@link RasterBackend} `ctx` seam, so it runs on the node stub.
 *
 * Coordinate spaces: both the layer cache and the selection mask are content-
 * sized (placed) surfaces with their own origins. `rect` is the edit region in
 * document (= layer-local) space; `targetOrigin` / `maskOrigin` are the target
 * cache surface's and the mask surface's document-space origins, so a document
 * coordinate `d` maps to target-surface `d - targetOrigin` and mask-surface
 * `d - maskOrigin`.
 *
 * Zero React, zero import-time side effects.
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
 * Fills `color` into `target` wherever `mask` is opaque, within `rect`. Builds a
 * `color`-filled scratch (region-local), intersects it with the mask
 * (`destination-in`), then draws the masked color over the target.
 *
 * `composite` is the final blit's op: `source-over` (default) fills every masked
 * pixel; `source-atop` honours a transparency lock — colour lands ONLY where the
 * target is already opaque, never into transparent space (mirrors the
 * transparency-locked brush in `paintTool.ts`).
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
  // Keep the color only where the mask is opaque (mask drawn at its offset within
  // the region-local scratch).
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

/**
 * Erases (`destination-out`) `target`'s pixels wherever `mask` is opaque, within
 * `rect`. The mask shape is copied into a region-local scratch and composited out
 * of the target.
 */
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
