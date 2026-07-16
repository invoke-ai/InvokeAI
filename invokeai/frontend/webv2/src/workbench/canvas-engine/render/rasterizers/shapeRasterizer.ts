/**
 * Rasterizes a `shape` layer source (rect / ellipse). Shape layers are
 * PARAMETRIC: their pixels are derived from the source params (`width`,
 * `height`, `fill`, `stroke`, `strokeWidth`, `kind`) rather than a persisted
 * bitmap, so they re-render for free whenever a param changes.
 *
 * Extent semantics: a shape's surface is sized to the source's own
 * `width`×`height` (its layer-local extent), NOT the document — the compositor
 * applies the layer transform (position/scale/rotation) when drawing, exactly
 * like an `image` layer. The stroke is drawn INSET by `strokeWidth / 2` so a
 * thick outline stays entirely within the extent rather than clipping at the
 * surface edge.
 *
 * `polygon` is intentionally NOT handled here (deferred until a points-editing
 * UX exists); the dispatch never routes a polygon source to this rasterizer in
 * this phase.
 *
 * Zero React, zero import-time side effects.
 */

import type { RasterSurface } from '@workbench/canvas-engine/render/raster';
import type { CanvasLayerSourceContract } from '@workbench/types';

import type { RasterizeDeps, RasterizeResult } from './types';

type ShapeSource = Extract<CanvasLayerSourceContract, { type: 'shape' }>;
type Ctx = RasterSurface['ctx'];

/** Builds the rect/ellipse path for `kind` inset by `inset` on every side into the current path. */
const buildShapePath = (ctx: Ctx, kind: ShapeSource['kind'], width: number, height: number, inset: number): void => {
  const w = Math.max(0, width - inset * 2);
  const h = Math.max(0, height - inset * 2);
  ctx.beginPath();
  if (kind === 'ellipse') {
    ctx.ellipse(width / 2, height / 2, w / 2, h / 2, 0, 0, Math.PI * 2);
    return;
  }
  // `rect` (and, defensively, `polygon` which the dispatch never sends here).
  ctx.rect(inset, inset, w, h);
};

/**
 * Draws a shape source onto a surface sized to the source extent. Reuses
 * `target` if provided (resizing it to the extent), matching the paint/image
 * rasterizer contract. Synchronous work wrapped in a resolved promise so it
 * shares the `rasterizeSource` dispatch signature.
 */
export const rasterizeShapeSource = (
  source: ShapeSource,
  deps: RasterizeDeps,
  target?: RasterSurface
): Promise<RasterizeResult> => {
  const width = Math.max(1, Math.round(source.width));
  const height = Math.max(1, Math.round(source.height));

  const surface = target ?? deps.backend.createSurface(width, height);
  if (surface.width !== width || surface.height !== height) {
    surface.resize(width, height);
  }
  const { ctx } = surface;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, width, height);

  if (source.fill) {
    ctx.fillStyle = source.fill;
    buildShapePath(ctx, source.kind, width, height, 0);
    ctx.fill();
  }

  if (source.stroke && source.strokeWidth > 0) {
    ctx.strokeStyle = source.stroke;
    ctx.lineWidth = source.strokeWidth;
    // Inset by half the stroke width so the (centered) stroke stays inside the extent.
    buildShapePath(ctx, source.kind, width, height, source.strokeWidth / 2);
    ctx.stroke();
  }

  return Promise.resolve({ rect: { height, width, x: 0, y: 0 }, surface });
};
