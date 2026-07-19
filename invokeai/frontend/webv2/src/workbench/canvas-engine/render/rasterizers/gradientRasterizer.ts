/**
 * Rasterizes a `gradient` layer source (linear / radial). Gradient layers are
 * PARAMETRIC and CONTENT-SIZED: the source carries an explicit `width`/`height`
 * extent (set bbox-sized at creation, preserved across angle edits) and the
 * gradient spans that extent. Legacy gradients that predate the extent field
 * default to the document dims (see {@link RasterizeDeps.documentSize}), so they
 * render identically. The compositor positions the extent via the layer transform.
 *
 * Linear: the gradient line runs through the extent center along `angle`
 * (degrees; 0° = left→right), long enough to cover the box corners. Radial:
 * centered on the extent, radius reaching the farthest corner.
 *
 * Zero React, zero import-time side effects.
 */

import type { CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import type { RasterizeDeps, RasterizeResult } from './types';

type GradientSource = Extract<CanvasLayerSourceContract, { type: 'gradient' }>;
type Ctx = RasterSurface['ctx'];

/** Builds the CSS-like gradient for the source across a `width`×`height` box. */
const buildGradient = (ctx: Ctx, source: GradientSource, width: number, height: number): CanvasGradient => {
  const cx = width / 2;
  const cy = height / 2;

  let gradient: CanvasGradient;
  if (source.kind === 'radial') {
    const radius = Math.hypot(width / 2, height / 2);
    gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
  } else {
    const rad = (source.angle * Math.PI) / 180;
    const dx = Math.cos(rad);
    const dy = Math.sin(rad);
    // Half-length along the direction that just covers the box (projection of
    // the box's half-extents onto the gradient direction).
    const half = (Math.abs(dx) * width + Math.abs(dy) * height) / 2;
    gradient = ctx.createLinearGradient(cx - dx * half, cy - dy * half, cx + dx * half, cy + dy * half);
  }

  for (const stop of source.stops) {
    // Clamp offsets defensively: `addColorStop` throws for out-of-range values.
    gradient.addColorStop(Math.min(1, Math.max(0, stop.offset)), stop.color);
  }
  return gradient;
};

/**
 * Draws a gradient source into a surface sized to its explicit extent (or the
 * document dims for legacy gradients), reusing `target` if provided. Synchronous
 * work wrapped in a resolved promise so it shares the `rasterizeSource` dispatch
 * signature.
 */
export const rasterizeGradientSource = (
  source: GradientSource,
  deps: RasterizeDeps,
  target?: RasterSurface
): Promise<RasterizeResult> => {
  const width = Math.max(1, Math.round(source.width ?? deps.documentSize.width));
  const height = Math.max(1, Math.round(source.height ?? deps.documentSize.height));

  const surface = target ?? deps.backend.createSurface(width, height);
  if (surface.width !== width || surface.height !== height) {
    surface.resize(width, height);
  }
  const { ctx } = surface;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, width, height);

  if (source.stops.length > 0) {
    ctx.fillStyle = buildGradient(ctx, source, width, height);
    ctx.fillRect(0, 0, width, height);
  }

  return Promise.resolve({ rect: { height, width, x: 0, y: 0 }, surface });
};
