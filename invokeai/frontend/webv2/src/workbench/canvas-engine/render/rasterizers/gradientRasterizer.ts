/**
 * Content-sized parametric gradients use explicit extent, falling back to document dimensions for legacy sources.
 * Linear angle 0 runs left-to-right over span; radial span is radius. Without anchors, center and fit the ramp to
 * extent corners.
 */

import type { CanvasLayerSourceContract } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import type { RasterizeDeps, RasterizeResult } from './types';

type GradientSource = Extract<CanvasLayerSourceContract, { type: 'gradient' }>;
type Ctx = RasterSurface['ctx'];

/** Builds the CSS-like gradient for the source across a `width`×`height` box. */
const buildGradient = (ctx: Ctx, source: GradientSource, width: number, height: number): CanvasGradient => {
  const cx = source.center?.x ?? width / 2;
  const cy = source.center?.y ?? height / 2;

  let gradient: CanvasGradient;
  if (source.kind === 'radial') {
    const radius = source.span ?? Math.hypot(width / 2, height / 2);
    gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
  } else {
    const rad = (source.angle * Math.PI) / 180;
    const dx = Math.cos(rad);
    const dy = Math.sin(rad);
    // Fitted fallback: the projection of the box's half-extents onto the
    // gradient direction, so the ramp just covers the corners.
    const half = source.span !== undefined ? source.span / 2 : (Math.abs(dx) * width + Math.abs(dy) * height) / 2;
    gradient = ctx.createLinearGradient(cx - dx * half, cy - dy * half, cx + dx * half, cy + dy * half);
  }

  for (const stop of source.stops) {
    // Clamp offsets defensively: `addColorStop` throws for out-of-range values.
    gradient.addColorStop(Math.min(1, Math.max(0, stop.offset)), stop.color);
  }
  return gradient;
};

/**
 * Rasterizes explicit or legacy document extent, reusing target. Synchronous drawing returns a resolved promise to
 * match source dispatch.
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
