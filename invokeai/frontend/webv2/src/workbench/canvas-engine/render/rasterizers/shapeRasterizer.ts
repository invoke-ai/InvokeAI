/**
 * Parametric shapes rasterize at source width/height; compositing applies transforms. Fill and inside-only strokes
 * share one silhouette mask, so acute and concave outlines contain the fill without color fringes.
 */

import type { CanvasLayerSourceContract, ParametricShapeKind } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import type { RasterizeDeps, RasterizeResult } from './types';

type ShapeSource = Extract<CanvasLayerSourceContract, { type: 'shape' }>;
type Ctx = RasterSurface['ctx'];

/** Pentagram inner/outer radius ratio: sin(18°) / sin(54°). */
const STAR_INNER_RATIO = 0.382;
const STAR_SPIKES = 5;

/** The original shape silhouette, shared by rasterized fill, its clipped outline and the tool preview. */
export const buildParametricShapePath = (
  ctx: Ctx,
  kind: ParametricShapeKind | 'polygon',
  x: number,
  y: number,
  width: number,
  height: number,
  points?: readonly { x: number; y: number }[]
): void => {
  const cx = x + width / 2;
  const cy = y + height / 2;
  ctx.beginPath();
  if (kind === 'polygon') {
    (points ?? []).forEach((point, index) => {
      const px = x + point.x;
      const py = y + point.y;
      if (index === 0) {
        ctx.moveTo(px, py);
      } else {
        ctx.lineTo(px, py);
      }
    });
    ctx.closePath();
    return;
  }
  if (kind === 'ellipse') {
    ctx.ellipse(cx, cy, width / 2, height / 2, 0, 0, Math.PI * 2);
    return;
  }
  if (kind === 'triangle') {
    ctx.moveTo(cx, cy - height / 2);
    ctx.lineTo(cx + width / 2, cy + height / 2);
    ctx.lineTo(cx - width / 2, cy + height / 2);
    ctx.closePath();
    return;
  }
  if (kind === 'star') {
    const rx = width / 2;
    const ry = height / 2;
    for (let i = 0; i < STAR_SPIKES * 2; i += 1) {
      const radius = i % 2 === 0 ? 1 : STAR_INNER_RATIO;
      const angle = -Math.PI / 2 + (i * Math.PI) / STAR_SPIKES;
      const px = cx + Math.cos(angle) * rx * radius;
      const py = cy + Math.sin(angle) * ry * radius;
      if (i === 0) {
        ctx.moveTo(px, py);
      } else {
        ctx.lineTo(px, py);
      }
    }
    ctx.closePath();
    return;
  }
  ctx.rect(x, y, width, height);
};

/**
 * Draw into an isolated transparent target: the shape rasterizer clears it, and the Shape tool supplies fresh
 * scratch pixels. With an outline, compose the styles first and mask them to the silhouette once; clipping the
 * stroke over an already-antialiased fill would expose fill color along the exterior edge. The final mask clears
 * everything outside the shape, so this must never draw directly over existing layer pixels.
 */
export const drawShapeSource = (
  ctx: Ctx,
  source: ShapeSource,
  x: number,
  y: number,
  width: number,
  height: number
): void => {
  const stroke = source.strokeWidth > 0 ? source.stroke : null;
  if (!source.fill && !stroke) {
    return;
  }
  ctx.save();
  ctx.globalCompositeOperation = 'source-over';
  buildParametricShapePath(ctx, source.kind, x, y, width, height, source.points);
  if (stroke) {
    if (source.fill) {
      // Cover the target, not the transformed shape bounds: the silhouette must be its only antialiased edge.
      ctx.save();
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = source.fill;
      ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
      ctx.restore();
    }
    ctx.strokeStyle = stroke;
    ctx.lineWidth = source.strokeWidth * 2;
    ctx.lineJoin = source.kind === 'rect' || source.kind === 'ellipse' ? 'miter' : 'round';
    ctx.stroke();
    // Canvas strokes are centered; one mask keeps the inside half at the configured width and applies coverage once.
    ctx.globalCompositeOperation = 'destination-in';
    ctx.globalAlpha = 1;
    ctx.fillStyle = '#000000';
    ctx.fill();
  } else if (source.fill) {
    ctx.fillStyle = source.fill;
    ctx.fill();
  }
  ctx.restore();
};

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

  drawShapeSource(ctx, source, 0, 0, width, height);

  return Promise.resolve({ rect: { height, width, x: 0, y: 0 }, surface });
};
