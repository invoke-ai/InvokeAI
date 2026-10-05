/**
 * Parametric shapes rasterize at source width/height; compositing applies transforms. Inset strokes by half-width
 * to contain outlines. Polygons with fewer than three vertices are refused.
 */

import type { CanvasLayerSourceContract, ParametricShapeKind } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import type { RasterizeDeps, RasterizeResult } from './types';

type ShapeSource = Extract<CanvasLayerSourceContract, { type: 'shape' }>;
type Ctx = RasterSurface['ctx'];

/** Pentagram inner/outer radius ratio: sin(18°) / sin(54°). */
const STAR_INNER_RATIO = 0.382;
const STAR_SPIKES = 5;

/**
 * Shared raster/preview path inside an inset box, with round joins for sharp corners. Inset rescales the box
 * rather than uniformly offsetting edges, so slanted thick outlines may drift slightly beyond the fill silhouette.
 */
export const buildParametricShapePath = (
  ctx: Ctx,
  kind: ParametricShapeKind | 'polygon',
  x: number,
  y: number,
  width: number,
  height: number,
  inset: number,
  points?: readonly { x: number; y: number }[]
): void => {
  const w = Math.max(0, width - inset * 2);
  const h = Math.max(0, height - inset * 2);
  const cx = x + width / 2;
  const cy = y + height / 2;
  ctx.beginPath();
  if (kind === 'polygon') {
    // Vertices are stored across the un-inset box; rescale them into the
    // inset one (the same box rescale the other kinds get).
    const sx = width > 0 ? w / width : 1;
    const sy = height > 0 ? h / height : 1;
    (points ?? []).forEach((point, index) => {
      const px = x + inset + point.x * sx;
      const py = y + inset + point.y * sy;
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
    ctx.ellipse(cx, cy, w / 2, h / 2, 0, 0, Math.PI * 2);
    return;
  }
  if (kind === 'triangle') {
    ctx.moveTo(cx, cy - h / 2);
    ctx.lineTo(cx + w / 2, cy + h / 2);
    ctx.lineTo(cx - w / 2, cy + h / 2);
    ctx.closePath();
    return;
  }
  if (kind === 'star') {
    const rx = w / 2;
    const ry = h / 2;
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
  ctx.rect(x + inset, y + inset, w, h);
};

/** Shared fill-then-stroke drawing keeps shape rasterization and pixel placement consistent. */
export const drawShapeSource = (
  ctx: Ctx,
  source: ShapeSource,
  x: number,
  y: number,
  width: number,
  height: number
): void => {
  if (source.fill) {
    ctx.fillStyle = source.fill;
    buildParametricShapePath(ctx, source.kind, x, y, width, height, 0, source.points);
    ctx.fill();
  }
  if (source.stroke && source.strokeWidth > 0) {
    ctx.strokeStyle = source.stroke;
    ctx.lineWidth = source.strokeWidth;
    // Round joins keep sharp vertices within the half-stroke inset (a miter
    // would spike past it); a rect keeps its miter — the 90° tip lands exactly
    // on the extent corner, and rounding it would re-render every stored rect.
    ctx.lineJoin = source.kind === 'rect' || source.kind === 'ellipse' ? 'miter' : 'round';
    // Inset by half the stroke width so the (centered) stroke stays inside the extent.
    buildParametricShapePath(ctx, source.kind, x, y, width, height, source.strokeWidth / 2, source.points);
    ctx.stroke();
  }
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
