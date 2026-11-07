/**
 * Parametric shapes rasterize at source width/height; compositing applies transforms. Inset strokes by half-width
 * to contain outlines. Polygons with fewer than three vertices are refused.
 */

import type { CanvasLayerSourceContract, ParametricShapeKind } from '@workbench/canvas-engine/contracts';
import type { RasterSurface } from '@workbench/canvas-engine/render/raster';

import type { RasterizeDeps, RasterizeResult } from './types';

import { polygonShapeVertices } from '@workbench/canvas-engine/math/shapeGeometry';

type ShapeSource = Extract<CanvasLayerSourceContract, { type: 'shape' }>;
type Ctx = RasterSurface['ctx'];

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
  if (kind === 'triangle' || kind === 'star') {
    const vertices = polygonShapeVertices(kind, { x: x + inset, y: y + inset, width: w, height: h });
    for (let index = 0; index < vertices.length; index += 1) {
      const point = vertices[index]!;
      if (index === 0) {
        ctx.moveTo(point.x, point.y);
      } else {
        ctx.lineTo(point.x, point.y);
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
