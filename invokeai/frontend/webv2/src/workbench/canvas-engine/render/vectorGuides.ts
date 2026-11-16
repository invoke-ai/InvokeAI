import type { CanvasBezierPathContract, CanvasBezierPointContract } from '@workbench/canvas-engine/contracts';
import type { Mat2d, Vec2 } from '@workbench/canvas-engine/types';

import { DEFAULT_VECTOR_GUIDE_COLOR } from '@workbench/canvas-engine/contracts';
import { applyToPoint, multiply } from '@workbench/canvas-engine/math/mat2d';

import type { RasterSurface } from './raster';

import { drawPolylineVertices, SHAPE_PREVIEW_COLOR, SHAPE_PREVIEW_DASH } from './rasterizers/shapeRasterizer';

export interface VectorGuideLayer {
  paths: readonly CanvasBezierPathContract[];
  matrix: Mat2d;
  opacity: number;
  guideColor?: string;
  /** Undefined outside Edit; null means Edit has no active path. */
  editActivePathId?: string | null;
}

/** Creation geometry is in document space; committed paths are in layer-local space. */
export interface VectorShapePreview {
  layerId: string;
  points: CanvasBezierPointContract[];
  isClosed: boolean;
  vertices: readonly Vec2[] | null;
  closeRadiusPx: number | null;
  closeArmed: boolean;
}

const GUIDE_COLOR = DEFAULT_VECTOR_GUIDE_COLOR;

/** Project controls before stroking so nonuniform layer scaling preserves the 1.5 CSS-px guide. */
const strokePath = (
  ctx: RasterSurface['ctx'],
  points: readonly CanvasBezierPointContract[],
  isClosed: boolean,
  matrix: Mat2d
): void => {
  const first = points[0];
  if (!first) {
    return;
  }
  const start = applyToPoint(matrix, first.anchor);
  ctx.beginPath();
  ctx.moveTo(start.x, start.y);
  const segments = isClosed ? points.length : points.length - 1;
  for (let index = 0; index < segments; index += 1) {
    const from = points[index]!;
    const to = points[(index + 1) % points.length]!;
    const a = applyToPoint(matrix, from.outHandle ?? from.anchor);
    const b = applyToPoint(matrix, to.inHandle ?? to.anchor);
    const end = applyToPoint(matrix, to.anchor);
    ctx.bezierCurveTo(a.x, a.y, b.x, b.y, end.x, end.y);
  }
  if (isClosed) {
    ctx.closePath();
  }
  ctx.stroke();
};

/** Overlay-only: never called by image sampling, raster export or generation composites. */
export const drawVectorGuides = (
  ctx: RasterSurface['ctx'],
  view: Mat2d,
  layers: readonly VectorGuideLayer[],
  preview: VectorShapePreview | null
): void => {
  if (layers.length === 0 && !preview) {
    return;
  }
  ctx.save();
  ctx.setLineDash([]);
  ctx.strokeStyle = GUIDE_COLOR;
  ctx.lineWidth = 1.5;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  for (const layer of layers) {
    ctx.globalAlpha = layer.opacity;
    const matrix = multiply(view, layer.matrix);
    const editing = layer.editActivePathId !== undefined;
    ctx.strokeStyle = layer.guideColor ?? GUIDE_COLOR;
    ctx.lineWidth = 1.5;
    for (const path of layer.paths) {
      if (editing && path.id === layer.editActivePathId) {
        continue;
      }
      strokePath(ctx, path.points, path.isClosed, matrix);
    }
    const active = editing ? layer.paths.find((path) => path.id === layer.editActivePathId) : null;
    if (active) {
      ctx.globalAlpha = layer.opacity;
      ctx.lineWidth = 2.25;
      strokePath(ctx, active.points, active.isClosed, matrix);
    }
  }
  ctx.globalAlpha = 1;
  if (preview) {
    ctx.strokeStyle = SHAPE_PREVIEW_COLOR;
    ctx.lineWidth = 1;
    ctx.setLineDash([...SHAPE_PREVIEW_DASH]);
    strokePath(ctx, preview.points, preview.isClosed, view);
    if (preview.vertices) {
      drawPolylineVertices(ctx, view, preview.vertices, preview.closeRadiusPx, preview.closeArmed);
    }
  }
  ctx.restore();
};
