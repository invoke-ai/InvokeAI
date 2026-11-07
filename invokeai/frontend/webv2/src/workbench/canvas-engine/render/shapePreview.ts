import type { Mat2d, Vec2 } from '@workbench/canvas-engine/types';

import { applyToPoint } from '@workbench/canvas-engine/math/mat2d';

import type { RasterSurface } from './raster';

export const SHAPE_PREVIEW_COLOR = '#38bdf8';
export const SHAPE_PREVIEW_DASH: readonly number[] = [4, 4];
const VERTEX_HALF_PX = 2;

/** Shared Shapes/polyline markers in CSS pixels: vertices and a close target filled while armed. */
export const drawPolylineVertices = (
  ctx: RasterSurface['ctx'],
  view: Mat2d,
  points: readonly Vec2[],
  closeRadiusPx: number | null,
  closeArmed: boolean
): void => {
  ctx.save();
  ctx.setLineDash([]);
  ctx.strokeStyle = SHAPE_PREVIEW_COLOR;
  ctx.fillStyle = SHAPE_PREVIEW_COLOR;
  ctx.lineWidth = 1;
  points.forEach((point, index) => {
    const p = applyToPoint(view, point);
    if (index === 0 && closeRadiusPx !== null) {
      ctx.beginPath();
      ctx.arc(p.x, p.y, closeRadiusPx, 0, Math.PI * 2);
      if (closeArmed) {
        ctx.fill();
      }
      ctx.stroke();
    } else {
      ctx.fillRect(p.x - VERTEX_HALF_PX, p.y - VERTEX_HALF_PX, VERTEX_HALF_PX * 2, VERTEX_HALF_PX * 2);
    }
  });
  ctx.restore();
};
