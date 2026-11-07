import type { Mat2d } from '@workbench/canvas-engine/types';
import type { VectorEditSession } from '@workbench/canvas-engine/vectorEditState';

import { applyToPoint, multiply } from '@workbench/canvas-engine/math/mat2d';
import { transformBounds } from '@workbench/canvas-engine/math/rect';

import type { RasterSurface } from './raster';

export interface VectorEditOverlay {
  session: VectorEditSession;
  matrix: Mat2d;
}

/** Editable anchors/handles use screen-sized hit targets; guide curves are drawn by vectorGuides. */
export const drawVectorEditOverlay = (ctx: RasterSurface['ctx'], view: Mat2d, overlay: VectorEditOverlay | null): void => {
  if (!overlay) { return; }
  const { session } = overlay;
  const matrix = multiply(view, overlay.matrix);
  const selected = new Set(session.selectedPoints.map((ref) => `${ref.pathId}:${ref.pointIndex}`));
  const active = session.selectedPoints.at(-1);
  ctx.save();
  ctx.setLineDash([]);
  ctx.lineWidth = 1;
  ctx.strokeStyle = 'rgba(90, 175, 255, 0.75)';
  for (const ref of session.selectedPoints) {
    const point = session.paths.find((path) => path.id === ref.pathId)?.points[ref.pointIndex];
    if (!point) { continue; }
    const anchor = applyToPoint(matrix, point.anchor);
    for (const handle of [point.inHandle, point.outHandle]) {
      if (!handle) { continue; }
      const p = applyToPoint(matrix, handle);
      ctx.beginPath(); ctx.moveTo(anchor.x, anchor.y); ctx.lineTo(p.x, p.y); ctx.stroke();
      ctx.beginPath(); ctx.arc(p.x, p.y, 3.5, 0, Math.PI * 2);
      ctx.fillStyle = '#fff'; ctx.fill(); ctx.stroke();
    }
  }
  for (const path of session.paths) {
    path.points.forEach((point, pointIndex) => {
      const isSelected = selected.has(`${path.id}:${pointIndex}`);
      const isActive = active?.pathId === path.id && active.pointIndex === pointIndex;
      const p = applyToPoint(matrix, point.anchor);
      ctx.beginPath(); ctx.arc(p.x, p.y, 4, 0, Math.PI * 2);
      ctx.lineWidth = 2;
      ctx.fillStyle = isSelected ? 'rgba(90, 175, 255, 0.8)' : '#fff';
      ctx.strokeStyle = isActive ? '#3ed684' : isSelected ? '#fff' : 'rgba(90, 175, 255, 1)';
      ctx.fill(); ctx.stroke();
    });
  }
  if (session.selectionRect) {
    const rect = transformBounds(view, session.selectionRect);
    ctx.fillStyle = 'rgba(90, 175, 255, 0.12)';
    ctx.strokeStyle = 'rgba(90, 175, 255, 0.9)';
    ctx.lineWidth = 1;
    ctx.fillRect(rect.x, rect.y, rect.width, rect.height);
    ctx.strokeRect(rect.x, rect.y, rect.width, rect.height);
  }
  ctx.restore();
};
