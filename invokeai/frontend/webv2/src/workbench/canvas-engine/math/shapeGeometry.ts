import type { Rect, Vec2 } from '@workbench/canvas-engine/types';

/** Shared outlines for raster Shapes and editable vector paths, before stroke insetting. */
export const polygonShapeVertices = (kind: 'triangle' | 'star', rect: Rect): Vec2[] => {
  const cx = rect.x + rect.width / 2;
  const cy = rect.y + rect.height / 2;
  if (kind === 'triangle') {
    return [
      { x: cx, y: rect.y },
      { x: rect.x + rect.width, y: rect.y + rect.height },
      { x: rect.x, y: rect.y + rect.height },
    ];
  }
  // Match the existing Shapes pentagram: five spikes with sin(18°) / sin(54°) inner radius.
  return Array.from({ length: 10 }, (_, index) => {
    const radius = index % 2 === 0 ? 1 : 0.382;
    const angle = -Math.PI / 2 + (index * Math.PI) / 5;
    return {
      x: cx + ((Math.cos(angle) * rect.width) / 2) * radius,
      y: cy + ((Math.sin(angle) * rect.height) / 2) * radius,
    };
  });
};
