import type { SplatRect } from 'features/controlLayers/components/SplatOverlay/state';

export type RectCorner = 'nw' | 'ne' | 'sw' | 'se';

type ResizeOptions = {
  /** Preserve the starting rect's aspect ratio (dominant-axis scale). */
  keepAspect: boolean;
  /** Minimum width/height in world px. */
  minSize: number;
};

/** Translate a rect by a world-space delta. */
export const moveRect = (start: SplatRect, dx: number, dy: number): SplatRect => ({
  ...start,
  x: start.x + dx,
  y: start.y + dy,
});

/**
 * Resize a rect by dragging one of its corners by a world-space delta, keeping the opposite corner
 * anchored. With `keepAspect`, the axis with the larger relative change drives a uniform scale.
 */
export const resizeRectFromCorner = (
  start: SplatRect,
  corner: RectCorner,
  dx: number,
  dy: number,
  options: ResizeOptions
): SplatRect => {
  const { keepAspect, minSize } = options;
  const signX = corner === 'nw' || corner === 'sw' ? -1 : 1;
  const signY = corner === 'nw' || corner === 'ne' ? -1 : 1;

  let width = start.width + signX * dx;
  let height = start.height + signY * dy;

  if (keepAspect && start.width > 0 && start.height > 0) {
    const scaleX = width / start.width;
    const scaleY = height / start.height;
    // The axis the user pulled further (relative to the rect's size) wins.
    let scale = Math.abs(scaleX - 1) >= Math.abs(scaleY - 1) ? scaleX : scaleY;
    scale = Math.max(scale, minSize / start.width, minSize / start.height);
    width = start.width * scale;
    height = start.height * scale;
  } else {
    width = Math.max(minSize, width);
    height = Math.max(minSize, height);
  }

  return {
    x: signX === -1 ? start.x + start.width - width : start.x,
    y: signY === -1 ? start.y + start.height - height : start.y,
    width,
    height,
  };
};
