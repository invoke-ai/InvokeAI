/**
 * Shared DOM-free transitions use screen = zoom * content + pan. Clamp zoom before anchor translation so hitting a
 * limit does not pan.
 */

/** Wheel exponential zoom sensitivity: `zoom *= exp(-deltaY * step)`. */
export const WHEEL_ZOOM_STEP = 0.0015;

export interface PanZoomPoint {
  x: number;
  y: number;
}

export interface PanZoomTransform {
  pan: PanZoomPoint;
  zoom: number;
}

export type ConstrainZoom = (zoom: number) => number;

/** Separation of two pointers, for measuring a pinch. */
export const distanceBetween = (a: PanZoomPoint, b: PanZoomPoint): number => Math.hypot(a.x - b.x, a.y - b.y);

/** Point halfway between two pointers — what a pinch zooms and pans around. */
export const midpointOf = (a: PanZoomPoint, b: PanZoomPoint): PanZoomPoint => ({
  x: (a.x + b.x) / 2,
  y: (a.y + b.y) / 2,
});

/** Sets zoom while keeping the content point under `screenAnchor` fixed. */
export const zoomAtPoint = (
  transform: PanZoomTransform,
  requestedZoom: number,
  screenAnchor: PanZoomPoint,
  constrainZoom: ConstrainZoom
): PanZoomTransform => {
  const zoom = constrainZoom(requestedZoom);

  if (zoom === transform.zoom) {
    return transform;
  }

  const contentAnchor = {
    x: (screenAnchor.x - transform.pan.x) / transform.zoom,
    y: (screenAnchor.y - transform.pan.y) / transform.zoom,
  };

  return {
    pan: {
      x: screenAnchor.x - zoom * contentAnchor.x,
      y: screenAnchor.y - zoom * contentAnchor.y,
    },
    zoom,
  };
};

/** Applies exponential wheel zoom around `screenAnchor`. */
export const wheelZoomAtPoint = (
  transform: PanZoomTransform,
  deltaY: number,
  screenAnchor: PanZoomPoint,
  { constrainZoom, step = WHEEL_ZOOM_STEP }: { constrainZoom: ConstrainZoom; step?: number }
): PanZoomTransform => {
  const target = constrainZoom(transform.zoom * Math.exp(-deltaY * step));

  return zoomAtPoint(transform, target, screenAnchor, constrainZoom);
};

/** Pans by a screen-space delta. */
export const panBy = (transform: PanZoomTransform, screenDelta: PanZoomPoint): PanZoomTransform => ({
  pan: { x: transform.pan.x + screenDelta.x, y: transform.pan.y + screenDelta.y },
  zoom: transform.zoom,
});

/**
 * Measure every pinch move from the starting transform. Carry its anchored content point to the current midpoint,
 * allowing pan at zoom limits without accumulated drift.
 */
export const pinchZoomAtPoints = (
  start: PanZoomTransform,
  gesture: { center: PanZoomPoint; distance: number; startCenter: PanZoomPoint; startDistance: number },
  constrainZoom: ConstrainZoom
): PanZoomTransform => {
  if (gesture.startDistance <= 0) {
    return start;
  }

  const zoom = constrainZoom(start.zoom * (gesture.distance / gesture.startDistance));
  const contentAnchor = {
    x: (gesture.startCenter.x - start.pan.x) / start.zoom,
    y: (gesture.startCenter.y - start.pan.y) / start.zoom,
  };

  return {
    pan: {
      x: gesture.center.x - zoom * contentAnchor.x,
      y: gesture.center.y - zoom * contentAnchor.y,
    },
    zoom,
  };
};
