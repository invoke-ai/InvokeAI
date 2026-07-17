/**
 * DOM-free pan/zoom state transitions shared by every workbench viewport.
 *
 * Coordinate convention: `screen = zoom * content + pan`. Callers provide
 * their own zoom constraint because canvas, preview, and comparison viewports
 * have different limits. Constraints are applied before anchor translation is
 * calculated, so reaching a zoom limit is an exact no-op instead of becoming
 * an accidental pan.
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
  {
    constrainZoom,
    snapZoom = (zoom) => zoom,
    step = WHEEL_ZOOM_STEP,
  }: {
    constrainZoom: ConstrainZoom;
    snapZoom?: (zoom: number) => number;
    step?: number;
  }
): PanZoomTransform => {
  const target = constrainZoom(transform.zoom * Math.exp(-deltaY * step));

  return zoomAtPoint(transform, snapZoom(target), screenAnchor, constrainZoom);
};

/** Pans by a screen-space delta. */
export const panBy = (transform: PanZoomTransform, screenDelta: PanZoomPoint): PanZoomTransform => ({
  pan: { x: transform.pan.x + screenDelta.x, y: transform.pan.y + screenDelta.y },
  zoom: transform.zoom,
});
