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

/** Zoom levels wheel zoom rests on (1 = 100%), as the v6 canvas did. */
export const WHEEL_ZOOM_SNAP_POINTS: readonly number[] = [0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5];

/** How near a snap point, as a fraction of it, wheel zoom must come to rest on it, and how far to leave it. */
export const WHEEL_ZOOM_SNAP_TOLERANCE = 0.02;

/**
 * A pause in wheel events longer than this ends a wheel gesture: the next event starts from the zoom shown, so wheel
 * movement left over inside a snap point's tolerance never carries into a later gesture (v6's snap timeout).
 */
export const WHEEL_ZOOM_SNAP_IDLE_MS = 300;

/**
 * Wheel zoom's memory between events. The zoom the wheel asks for (`intendedZoom`) is tracked apart from the zoom
 * shown, so a resting snap point holds until the accumulated wheel movement leaves its tolerance, then zoom continues
 * from where the wheel would have been. Any other zoom change (fit, preset, hotkey, pinch) discards it.
 */
export interface WheelZoomSnap {
  /** The snap point the zoom rests on, or null. */
  activeSnapPoint: number | null;
  intendedZoom: number;
  /** The zoom this state produced; a current zoom that differs means another change intervened. */
  zoom: number;
  /** When the wheel event that produced it arrived, in the caller's clock (milliseconds). */
  at: number;
}

/** Relative difference within which two zooms are the same one, such as a zoom converted through a fit ratio and back. */
const SAME_ZOOM_EPSILON = 1e-9;

/**
 * One exponential wheel step (`zoom *= exp(-deltaY * step)`) with sticky snapping: resting on a snap point once the
 * intended zoom comes within tolerance of it, until the intended zoom leaves that tolerance. Pass the previous
 * result's `snap`, or null to start from `currentZoom`, and the event's time as `now`. After a pause longer than
 * {@link WHEEL_ZOOM_SNAP_IDLE_MS} the intended zoom starts again from `currentZoom`, still resting on its snap point.
 * Zoom values are actual zoom (1 = 100%).
 */
export const snapWheelZoom = (
  snap: WheelZoomSnap | null,
  currentZoom: number,
  deltaY: number,
  { constrainZoom, now, step = WHEEL_ZOOM_STEP }: { constrainZoom: ConstrainZoom; now: number; step?: number }
): WheelZoomSnap => {
  const current = snap !== null && Math.abs(snap.zoom - currentZoom) <= currentZoom * SAME_ZOOM_EPSILON ? snap : null;
  const isSameGesture = current !== null && now - current.at <= WHEEL_ZOOM_SNAP_IDLE_MS;
  const fromZoom = isSameGesture ? current.intendedZoom : currentZoom;
  const intendedZoom = constrainZoom(fromZoom * Math.exp(-deltaY * step));
  const active = current?.activeSnapPoint ?? null;

  if (active !== null && Math.abs(intendedZoom - active) <= active * WHEEL_ZOOM_SNAP_TOLERANCE) {
    return { activeSnapPoint: active, at: now, intendedZoom, zoom: active };
  }

  const snapPoint = WHEEL_ZOOM_SNAP_POINTS.find(
    (point) =>
      Math.abs(constrainZoom(point) - point) <= point * SAME_ZOOM_EPSILON &&
      Math.abs(intendedZoom - point) < point * WHEEL_ZOOM_SNAP_TOLERANCE
  );

  return snapPoint === undefined
    ? { activeSnapPoint: null, at: now, intendedZoom, zoom: intendedZoom }
    : { activeSnapPoint: snapPoint, at: now, intendedZoom, zoom: snapPoint };
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
