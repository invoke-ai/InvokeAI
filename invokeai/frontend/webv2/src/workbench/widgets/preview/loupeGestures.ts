import type { PanZoomPoint } from '@workbench/panZoom';

/** Share pointer tracking between stage and comparison loupes; each uses its own zoom coordinate space. */

/** Reset on the first touch or mouse press to discard pointers whose releases were missed before arming a pinch. */
export const trackPointerDown = (
  pointers: Map<number, PanZoomPoint>,
  event: { clientX: number; clientY: number; isPrimary: boolean; pointerId: number }
): [number, number] | null => {
  if (event.isPrimary) {
    pointers.clear();
  }

  pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

  if (pointers.size !== 2) {
    return null;
  }

  const [first, second] = [...pointers.keys()];

  return [first!, second!];
};

/**
 * Tolerate capture failure for already-released pointers; uncaptured gestures still work while pointers remain on
 * the surface.
 */
export const capturePointer = (element: Element, pointerId: number): void => {
  try {
    element.setPointerCapture(pointerId);
  } catch {
    // Pointer ended before the gesture could claim it.
  }
};

/** Releases a capture this surface holds, if it holds one. */
export const releasePointer = (element: Element, pointerId: number): void => {
  if (element.hasPointerCapture(pointerId)) {
    element.releasePointerCapture(pointerId);
  }
};
