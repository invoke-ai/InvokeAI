import type { AxisRanges } from './imageMapViewport';

import { zoomFactorFromWheel, zoomRangesAroundFraction } from './imageMapViewport';

/**
 * Custom wheel + two-finger-pinch zoom for the plotly map. Plotly's built-in
 * scrollZoom is deliberately off (long-standing Safari/trackpad issues —
 * PhotoMapAI replaced it the same way); single-finger drag still uses
 * plotly's native pan.
 */

interface ZoomHost {
  /** Read the current axis ranges (null while plotly is initializing). */
  readRanges: () => AxisRanges | null;
  /** Apply new axis ranges (a plotly relayout). */
  applyRanges: (ranges: AxisRanges) => void;
  /** Stamp pinch activity so a finger-lift is not treated as a click. */
  onPinch?: () => void;
}

const touchDistance = (touches: TouchList): number => {
  const dx = touches[0].clientX - touches[1].clientX;
  const dy = touches[0].clientY - touches[1].clientY;

  return Math.hypot(dx, dy);
};

export const attachWheelZoom = (element: HTMLElement, host: ZoomHost): (() => void) => {
  const zoomAtClientPoint = (clientX: number, clientY: number, factor: number) => {
    const ranges = host.readRanges();

    if (!ranges) {
      return;
    }

    const rect = element.getBoundingClientRect();

    if (rect.width === 0 || rect.height === 0) {
      return;
    }

    const fractionX = (clientX - rect.left) / rect.width;
    // Screen y grows downward; data y grows upward.
    const fractionY = 1 - (clientY - rect.top) / rect.height;
    host.applyRanges(zoomRangesAroundFraction(ranges, fractionX, fractionY, factor));
  };

  const handleWheel = (event: WheelEvent) => {
    // Also stops ctrl+wheel browser page zoom over the map.
    event.preventDefault();
    zoomAtClientPoint(event.clientX, event.clientY, zoomFactorFromWheel(event.deltaY, event.ctrlKey));
  };

  let pinchDistance: number | null = null;

  const handleTouchStart = (event: TouchEvent) => {
    if (event.touches.length === 2) {
      pinchDistance = touchDistance(event.touches);
    }
  };

  const handleTouchMove = (event: TouchEvent) => {
    if (event.touches.length !== 2 || pinchDistance === null) {
      return;
    }

    // Coincident touches at gesture start would make the first factor 0 and
    // collapse the viewport; use the first separated move as the baseline.
    if (pinchDistance === 0) {
      pinchDistance = touchDistance(event.touches);

      return;
    }

    // Capture-phase + stopPropagation keeps plotly's own touch handling from
    // fighting the pinch (it would pan with one of the two fingers).
    event.preventDefault();
    event.stopPropagation();
    const distance = touchDistance(event.touches);

    if (distance > 0) {
      const centerX = (event.touches[0].clientX + event.touches[1].clientX) / 2;
      const centerY = (event.touches[0].clientY + event.touches[1].clientY) / 2;
      zoomAtClientPoint(centerX, centerY, pinchDistance / distance);
      pinchDistance = distance;
      host.onPinch?.();
    }
  };

  const handleTouchEnd = (event: TouchEvent) => {
    if (event.touches.length < 2 && pinchDistance !== null) {
      pinchDistance = null;
      // Stamp gesture end too: a pinch held still before lifting would
      // otherwise let plotly's synthetic click through.
      host.onPinch?.();
    }
  };

  element.addEventListener('wheel', handleWheel, { passive: false });
  element.addEventListener('touchstart', handleTouchStart, { capture: true, passive: true });
  element.addEventListener('touchmove', handleTouchMove, { capture: true, passive: false });
  element.addEventListener('touchend', handleTouchEnd, { capture: true, passive: true });
  element.addEventListener('touchcancel', handleTouchEnd, { capture: true, passive: true });

  return () => {
    element.removeEventListener('wheel', handleWheel);
    element.removeEventListener('touchstart', handleTouchStart, { capture: true });
    element.removeEventListener('touchmove', handleTouchMove, { capture: true });
    element.removeEventListener('touchend', handleTouchEnd, { capture: true });
    element.removeEventListener('touchcancel', handleTouchEnd, { capture: true });
  };
};
