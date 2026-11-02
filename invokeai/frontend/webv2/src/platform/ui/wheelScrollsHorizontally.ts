import type { RefCallback } from 'react';

// One wheel notch in `deltaMode` lines, matching what browsers scroll for it.
const LINE_HEIGHT_PX = 16;

const toPixels = (event: WheelEvent, delta: number): number =>
  event.deltaMode === WheelEvent.DOM_DELTA_LINE ? delta * LINE_HEIGHT_PX : delta;

// Wheeling down travels toward the strip's end, which a right-to-left strip reaches at negative `scrollLeft`.
const scrollByVerticalWheel = (strip: HTMLElement, event: WheelEvent): void => {
  const delta = toPixels(event, event.deltaY);
  strip.scrollLeft += getComputedStyle(strip).direction === 'rtl' ? -delta : delta;
};

const handleWheel = (event: WheelEvent): void => {
  const element = event.currentTarget;
  if (!(element instanceof HTMLElement) || event.deltaX !== 0 || event.deltaY === 0) {
    return;
  }
  if (element.scrollWidth <= element.clientWidth) {
    return;
  }
  event.preventDefault();
  scrollByVerticalWheel(element, event);
};

/** Map vertical wheels to horizontal scrolling; preserve horizontal gestures and non-overflowing strips. */
export const wheelScrollsHorizontally: RefCallback<HTMLElement> = (element) => {
  if (!element) {
    return;
  }
  element.addEventListener('wheel', handleWheel, { passive: false });
  return () => {
    element.removeEventListener('wheel', handleWheel);
  };
};

/**
 * Scroll `strip` for a wheel that landed on something laid over it, such as an overflow cue. A vertical wheel maps as
 * `wheelScrollsHorizontally` maps it; a horizontal one, which the browser cannot scroll natively from outside the strip,
 * moves it by its own delta. A strip whose content fits passes the wheel through.
 */
export const forwardWheelToStrip = (strip: HTMLElement, event: WheelEvent): void => {
  if (strip.scrollWidth <= strip.clientWidth || (event.deltaX === 0 && event.deltaY === 0)) {
    return;
  }
  event.preventDefault();
  if (event.deltaX !== 0) {
    // Physical: a positive deltaX moves the view right, which raises `scrollLeft` in either direction.
    strip.scrollLeft += toPixels(event, event.deltaX);
  } else {
    scrollByVerticalWheel(strip, event);
  }
};
