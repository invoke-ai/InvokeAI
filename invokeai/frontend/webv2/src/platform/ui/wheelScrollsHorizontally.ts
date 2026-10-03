import type { RefCallback } from 'react';

// One wheel notch in `deltaMode` lines, matching what browsers scroll for it.
const LINE_HEIGHT_PX = 16;

const handleWheel = (event: WheelEvent): void => {
  const element = event.currentTarget;
  if (!(element instanceof HTMLElement) || event.deltaX !== 0 || event.deltaY === 0) {
    return;
  }
  if (element.scrollWidth <= element.clientWidth) {
    return;
  }
  event.preventDefault();
  const delta = event.deltaMode === WheelEvent.DOM_DELTA_LINE ? event.deltaY * LINE_HEIGHT_PX : event.deltaY;
  // Wheeling down travels toward the strip's end, which a right-to-left strip reaches at negative `scrollLeft`.
  element.scrollLeft += getComputedStyle(element).direction === 'rtl' ? -delta : delta;
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
