import { CANVAS_TAB_TESTID } from 'features/canvas/store/constants';

/**
 * Determines if an element is a child of the canvas tab. Uses the canvas data-testid,
 * actually checking against the *parent* of that element, which is the canvas's
 * panel from `react-resizable-panels`. This panel element has dynamic children, so
 * it's safer to check the canvas tab and grab its parent.
 */
export const isElChildOfCanvasTab = (el: HTMLElement) => {
  const canvasContainerEl = document.querySelector(`[data-testid="${CANVAS_TAB_TESTID}"]`);
  return Boolean(canvasContainerEl?.parentElement?.contains(el));
};
