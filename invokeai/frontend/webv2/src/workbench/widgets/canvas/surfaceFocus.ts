/** Pointer focus enters the canvas subtree so nonfocusable targets still route tool hotkeys here. */

/** Marks the focusable canvas surface, for chrome that hands focus back to it once it unmounts. */
export const CANVAS_SURFACE_ATTR = 'data-canvas-surface';

/** Elements owning their own caret that the surface must not steal focus from (text overlay, inline inputs). */
export const INLINE_EDIT_SELECTOR = 'input, textarea, select, [contenteditable="true"], [role="textbox"]';

/** Keeps focus inside the surface: refocusing in capture would blur-commit before the engine swallows the click. */
export const shouldFocusCanvasSurface = (
  container: HTMLElement,
  target: EventTarget | null,
  activeElement: Element | null
): boolean => {
  if (activeElement && container.contains(activeElement)) {
    return false;
  }
  if (target instanceof Element && target.closest(INLINE_EDIT_SELECTOR)) {
    return false;
  }
  return true;
};

/** The focusable surface of one canvas widget instance, wherever its chrome renders; null while it is not mounted. */
export const findCanvasSurface = (instanceId: string): HTMLElement | null =>
  document.querySelector<HTMLElement>(
    `[data-hotkey-widget-instance-id="${CSS.escape(instanceId)}"] [${CANVAS_SURFACE_ATTR}]`
  );
