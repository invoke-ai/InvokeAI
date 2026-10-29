/**
 * Move pointer focus into the canvas subtree so nonfocusable canvas targets do not leave tool hotkeys routed to
 * another widget.
 */

/**
 * Elements that own their own focus/caret and must NOT be focus-stolen by the
 * surface: the text tool's contenteditable overlay and any inline input the
 * chrome overlays on the surface.
 */
export const INLINE_EDIT_SELECTOR = 'input, textarea, select, [contenteditable="true"], [role="textbox"]';

/**
 * Preserve focus already inside the surface or inline editors. Refocusing during capture would blur-commit before
 * the engine can swallow click-away, creating a stray text session.
 */
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
