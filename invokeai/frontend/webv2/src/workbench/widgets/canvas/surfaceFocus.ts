/**
 * Pure predicate for the canvas surface's focus-on-pointerdown behavior, kept
 * free of React/Chakra so it is node-testable (see `surfaceFocus.test.ts`).
 *
 * The two `<canvas>` render targets aren't focusable, so a plain click never
 * moves DOM focus — it stays wherever it was (e.g. a layers-panel button in a
 * DIFFERENT widget), and the hotkey runtime's DOM-focus-based widget resolution
 * (`getHotkeyTargetWidget`) keeps routing tool hotkeys (`b`/`e`/…) to that other
 * widget. Focusing the surface container on pointerdown moves focus into the
 * canvas widget's subtree so its hotkeys fire immediately.
 */

/**
 * Elements that own their own focus/caret and must NOT be focus-stolen by the
 * surface: the text tool's contenteditable overlay and any inline input the
 * chrome overlays on the surface.
 */
export const INLINE_EDIT_SELECTOR = 'input, textarea, select, [contenteditable="true"], [role="textbox"]';

/**
 * Whether a pointerdown on the canvas surface should focus the surface
 * container.
 *
 * - **Focus already inside the surface → no.** Most importantly the text
 *   tool's contenteditable (rendered inside the container by `TextEditPortal`):
 *   clicking the canvas *outside* it is the "click away to commit" gesture,
 *   which the ENGINE owns (`commitOpenTextSession` in its pointerdown, which
 *   then swallows the click). Focusing the container here would fire in the
 *   capture phase — before the engine's bubble-phase handler — synchronously
 *   blurring the editable, committing + nulling the session via its `onBlur`,
 *   so the engine would then see no session, not swallow the click, and open a
 *   stray new text session at the click point. Skipping is also harmless for
 *   hotkey scope: focus inside the surface already resolves to the canvas
 *   widget.
 * - **Click landing in an inline editor → no.** Belt and suspenders for
 *   portal-rendered editors whose DOM may sit outside the container (so the
 *   `contains` check above wouldn't spare them).
 * - **Otherwise → yes** (e.g. focus is on a button in another widget's panel).
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
