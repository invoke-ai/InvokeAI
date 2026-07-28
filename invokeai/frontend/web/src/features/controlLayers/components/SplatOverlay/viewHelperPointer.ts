/** The fixed corner-viewport size (layout px) hardcoded as `dim` inside three's ViewHelper. */
export const VIEW_HELPER_DIM = 128;

export type ElementBox = {
  /** getBoundingClientRect() of the renderer's canvas — visual coords, includes any ancestor CSS scale. */
  rect: { left: number; top: number; right: number; bottom: number; width: number; height: number };
  /** Layout (untransformed) size of the canvas element. */
  offsetWidth: number;
  offsetHeight: number;
};

/**
 * three's ViewHelper.handleClick derives its raycast NDC by mixing visual coordinates
 * (getBoundingClientRect, event.clientX/Y) with layout sizes (offsetWidth/offsetHeight and its fixed
 * 128px `dim`). Those spaces only agree when the element is unscaled — but the splat viewport is
 * CSS-scaled by the canvas stage transform, so at any stage zoom other than 100% the gizmo's click
 * hotspot drifts off the drawn gizmo and axis clicks miss.
 *
 * This inverts ViewHelper's formula: it returns clientX/Y such that handleClick's mixed-space math
 * yields the NDC of the pointer's true layout-space position. At scale 1 it is the identity.
 */
export const remapPointerForViewHelper = (
  box: ElementBox,
  clientX: number,
  clientY: number,
  dim: number = VIEW_HELPER_DIM
): { clientX: number; clientY: number } => {
  const { rect, offsetWidth, offsetHeight } = box;
  const scaleX = rect.width / (offsetWidth || 1) || 1;
  const scaleY = rect.height / (offsetHeight || 1) || 1;
  // Pointer position in the element's layout space — where the click "really" is on the gizmo.
  const layoutX = (clientX - rect.left) / scaleX;
  const layoutY = (clientY - rect.top) / scaleY;
  // ViewHelper's mixed-space gizmo origin and viewport extent (its exact internal expressions).
  const offsetX = rect.left + (offsetWidth - dim);
  const offsetY = rect.top + (offsetHeight - dim);
  return {
    clientX: offsetX + ((rect.right - offsetX) * (layoutX - (offsetWidth - dim))) / dim,
    clientY: offsetY + ((rect.bottom - offsetY) * (layoutY - (offsetHeight - dim))) / dim,
  };
};
