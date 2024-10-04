/**
 * Gets whether the item is visible in the root element.
 */

export const getIsVisible = (itemRect: DOMRect, rootRect: DOMRect) => {
  return (
    itemRect.top >= rootRect.top &&
    itemRect.bottom <= rootRect.bottom &&
    itemRect.left >= rootRect.left &&
    itemRect.right <= rootRect.right
  );
};
