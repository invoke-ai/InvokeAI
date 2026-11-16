/**
 * The value of `attribute` on the element inside `root` whose rect holds the point, as a pointer drag library's
 * collision finds a droppable: whatever is painted above the zone, and with the zone itself click-through.
 */
export const findDropZoneAt = (
  root: Element,
  attribute: string,
  point: { clientX: number; clientY: number }
): string | null => {
  for (const zone of root.querySelectorAll(`[${attribute}]`)) {
    const rect = zone.getBoundingClientRect();
    if (
      point.clientX >= rect.left &&
      point.clientX < rect.right &&
      point.clientY >= rect.top &&
      point.clientY < rect.bottom
    ) {
      return zone.getAttribute(attribute);
    }
  }
  return null;
};
