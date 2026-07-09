/**
 * Pure array move used by the layers panel's grouped drag-to-reorder and its
 * context-menu z-arrange commands (see `layerGroups.ts`, which maps a
 * within-group move back onto the single global z-ordered `layers` array).
 *
 * Kept free of dnd-kit and React so the mapping is unit-testable in node.
 */

/** Moves the item at `fromIndex` to `toIndex`, shifting the others. Returns a new array. */
export const moveItem = <T>(items: readonly T[], fromIndex: number, toIndex: number): T[] => {
  const next = [...items];
  const [moved] = next.splice(fromIndex, 1);
  if (moved === undefined) {
    return next;
  }
  next.splice(toIndex, 0, moved);
  return next;
};
