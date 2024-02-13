import type { CollisionDetection } from '@dnd-kit/core';
import { pointerWithin } from '@dnd-kit/core';

/**
 * Filters out droppable elements that are overflowed, then applies the pointerWithin collision detection.
 *
 * Fixes collision detection firing on droppables that are not visible, having been scrolled out of view.
 *
 * See https://github.com/clauderic/dnd-kit/issues/1198
 */
export const customPointerWithin: CollisionDetection = (arg) => {
  if (!arg.pointerCoordinates) {
    // sanity check
    return [];
  }

  // Get all elements at the pointer coordinates. This excludes elements which are overflowed,
  // so it won't include the droppable elements that are scrolled out of view.
  const targetElements = document.elementsFromPoint(arg.pointerCoordinates.x, arg.pointerCoordinates.y);

  const filteredDroppableContainers = arg.droppableContainers.filter((container) => {
    if (!container.node.current) {
      return false;
    }
    // Only include droppable elements that are in the list of elements at the pointer coordinates.
    return targetElements.includes(container.node.current);
  });

  // Run the provided collision detection with the filtered droppable elements.
  return pointerWithin({
    ...arg,
    droppableContainers: filteredDroppableContainers,
  });
};
