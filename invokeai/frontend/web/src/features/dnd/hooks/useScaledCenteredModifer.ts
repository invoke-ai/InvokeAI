import type { Modifier } from '@dnd-kit/core';
import { getEventCoordinates } from '@dnd-kit/utilities';
import { createSelector } from '@reduxjs/toolkit';
import { stateSelector } from 'app/store/store';
import { useAppSelector } from 'app/store/storeHooks';
import { activeTabNameSelector } from 'features/ui/store/uiSelectors';
import { useCallback } from 'react';

const selectZoom = createSelector(
  [stateSelector, activeTabNameSelector],
  ({ nodes }, activeTabName) =>
    activeTabName === 'nodes' ? nodes.viewport.zoom : 1
);

/**
 * Applies scaling to the drag transform (if on node editor tab) and centers it on cursor.
 */
export const useScaledModifer = () => {
  const zoom = useAppSelector(selectZoom);
  const modifier: Modifier = useCallback(
    ({ activatorEvent, draggingNodeRect, transform }) => {
      if (draggingNodeRect && activatorEvent) {
        const activatorCoordinates = getEventCoordinates(activatorEvent);

        if (!activatorCoordinates) {
          return transform;
        }

        const offsetX = activatorCoordinates.x - draggingNodeRect.left;
        const offsetY = activatorCoordinates.y - draggingNodeRect.top;

        const x = transform.x + offsetX - draggingNodeRect.width / 2;
        const y = transform.y + offsetY - draggingNodeRect.height / 2;
        const scaleX = transform.scaleX * zoom;
        const scaleY = transform.scaleY * zoom;

        return {
          x,
          y,
          scaleX,
          scaleY,
        };
      }

      return transform;
    },
    [zoom]
  );

  return modifier;
};
