import type { Modifier } from '@dnd-kit/core';
import { getEventCoordinates } from '@dnd-kit/utilities';
import { useStore } from '@nanostores/react';
import { useAppSelector } from 'app/store/storeHooks';
import { $viewport } from 'features/nodes/store/nodesSlice';
import { selectActiveTab } from 'features/ui/store/uiSelectors';
import { useCallback } from 'react';

/**
 * Applies scaling to the drag transform (if on node editor tab) and centers it on cursor.
 */
export const useScaledModifer = () => {
  const activeTabName = useAppSelector(selectActiveTab);
  const workflowsViewport = useStore($viewport);
  const modifier: Modifier = useCallback(
    ({ activatorEvent, draggingNodeRect, transform }) => {
      if (draggingNodeRect && activatorEvent) {
        const zoom = activeTabName === 'workflows' ? workflowsViewport.zoom : 1;
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
    [activeTabName, workflowsViewport.zoom]
  );

  return modifier;
};
