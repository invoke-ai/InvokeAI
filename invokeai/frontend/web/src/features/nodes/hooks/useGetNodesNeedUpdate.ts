import { useStore } from '@nanostores/react';
import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { $templates } from 'features/nodes/store/nodesSlice';
import { selectNodes } from 'features/nodes/store/selectors';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { getNeedsUpdate } from 'features/nodes/util/node/nodeUpdate';
import { useMemo } from 'react';

export const useGetNodesNeedUpdate = () => {
  const templates = useStore($templates);
  const selector = useMemo(
    () =>
      createSelector(selectNodes, (nodes) =>
        nodes.some((node) => {
          if (!isInvocationNode(node)) {
            return false; // Invocation nodes do not need updates
          }
          const template = templates[node.data.type];
          if (!template) {
            return false;
          }
          return getNeedsUpdate(node.data, template);
        })
      ),
    [templates]
  );
  const needsUpdate = useAppSelector(selector);
  return needsUpdate;
};
