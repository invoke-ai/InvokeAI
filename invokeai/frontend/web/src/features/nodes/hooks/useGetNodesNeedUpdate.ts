import { createSelector } from '@reduxjs/toolkit';
import { useAppSelector } from 'app/store/storeHooks';
import { selectNodesSlice } from 'features/nodes/store/nodesSlice';
import { selectNodeTemplatesSlice } from 'features/nodes/store/nodeTemplatesSlice';
import { isInvocationNode } from 'features/nodes/types/invocation';
import { getNeedsUpdate } from 'features/nodes/util/node/nodeUpdate';

const selector = createSelector(selectNodesSlice, selectNodeTemplatesSlice, (nodes, nodeTemplates) =>
  nodes.nodes.filter(isInvocationNode).some((node) => {
    const template = nodeTemplates.templates[node.data.type];
    if (!template) {
      return false;
    }
    return getNeedsUpdate(node, template);
  })
);

export const useGetNodesNeedUpdate = () => {
  const getNeedsUpdate = useAppSelector(selector);
  return getNeedsUpdate;
};
